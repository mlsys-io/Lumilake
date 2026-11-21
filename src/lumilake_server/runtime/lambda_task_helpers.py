"""Worker-side helpers for generated standalone Lambda and reshape tasks."""

import json
import os
import re
from collections.abc import Callable
from typing import Any

FAIL_KEY = "_fail"
_FENCE = re.compile(r"```[\w-]*\n(.*?)\n?```", re.DOTALL)


def fail_marker(op: str, reason: str) -> dict[str, Any]:
    """The value of a failed record: the op that failed and why."""
    return {FAIL_KEY: {"op": op, "reason": reason}}


def failure_of(value: Any) -> dict[str, Any] | None:
    """The marker a value carries, or ``None`` when its record is ok."""
    if isinstance(value, dict) and FAIL_KEY in value:
        return value
    if isinstance(value, str) and FAIL_KEY in value:
        try:
            decoded = json.loads(value)
        except ValueError:
            return None
        if isinstance(decoded, dict) and FAIL_KEY in decoded:
            return decoded
    return None


def reject_reserved_key(
    value: Any, op: str, row: int | None = None, originals: tuple[Any, ...] = ()
) -> None:
    """Refuse a user value that carries the reserved ``_fail`` key, so it is
    never mistaken for a failed record. A value the step received itself is a
    pass-through, not a user value, and is kept."""
    if isinstance(value, dict) and FAIL_KEY in value:
        if any(value is original for original in originals):
            return
        where = f" for row {row}" if row is not None else ""
        raise ValueError(
            f"{op!r}{where} returned an object with the reserved {FAIL_KEY!r}"
            " key; failed records carry that key and user functions must not"
        )


def parse_row(spec: dict[str, Any], text: Any) -> Any:
    """Parse one record's text by a ParseOp's spec; raise a ``ValueError``
    saying why the parse failed. ``json`` drops one code fence and
    any prose before the first ``{`` or ``[``; ``regex`` returns the named
    groups of the first match."""
    if spec["format"] == "regex":
        mask = 0
        for name, flag in (("i", re.IGNORECASE), ("s", re.DOTALL), ("m", re.MULTILINE)):
            if name in (spec.get("flags") or ""):
                mask |= flag
        match = re.compile(spec["pattern"] or "", mask).search(str(text))
        if match is None:
            raise ValueError("pattern did not match")
        return match.groupdict()
    if not isinstance(text, str):
        raise ValueError(f"input {text!r} is not text")
    fenced = _FENCE.search(text)
    body = fenced.group(1) if fenced else text
    try:
        value = json.loads(body, strict=False)
    except ValueError:
        value = _decode_trailing(text)
    require = spec.get("require")
    if require:
        missing = next((key for key in require if not _has_key(value, key)), None)
        if missing is not None:
            raise ValueError(f"missing key '{missing}'")
    return value


def _decode_trailing(text: str) -> Any:
    """Decode the JSON record that starts after any prose, ignoring text after
    it, so ``Here it is: {...} hope that helps`` yields the object."""
    decoder = json.JSONDecoder(strict=False)
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text, index)
        except ValueError:
            continue
        return value
    raise ValueError("no JSON object")


def _has_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value
    return isinstance(value, list) and any(_has_key(item, key) for item in value)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def _walk_path(value: Any, parts: list[str], where: str) -> Any:
    for idx, part in enumerate(parts):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                raise ValueError(f"bad JSON at {where}")
        if (
            isinstance(value, dict)
            and list(value) == ["df"]
            and isinstance(value["df"], str)
        ):
            try:
                table = json.loads(value["df"])
            except ValueError:
                raise ValueError(f"bad JSON at {where}")
            if part not in table:
                raise ValueError(f"missing {part!r} at {where}")
            column = table[part]
            if not isinstance(column, dict):
                raise ValueError(f"missing {part!r} at {where}")
            value = [column[k] for k in sorted(column, key=int)]
            rest = parts[idx + 1 :]
            if rest:
                return [_walk_path(v, rest, where) for v in value]
            return value
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list):
            try:
                index = int(part)
            except (TypeError, ValueError):
                raise ValueError(f"list index must be an integer at {where}")
            if index < 0 or index >= len(value):
                raise ValueError(f"list index out of range at {where}")
            value = value[index]
        else:
            raise ValueError(f"missing {part!r} at {where}")
    return value


def _row_content(row: Any) -> Any:
    try:
        return row["json"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return row.get("text", row)


def _column(
    entry: dict[str, Any], inputs: dict[str, str], text: Callable[[Any], str]
) -> list[str]:
    if entry["kind"] == "literal":
        if "file" in entry:
            with open(entry["file"], encoding="utf-8") as fh:
                values = json.load(fh)
        else:
            values = entry["values"]
        return [text(v) for v in values]
    with open(os.path.join(inputs[entry["node"]], "results.json")) as fh:
        envelope = json.load(fh)
    result = envelope.get("result", envelope)
    items = result.get("items")
    if items is None and isinstance(result.get("value"), dict):
        items = result["value"].get("items")
    if items is None and isinstance(result.get("text"), str):
        items = [{"output": result["text"]}]
    if not isinstance(items, list):
        raise ValueError(f"upstream {entry['node']!r} produced no items")
    path = entry.get("path")
    if path:
        parts = path.split(".")
        return [
            _walk_path(i, parts, f"{entry['node']} item {idx} path {path}")
            for idx, i in enumerate(items)
        ]
    if entry.get("flatten"):
        flat = []
        for i in items:
            if isinstance(i, dict) and "rows" in i:
                for row in i["rows"]:
                    flat.append(text(_row_content(row)))
            else:
                out = i["output"] if isinstance(i, dict) and "output" in i else i
                if isinstance(out, list):
                    flat.extend(text(v) for v in out)
                else:
                    flat.append(text(out))
        return flat
    output = [
        i["output"] if isinstance(i, dict) and "output" in i else i for i in items
    ]
    return [text(v) if failure_of(v) is None else v for v in output]
