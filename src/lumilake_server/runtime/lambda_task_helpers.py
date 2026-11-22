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


def _is_marker(value: Any) -> bool:
    """Whether a decoded value is a marker, an object whose only key is
    ``_fail``."""
    return isinstance(value, dict) and list(value) == [FAIL_KEY]


def failure_of(value: Any) -> dict[str, Any] | None:
    """The marker a value carries, or ``None`` when its record is ok."""
    if _is_marker(value):
        return value
    if isinstance(value, str) and FAIL_KEY in value:
        try:
            decoded = json.loads(value)
        except ValueError:
            return None
        if _is_marker(decoded):
            return decoded
    return None


def reject_reserved_key(
    value: Any, op: str, row: int | None = None, originals: tuple[Any, ...] = ()
) -> None:
    """Refuse a user value carrying the reserved ``_fail`` key; a marker the
    step received passes through."""
    if isinstance(value, dict) and FAIL_KEY in value:
        if any(value is original or value == original for original in originals):
            return
        where = f" for row {row}" if row is not None else ""
        raise ValueError(
            f"{op!r}{where} returned an object with the reserved {FAIL_KEY!r}"
            " key; failed records carry that key and user functions must not"
        )


def parse_row(spec: dict[str, Any], text: Any) -> Any:
    """Parse one record's text by a ParseOp's spec, raising ``ValueError`` on
    failure."""
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
    value = _decode_record(body)
    require = spec.get("require")
    if require:
        if not isinstance(value, dict):
            raise ValueError("require needs a JSON object, not a list")
        missing = next((key for key in require if key not in value), None)
        if missing is not None:
            raise ValueError(f"missing key '{missing}'")
    return value


def _decode_record(body: str) -> Any:
    """Decode the JSON record after any prose, which must end the reply."""
    decoder = json.JSONDecoder(strict=False)
    start = next((index for index, char in enumerate(body) if char in "{["), None)
    if start is None:
        raise ValueError("no JSON object")
    try:
        value, end = decoder.raw_decode(body, start)
    except ValueError:
        raise ValueError("no JSON object") from None
    if any(char in "{[" for char in body[end:]):
        raise ValueError("no JSON object")
    return value


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


def _decoded_cell(value: Any) -> Any:
    """JSON object or array text becomes the value it holds; anything else
    passes through."""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            return value
        if isinstance(decoded, (dict, list)):
            return decoded
    return value


def _table_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    """A column-major table as one dict per row index, in index order."""
    rows: dict[str, dict[str, Any]] = {}
    for name, column in table.items():
        if not isinstance(column, dict):
            continue
        for index, value in column.items():
            rows.setdefault(index, {})[name] = value
    try:
        indexes = sorted(rows, key=int)
    except ValueError:
        indexes = list(rows)
    return [rows[index] for index in indexes]


def _decoded_rows(value: Any, where: str) -> list[Any]:
    """The rows a decoded cell carries: a ``{"df": "<json>"}`` table's rows,
    a row object, or a list of these."""
    if (
        isinstance(value, dict)
        and list(value) == ["df"]
        and isinstance(value["df"], str)
    ):
        try:
            table = json.loads(value["df"])
        except ValueError:
            raise ValueError(f"{where}: table is not JSON")
        if not isinstance(table, dict):
            raise ValueError(f"{where}: table is not a column table")
        return _table_rows(table)
    if isinstance(value, list):
        rows: list[Any] = []
        for item in value:
            rows.extend(_decoded_rows(_decoded_cell(item), where))
        return rows
    return [value]


def _row_content(row: Any) -> Any:
    try:
        return row["json"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return row.get("text", row)


def _serialized_table(value: Any) -> dict[str, Any] | None:
    """The ``column -> row index -> value`` table ``value`` holds as a
    ``{"df": <json text>}`` mapping, or ``None``."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if (
        isinstance(value, dict)
        and list(value) == ["df"]
        and isinstance(value["df"], str)
    ):
        try:
            table = json.loads(value["df"])
        except ValueError:
            return None
        if isinstance(table, dict):
            return table
    return None


def _decode(value: Any) -> Any:
    """Decode one column value before the function sees it: JSON text is
    parsed and a serialized table becomes its row dicts."""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            return value
        return _decode(decoded) if isinstance(decoded, (dict, list)) else value
    if isinstance(value, dict):
        table = _serialized_table(value)
        if table is None and "table" in value:
            table = _serialized_table(value["table"])
        if table is not None:
            return _table_rows(table)
    return value


def _rows(
    columns: list[list[Any]], plan: list[dict[str, Any]], op_id: str
) -> list[tuple[Any, ...]]:
    """The per-call rows of a decoded column set: a one-row column broadcasts,
    and the others must agree on the count."""
    n = max((len(c) for c in columns), default=1)
    for entry, col in zip(plan, columns):
        if len(col) not in (1, n):
            raise ValueError(
                f"{op_id}: input {entry.get('node', 'literal')!r} has {len(col)}"
                f" rows; expected 1 or {n}"
            )
    return [tuple(c[0] if len(c) == 1 else c[i] for c in columns) for i in range(n)]


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
