"""Worker-side helpers for generated standalone Lambda and reshape tasks."""

import json
import os
from collections.abc import Callable
from typing import Any


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


def _table_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    """A serialized table of ``column -> row index -> value`` as one dict per
    row, in row-index order."""
    rows: dict[str, dict[str, Any]] = {}
    for column, values in table.items():
        if not isinstance(values, dict):
            continue
        for index, value in values.items():
            rows.setdefault(index, {})[column] = value
    try:
        ordered = sorted(rows, key=int)
    except ValueError:
        ordered = list(rows)
    return [rows[index] for index in ordered]


def _decode(value: Any) -> Any:
    """Decode one column value for a per-row or per-group function: JSON object
    or array text is parsed, a serialized table becomes its rows, and anything
    else is passed as is."""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            return value
        return _decode(decoded) if isinstance(decoded, (dict, list)) else value
    if (
        isinstance(value, dict)
        and list(value) == ["df"]
        and isinstance(value["df"], str)
    ):
        try:
            table = json.loads(value["df"])
        except ValueError:
            return value
        if isinstance(table, dict):
            return _table_rows(table)
    return value


def _at_path(value: Any, path: str) -> Any:
    """The value at a dotted path inside one decoded column value."""
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list):
            try:
                index = int(part)
            except ValueError:
                raise ValueError(f"list index {part!r} is not an integer")
            if not 0 <= index < len(value):
                raise ValueError(f"list index {part!r} is out of range")
            value = value[index]
        else:
            raise ValueError(f"missing {part!r}")
    return value


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
    return [
        text(i["output"]) if isinstance(i, dict) and "output" in i else text(i)
        for i in items
    ]
