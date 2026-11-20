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


def _decoded_cell(value: Any) -> Any:
    """Decode one cell of a column: JSON object or array text becomes the
    object or list it holds, and any other value passes through."""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            return value
        if isinstance(decoded, (dict, list)):
            return decoded
    return value


def _table_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    """A column-major table, a mapping from row index to cell per column, as
    one row per index in index order."""
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


def _decoded_rows(value: Any) -> list[Any]:
    """The rows one decoded cell carries: a table's ``{"df": "<json>"}``
    envelope, a row object, or a list of any of these."""
    if (
        isinstance(value, dict)
        and list(value) == ["df"]
        and isinstance(value["df"], str)
    ):
        try:
            table = json.loads(value["df"])
        except ValueError:
            raise ValueError("join input df is not JSON")
        if not isinstance(table, dict):
            raise ValueError("join input df is not a column table")
        return _table_rows(table)
    if isinstance(value, list):
        rows: list[Any] = []
        for item in value:
            rows.extend(_decoded_rows(_decoded_cell(item)))
        return rows
    return [value]


def _row_content(row: Any) -> Any:
    try:
        return row["json"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return row.get("text", row)


def _serialized_table(value: Any) -> dict[str, Any] | None:
    """The ``column -> row index -> value`` mapping of a serialized table, or
    ``None`` when ``value`` is not one, either as a ``{"df": <json text>}``
    mapping or as the JSON text of one."""
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
    """Decode one column value for a per-row or per-group function: JSON object
    or array text is parsed, a serialized table becomes its rows (also when a
    retrieval item carries it under ``table``), and anything else is passed as
    is."""
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
    """The per-call rows of a decoded column set: a one-row column applies to
    every row, and every other column must carry the same count."""
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
    return [
        text(i["output"]) if isinstance(i, dict) and "output" in i else text(i)
        for i in items
    ]
