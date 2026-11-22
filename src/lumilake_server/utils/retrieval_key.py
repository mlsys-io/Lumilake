"""A ``DataRetrievalOp``'s declared record key.

``key: {column: <name>}`` names the column whose value is each emitted row's
record key. The key is declared once, where the rows are produced, and a
consumer reads it at ``items.table.<column>``; without the declaration a row
has no logical key. The declaration is checked when the result arrives: a null
key, a duplicate key and a missing key column each fail the job, naming the op.
"""

import json
from typing import Any

KEY_FIELD = "key"

COLUMN_FIELD = "column"


def retrieval_key_problem(data_spec: dict[str, Any], op_id: str) -> str | None:
    """Describe the first problem with a declared ``key``, or ``None`` when it
    is absent or well-formed."""
    key = data_spec.get(KEY_FIELD)
    if key is None:
        return None
    if not isinstance(key, dict):
        return f"key must be a mapping of column, got {key!r}"
    if set(key) != {COLUMN_FIELD}:
        return f"key takes only a column field, got {sorted(key)}"
    column = key[COLUMN_FIELD]
    if not isinstance(column, str) or not column.strip():
        return f"key column must be a non-empty string, got {column!r}"
    mode = data_spec.get("mode")
    if mode in (None, "sql"):
        return None
    return (
        f"key is refused for mode {mode!r}; only a sql retrieval emits one row"
        " per table row, so only its rows carry a column as their key"
    )


def retrieval_key_column(data_spec: dict[str, Any]) -> str | None:
    """The column that holds each emitted row's record key, when the spec
    declares ``key: {column: <name>}``."""
    key = data_spec.get(KEY_FIELD)
    return key[COLUMN_FIELD] if isinstance(key, dict) else None


def key_problem(op_id: str, items: Any, column: str) -> str | None:
    """Describe the first problem a keyed retrieval's result has with its
    declared key, or ``None`` when every row carries a distinct, non-null
    value in ``column``."""
    if not isinstance(items, list):
        return None
    seen: dict[str, tuple[int, int | None]] = {}
    for item_position, item in enumerate(items):
        rows = _item_rows(item)
        item_index = _item_index(item, item_position)
        for row_position, row in enumerate(rows):
            where = f"row {row_position} of item {item_index}"
            if not isinstance(row, dict) or column not in row:
                return (
                    f"DataRetrievalOp '{op_id}' declares key {column!r}, but"
                    f" {where} has no such column"
                )
            value = row[column]
            if value is None:
                return (
                    f"DataRetrievalOp '{op_id}' declares key {column!r}, but"
                    f" {where} has a null key; a record key is never null"
                )
            key = _key_text(op_id, column, value)
            if key in seen:
                first_row, first_item = seen[key]
                return (
                    f"DataRetrievalOp '{op_id}' declares key {column!r}, but the"
                    f" key {key!r} is repeated at row {first_row} of item"
                    f" {first_item} and at {where}; one retrieval emits one"
                    " record per key"
                )
            seen[key] = (row_position, item_index)
    return None


def _key_text(op_id: str, column: str, value: Any) -> str:
    """A key value as the text a repeat is reported and compared by: a string
    key is its own text, and any other JSON value is written canonically so
    ``1`` and ``"1"`` stay distinct keys."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        raise ValueError(
            f"DataRetrievalOp '{op_id}' declares key {column!r}, but the value"
            f" {value!r} is not a JSON value"
        ) from None


def _item_index(item: Any, position: int) -> Any:
    """The index a retrieval item reports for itself, else its position."""
    if isinstance(item, dict) and "index" in item:
        return item["index"]
    return position


def _item_rows(item: Any) -> list[Any]:
    """The rows one retrieval item carries: its ``table``, as a column table
    either bare or in its ``{"df": <json text>}`` envelope, with a table
    written as text decoded."""
    if not isinstance(item, dict):
        return []
    table = _decoded(item.get("table"))
    if isinstance(table, dict) and set(table) == {"df"}:
        table = _decoded(table["df"])
    if isinstance(table, dict):
        return _column_rows(table)
    if isinstance(table, list):
        return table
    return []


def _decoded(value: Any) -> Any:
    """A JSON object or array written as text, as the value it holds; any
    other value passes through."""
    if not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except ValueError:
        return value
    return decoded


def _column_rows(table: dict[str, Any]) -> list[Any]:
    """A column-major table, each column mapping a row index to its cell, as
    one row per index in index order."""
    rows: dict[str, dict[str, Any]] = {}
    for name, column in table.items():
        if not isinstance(column, dict):
            return []
        for index, cell in column.items():
            rows.setdefault(index, {})[name] = cell
    try:
        indexes = sorted(rows, key=int)
    except ValueError:
        indexes = sorted(rows)
    return [rows[index] for index in indexes]
