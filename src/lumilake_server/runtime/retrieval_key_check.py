"""The arrival check of a retrieval's declared record key.

A keyed ``DataRetrievalOp``'s result is checked as its task's response
arrives: a row without the key column, a row whose key is null and a repeated
key each fail the job, naming the op. A task carries one slice of a
parametrized run's rows, and the slices are not compared with each other.
"""

from typing import Any

from lumilake_server.runtime.lambda_task_helpers import serialized_table, table_rows
from lumilake_server.runtime.record_store import key_text


def key_problem(op_id: str, items: Any, column: str) -> str | None:
    """Describe the first problem one retrieval task's result has with its
    declared key, or ``None`` when every row of it carries a distinct,
    non-null value in ``column``."""
    if not isinstance(items, list):
        return None
    seen: dict[str, tuple[int, int | None]] = {}
    for item_position, item in enumerate(items):
        item_index = _item_index(item, item_position)
        rows = _item_rows(item)
        if rows is None:
            return (
                f"DataRetrievalOp '{op_id}' declares key {column!r}, but item"
                f" {item_index} of this batch has a table that could not be"
                " read as a column table"
            )
        for row_position, row in enumerate(rows):
            where = f"{_position(row_position, item_index)} of this batch"
            if column not in row:
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
            key = key_text(value)
            if key in seen:
                first_row, first_item = seen[key]
                return (
                    f"DataRetrievalOp '{op_id}' declares key {column!r}, but the"
                    f" key {key!r} is repeated at {_position(first_row, first_item)}"
                    f" and {_position(row_position, item_index)} of this batch;"
                    " one retrieval emits one record per key"
                )
            seen[key] = (row_position, item_index)
    return None


def _position(row_position: int, item_index: Any) -> str:
    """One row of a task's batch, as the check reports it."""
    return f"row {row_position} of item {item_index}"


def _item_index(item: Any, position: int) -> Any:
    """The index a retrieval item reports for itself, else its position."""
    if isinstance(item, dict) and "index" in item:
        return item["index"]
    return position


def _item_rows(item: Any) -> list[dict[str, Any]] | None:
    """The rows one retrieval item's ``table`` carries, in row order, or
    ``None`` when the item carries a ``table`` that is not a column table."""
    if not isinstance(item, dict):
        return []
    if "table" not in item:
        return []
    table = serialized_table(item["table"])
    return None if table is None else table_rows(table)
