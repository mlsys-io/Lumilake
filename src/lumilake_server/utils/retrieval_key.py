"""A ``DataRetrievalOp``'s declared record key.

``key: {column: <name>}`` names the column whose value is each emitted row's
record key. The key is declared once, where the rows are produced, and a
consumer reads it at ``items.table.<column>``; without the declaration a row
has no logical key. The declaration is checked where the op is built: a key
that is not a non-empty column, or that a non-``sql`` mode cannot carry, is
refused. The keys of each arrived result are checked separately, in
``retrieval_key_check``.
"""

from typing import Any

from lumilake_server.data_specs import DataSpecModel, RetrievalData


def retrieval_key_problem(data_spec: dict[str, Any]) -> str | None:
    """The first problem with a declared ``key``, or ``None`` when it is sound."""
    key = data_spec.get("key")
    if key is None:
        return None
    if not isinstance(key, dict):
        return f"key must be a mapping of column, got {key!r}"
    if set(key) != {"column"}:
        return f"key takes only a column field, got {sorted(key)}"
    column = key["column"]
    if not isinstance(column, str) or not column.strip():
        return f"key column must be a non-empty string, got {column!r}"
    mode = data_spec.get("mode")
    if mode in (None, "sql"):
        return None
    return (
        f"key is refused for mode {mode!r}; only a sql retrieval emits one row per"
        " table row, so only its rows carry a column as their key"
    )


def retrieval_key_column(data_spec: DataSpecModel | dict[str, Any]) -> str | None:
    """The column holding each row's record key, when the spec declares one."""
    if isinstance(data_spec, RetrievalData):
        return None if data_spec.key is None else data_spec.key.column
    if isinstance(data_spec, dict):
        key = data_spec.get("key")
        return key["column"] if isinstance(key, dict) else None
    return None
