"""A ``DataRetrievalOp``'s declared record key.

``key: {column: <name>}`` names the column whose value is each emitted row's
key, read at ``items.table.<column>``. The declaration is checked where the op
is built; the keys of each arrived result in ``retrieval_key_check``.
"""

from typing import Any

KEY_FIELD = "key"

COLUMN_FIELD = "column"


def retrieval_key_problem(data_spec: dict[str, Any]) -> str | None:
    """The first problem with a declared ``key``, or ``None`` if it is
    well-formed."""
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
    """The column that holds each emitted row's record key, if declared."""
    key = data_spec.get(KEY_FIELD)
    return key[COLUMN_FIELD] if isinstance(key, dict) else None
