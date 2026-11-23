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

from pydantic import ValidationError

from lumilake_server.data_specs import DataSpecModel, RetrievalData, RetrievalKey


def retrieval_key_problem(data_spec: dict[str, Any]) -> str | None:
    """The first problem with a declared ``key``, or ``None`` when it is sound."""
    raw_key = data_spec.get("key")
    if raw_key is None:
        return None
    try:
        key = RetrievalKey.model_validate(raw_key)
    except ValidationError as exc:
        return f"key is not a column: {exc.errors()[0]['msg']}"
    if not key.column.strip():
        return f"key column must be a non-empty string, got {key.column!r}"
    mode = data_spec.get("mode")
    if mode in (None, "sql"):
        return None
    return (
        f"key is refused for mode {mode!r}; only a sql retrieval emits one row per"
        " table row, so only its rows carry a column as their key"
    )


def retrieval_key_column(data_spec: DataSpecModel) -> str | None:
    """The column holding each row's record key, when the spec declares one."""
    key = data_spec.key if isinstance(data_spec, RetrievalData) else None
    return None if key is None else key.column
