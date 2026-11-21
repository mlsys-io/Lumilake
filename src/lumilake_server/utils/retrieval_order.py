"""Submit-time checks for a ``DataRetrievalOp``'s row order.

A SQL result has no defined row order without an ``ORDER BY``, so a spec that
returns several rows from one parameter row must declare one. A lookup that
returns at most one row per parameter row declares ``rows: one`` instead; the
runtime then checks that bound when it reads the result.
"""

import sqlparse
from sqlparse.sql import Parenthesis, TokenList
from sqlparse.tokens import Keyword

ROWS_ONE = "one"

ROWS_KEY = "rows"

_ORDER_BY = "ORDER BY"


def order_by_problem(data_spec: dict[str, object]) -> str | None:
    """Describe why a retrieval spec lacks a deterministic row order, or
    return ``None`` when its order is defined or does not matter."""
    mode = data_spec.get("mode")
    if mode not in {"sql", "agent"}:
        return None
    if data_spec.get(ROWS_KEY) == ROWS_ONE:
        return None
    template = data_spec.get("template") if mode == "sql" else None
    if not isinstance(template, str):
        return None
    if sql_has_top_level_order_by(template):
        return None
    return (
        f"a mode: {mode} retrieval returns rows in no defined order without a "
        "top-level ORDER BY. End the query with an ORDER BY that breaks ties "
        "(for example on the primary key), or declare 'rows: one' when each "
        "parameter row returns at most one row"
    )


def validate_rows(value: object) -> str | None:
    """Reject a ``rows`` value other than ``"one"``."""
    if value is None or value == ROWS_ONE:
        return None
    return f"'rows' must be '{ROWS_ONE}' when declared (got {value!r})"


def sql_has_top_level_order_by(query: str) -> bool:
    """Whether a SQL query orders its final result.

    The last statement's outer query counts, and so does a query wrapped whole
    in parentheses. An ``ORDER BY`` inside a subquery, a CTE, or an
    aggregate's argument is not a check: sqlparse groups those under a
    ``Parenthesis`` or a function call, and the scan stops there.
    """
    statements = [s for s in sqlparse.parse(query) if s.value.strip()]
    if not statements:
        return False
    statement: TokenList = statements[-1]
    if (
        len(statement.tokens) == 1
        and isinstance(statement.tokens[0], Parenthesis)
        and isinstance(statement.tokens[0], TokenList)
    ):
        statement = statement.tokens[0]
    return any(
        token.ttype is Keyword and token.normalized == _ORDER_BY
        for token in statement.tokens
    )
