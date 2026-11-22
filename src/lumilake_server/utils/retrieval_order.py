"""Submit-time checks for a ``DataRetrievalOp``'s row order.

A SQL result has no defined row order without an ``ORDER BY``, so a spec that
returns several rows from one parameter row must declare one. A lookup that
returns at most one row per parameter row declares ``rows: one`` instead; the
runtime then checks that bound when it reads the result.
"""

import re
from typing import Any

import sqlparse
from sqlparse.sql import Parenthesis, Token, TokenList
from sqlparse.tokens import Comment, Keyword, Punctuation

ROWS_ONE = "one"

ROWS_KEY = "rows"

_ORDER_BY = "ORDER BY"

_ORDER_BY_SPACING = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)


def row_order_problem(data_spec: dict[str, Any]) -> str | None:
    """Describe the first problem a retrieval spec has with its row order.

    Returns ``None`` when the declared ``rows`` value is valid and the spec's
    order is either defined or does not matter.
    """
    rows = data_spec.get(ROWS_KEY)
    if rows is not None and rows != ROWS_ONE:
        return f"'rows' must be '{ROWS_ONE}' when declared (got {rows!r})"
    if data_spec.get("mode") != "sql" or rows == ROWS_ONE:
        return None
    template = data_spec.get("template")
    if not isinstance(template, str):
        return None
    if sql_has_top_level_order_by(template):
        return None
    return (
        "a mode: sql retrieval returns rows in no defined order without a "
        "top-level ORDER BY. End the query with an ORDER BY that breaks ties "
        "(for example on the primary key), or declare 'rows: one' when each "
        "parameter row returns at most one row"
    )


def _carrying_tokens(tokens: list[Token]) -> list[Token]:
    """The statement's tokens with whitespace, comments and ``;`` removed."""
    return [
        token
        for token in tokens
        if not (
            token.is_whitespace
            or token.ttype in Comment
            or (token.ttype in Punctuation and token.value == ";")
        )
    ]


def sql_has_top_level_order_by(query: str) -> bool:
    """Whether a SQL query orders its final result.

    The last statement's outer query counts, and so does a query wrapped whole
    in parentheses. An ``ORDER BY`` inside a subquery, a CTE, or an
    aggregate's argument does not count: sqlparse groups those under a
    ``Parenthesis`` or a function call, and the scan stops there.

    Two rewrites make the scan see the keyword where a reader would: one
    regular expression collapses whitespace inside ``ORDER BY`` itself, since
    sqlparse's ``WHERE`` group only closes on the exact keyword and would
    otherwise swallow an ``ORDER BY`` written across lines, and comments are
    stripped so a trailing comment is not mistaken for another statement.
    """
    collapsed = _ORDER_BY_SPACING.sub(_ORDER_BY, query)
    stripped = sqlparse.format(collapsed, strip_comments=True)
    statements = [s for s in sqlparse.parse(stripped) if _carrying_tokens(s.tokens)]
    if not statements:
        return False
    last: TokenList = statements[-1]
    tokens = _carrying_tokens(last.tokens)
    while len(tokens) == 1 and isinstance(tokens[0], Parenthesis):
        tokens = _carrying_tokens(tokens[0].tokens[1:-1])
    return any(
        token.ttype is Keyword and token.normalized == _ORDER_BY for token in tokens
    )


def multi_row_item_indexes(items: Any) -> list[int]:
    """The indexes of the retrieval items that returned more than one row.

    A ``rows: one`` retrieval fails when this list is non-empty: the declared
    bound failed for some item, so every row that item feeds downstream would
    be silently reduced to its first.
    """
    if not isinstance(items, list):
        return []
    multi: list[int] = []
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        rows = item.get("rows", 0)
        if isinstance(rows, int) and rows > 1:
            multi.append(item.get("index", position))
    return multi
