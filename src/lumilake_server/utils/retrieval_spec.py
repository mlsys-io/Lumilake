"""The ``data_spec`` of a ``DataRetrievalOp`` and its row-order rule.

A SQL result has no defined row order without an ``ORDER BY``, so a spec that
may return several rows from one parameter row must declare one, or declare
``rows: one`` for a lookup that returns at most one.
"""

import re
from typing import Any, Self

import sqlparse
from pydantic import (
    ConfigDict,
    StrictBool,
    StrictInt,
    ValidationError,
    ValidationInfo,
    model_validator,
)
from sqlparse.sql import Parenthesis, Token, TokenList
from sqlparse.tokens import Comment, Keyword, Punctuation

from lumilake_server.data_specs import RetrievalData

ROWS_ONE = "one"

_ORDER_BY = "ORDER BY"

_ORDER_BY_SPACING = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)

_CHECK_ROW_ORDER = "check_row_order"


class RetrievalSpec(RetrievalData):
    """The user-authored ``data_spec`` of a ``DataRetrievalOp``, with its
    row-order rule and any other key the user wrote for the worker."""

    model_config = ConfigDict(extra="allow")

    sample_value: Any = None
    verify: StrictBool | None = None
    max_steps: StrictInt | None = None

    @model_validator(mode="after")
    def _row_order(self, info: ValidationInfo) -> Self:
        if not (info.context or {}).get(_CHECK_ROW_ORDER):
            return self
        if self.mode != "sql" or self.rows is not None or self.template is None:
            return self
        if sql_has_top_level_order_by(self.template):
            return self
        raise ValueError(
            "a mode: sql retrieval returns rows in no defined order without a "
            "top-level ORDER BY. End the query with an ORDER BY that breaks ties "
            "(for example on the primary key), or declare 'rows: one' when each "
            "parameter row returns at most one row"
        )

    @classmethod
    def load(cls, data: Any, where: str, check_order: bool = True) -> Self:
        """Validate a ``data_spec``, naming ``where`` and the field on failure."""
        try:
            return cls.model_validate(data, context={_CHECK_ROW_ORDER: check_order})
        except ValidationError as exc:
            raise ValueError(f"{where}: {problem_text(exc)}") from None


def problem_text(exc: ValidationError) -> str:
    """The first problem of a failed validation, prefixed with its field."""
    problem = exc.errors()[0]
    field = ".".join(str(part) for part in problem["loc"])
    if problem["type"] in {"model_type", "dict_type"}:
        message = "must be a mapping"
    else:
        message = problem["msg"].removeprefix("Value error, ")
    return f"{field}: {message}" if field else message


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
    """Whether a SQL query orders its own final result.

    Only the last statement's outer query counts, and so does a query wrapped
    whole in parentheses; an ``ORDER BY`` inside a subquery, a CTE or an
    aggregate argument does not.
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
    """The indexes of the retrieval items that returned more than one row."""
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
