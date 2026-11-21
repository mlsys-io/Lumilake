"""The ``data_spec`` of a ``DataRetrievalOp`` and its SQL row-order rule."""

import re
from typing import Any, Self

import sqlparse
from pydantic import (
    ConfigDict,
    StrictBool,
    StrictInt,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from sqlparse.sql import Parenthesis, Token, TokenList
from sqlparse.tokens import Comment, Keyword, Punctuation

from lumilake_server.data_specs import RetrievalData

ROWS_ONE = "one"

ROWS_TABLE = "table"

ROWS_KEY = "rows"

ROWS_VALUES = (ROWS_ONE, ROWS_TABLE)

_ORDER_BY = "ORDER BY"

_ORDER_BY_SPACING = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)

_CHECK_ROW_ORDER = "check_row_order"


class RetrievalSpec(RetrievalData):
    """The user-authored ``data_spec`` of a ``DataRetrievalOp``."""

    model_config = ConfigDict(extra="allow")

    sample_value: Any = None
    verify: StrictBool | None = None
    max_steps: StrictInt | None = None

    @field_validator("rows", mode="before")
    @classmethod
    def _rows_declared(cls, value: Any) -> Any:
        if value is not None and value not in ROWS_VALUES:
            raise ValueError(
                f"'{ROWS_KEY}' must be '{ROWS_ONE}' or '{ROWS_TABLE}' when declared"
                f" (got {value!r})"
            )
        return value

    @model_validator(mode="after")
    def _row_order(self, info: ValidationInfo) -> Self:
        if not (info.context or {}).get(_CHECK_ROW_ORDER):
            return self
        if self.rows is not None and self.mode != "sql":
            raise ValueError(
                f"'{ROWS_KEY}' is declared only on a mode: sql retrieval"
                f" (got mode {self.mode!r})"
            )
        if self.mode != "sql" or self.rows is not None or self.template is None:
            return self
        if sql_has_top_level_order_by(self.template):
            return self
        raise ValueError(
            "a mode: sql retrieval returns rows in no defined order without a "
            "top-level ORDER BY. End the query with an ORDER BY that breaks ties "
            "(for example on the primary key), declare 'rows: one' when each "
            "parameter row returns at most one row, or declare 'rows: table' when "
            "the whole result is one value"
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
    """Whether the last statement's outer query, unwrapped, ends in ``ORDER BY``."""
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
