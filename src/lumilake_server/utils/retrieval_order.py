"""The ``data_spec`` of a ``DataRetrievalOp`` and its row-order rule.

A SQL result has no defined row order without an ``ORDER BY``, so a spec that
may return several rows from one parameter row must declare one, or declare
``rows: one`` for a lookup that returns at most one.
"""

import re
from typing import Any, Literal, Self

import sqlparse
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from sqlparse.sql import Parenthesis, Token, TokenList
from sqlparse.tokens import Comment, Keyword, Punctuation

ROWS_ONE = "one"

ROWS_KEY = "rows"

_ORDER_BY = "ORDER BY"

_ORDER_BY_SPACING = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)

_CHECK_ROW_ORDER = "check_row_order"


class RetrievalParam(BaseModel):
    """One template slot: a literal ``data`` list or an upstream ``node``."""

    model_config = ConfigDict(extra="allow")

    label: StrictStr | None = None
    node: StrictStr | None = None
    path: StrictStr | None = None
    data: dict[StrictStr, Any] | None = None
    input_ref: dict[StrictStr, Any] | None = None


class RetrievalSpec(BaseModel):
    """The ``data_spec`` block of a ``DataRetrievalOp``."""

    model_config = ConfigDict(extra="allow")

    type: Literal["lumid"] = "lumid"
    mode: Literal["sql", "s3", "agent"]
    rows: Literal["one"] | None = None
    template: StrictStr | None = None
    description: StrictStr | None = None
    params: list[RetrievalParam] = []
    table: StrictStr | None = None
    output_format: StrictStr | None = None
    encoding: StrictStr | None = None
    schema_scope: StrictStr | None = None
    max_steps: StrictInt | None = None
    model: StrictStr | None = None
    verify: StrictBool | None = None
    sample_value: Any = None

    @field_validator("rows", mode="before")
    @classmethod
    def _rows_one(cls, value: Any) -> Any:
        if value is not None and value != ROWS_ONE:
            raise ValueError(
                f"'{ROWS_KEY}' must be '{ROWS_ONE}' when declared (got {value!r})"
            )
        return value

    @model_validator(mode="after")
    def _row_order(self, info: ValidationInfo) -> Self:
        if not (info.context or {}).get(_CHECK_ROW_ORDER):
            return self
        if self.mode != "sql" or self.rows == ROWS_ONE or self.template is None:
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
    def load(cls, data: Any, where: str) -> Self:
        """Validate a user-authored ``data_spec``, naming ``where`` on failure."""
        try:
            return cls.model_validate(data, context={_CHECK_ROW_ORDER: True})
        except ValidationError as exc:
            raise ValueError(f"{where}: {problem_text(exc)}") from None

    @classmethod
    def parse(cls, data: Any, where: str) -> Self:
        """Convert a spec built in process, naming ``where`` on a shape error."""
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"{where}: {problem_text(exc)}") from None


def problem_text(exc: ValidationError) -> str:
    """The first problem of a failed validation, prefixed with its field."""
    problem = exc.errors()[0]
    field = ".".join(str(part) for part in problem["loc"])
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
