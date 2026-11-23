"""The ``ParseOp``: turn one input value per row into a structured record.

A row it cannot parse emits a failed-record marker instead of failing the
task, or keeps its text when ``on_fail`` is ``raw``.
"""

import re
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictStr,
    field_validator,
    model_validator,
)

from lumilake_server.data_specs import ParseDescription
from lumilake_server.ops.data_ops import DataOp
from lumilake_server.ops.ops import FunctionalOp, Op
from lumilake_server.ops.spec_validation import validate_fields

PARSE_OP = "ParseOp"
MARKER: Literal["marker", "raw"] = "marker"
FLAGS = "ism"
_FLAG_VALUES = set(FLAGS)


class ParseSpec(BaseModel):
    """The YAML `ParseOp` entry fields, built by every source of the op."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["json", "regex"] = "json"
    on_fail: Literal["marker", "raw"] = MARKER
    pattern: StrictStr | None = None
    require: list[StrictStr] | None = None
    flags: StrictStr | None = None

    @field_validator("require")
    @classmethod
    def _field_names(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and (not value or not all(value)):
            raise ValueError("must be a non-empty list of field names")
        return value

    @field_validator("flags")
    @classmethod
    def _flag_letters(cls, value: str | None) -> str | None:
        if value is not None and set(value) - _FLAG_VALUES:
            raise ValueError(f"must be a string of {list(FLAGS)}")
        return value

    @model_validator(mode="after")
    def _format_fields(self) -> Self:
        if self.format == "json":
            if self.pattern is not None:
                raise ValueError("takes 'pattern' only with format: regex")
            if self.flags is not None:
                raise ValueError("takes 'flags' only with format: regex")
            return self
        if self.require is not None:
            raise ValueError("takes 'require' only with format: json")
        if not self.pattern:
            raise ValueError("format: regex requires a non-empty 'pattern'")
        try:
            compiled = re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(f"pattern does not compile: {exc}") from exc
        if not compiled.groupindex:
            raise ValueError(
                "pattern must name at least one group, such as (?P<name>...)"
            )
        return self


def build_parse_spec(fields: dict[str, Any], where: str) -> ParseSpec:
    """Validate one `ParseOp` entry, naming the op and field in the error."""
    return validate_fields(ParseSpec, fields, where)


@Op.registry.register(PARSE_OP)
class ParseOp(FunctionalOp):
    """Parse each row's value with ``format`` or ``pattern``."""

    spec: ParseSpec

    def __init__(self, inputs: list[str] | list[Op] | Op, spec: ParseSpec) -> None:
        if isinstance(inputs, (Op, str)):
            inputs = [inputs]
        if len(inputs) != 1:
            raise ValueError(f"ParseOp takes exactly one input, got {len(inputs)}")
        input_ops = [
            (
                inp
                if isinstance(inp, Op)
                else DataOp([inp] if isinstance(inp, str) else inp)
            )
            for inp in inputs
        ]
        super().__init__(input_ops)
        self.spec = spec

    @property
    def row_source(self) -> Op:
        return self.inputs[0]

    @property
    def emits_markers(self) -> bool:
        """Whether an unparsable row becomes a failed-record marker."""
        return self.spec.on_fail == MARKER

    @property
    def parse_spec(self) -> ParseDescription:
        """The parse step's description of this op."""
        return ParseDescription(
            format=self.spec.format,
            on_fail=self.spec.on_fail,
            pattern=self.spec.pattern,
            require=self.spec.require,
            flags=self.spec.flags,
        )

    def _serialize(self) -> dict[str, Any]:
        return self.spec.model_dump(exclude_none=True)

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "ParseOp":
        fields = {key: value for key, value in data.items() if not key.startswith("_")}
        spec = build_parse_spec(fields, f"ParseOp '{data.get('_id', '?')}'")
        return cls([other_ops[inp] for inp in data["_inputs"]], spec)


def parse_op(inputs: list[str] | list[Op] | Op, spec: ParseSpec) -> ParseOp:
    return ParseOp(inputs, spec)
