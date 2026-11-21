"""The ``ParseOp``: turn one input value per row into a structured record.

A row it cannot parse emits a failed-record marker instead of failing the
task, or keeps its text when ``on_fail`` is ``raw``.
"""

import re
from typing import Any

from lumilake_server.ops.data_ops import DataOp
from lumilake_server.ops.ops import FunctionalOp, Op

PARSE_OP = "ParseOp"
FORMATS = ("json", "regex")
FLAGS = "ism"
ON_FAIL = ("marker", "raw")
MARKER = ON_FAIL[0]


def validate_parse(
    form: Any,
    pattern: Any,
    require: Any,
    flags: Any,
    on_fail: Any,
    where: str,
) -> None:
    """Check a parse spec, naming the op in ``where``."""
    if form not in FORMATS:
        raise ValueError(f"{where} format must be one of {FORMATS}, got {form!r}")
    if on_fail not in ON_FAIL:
        raise ValueError(f"{where} on_fail must be one of {ON_FAIL}, got {on_fail!r}")
    if require is not None and (
        not isinstance(require, list)
        or not require
        or not all(isinstance(key, str) and key for key in require)
    ):
        raise ValueError(
            f"{where} require must be a non-empty list of field names, got"
            f" {require!r}"
        )
    if form == "json":
        if pattern is not None:
            raise ValueError(f"{where} takes 'pattern' only with format: regex")
        if flags:
            raise ValueError(f"{where} takes 'flags' only with format: regex")
        return
    if require is not None:
        raise ValueError(f"{where} takes 'require' only with format: json")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError(f"{where} format: regex requires a non-empty 'pattern'")
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"{where} pattern does not compile: {exc}") from exc
    if not compiled.groupindex:
        raise ValueError(
            f"{where} pattern must name at least one group, such as (?P<name>...)"
        )
    if not isinstance(flags, str | None) or set(flags or "") - set(FLAGS):
        raise ValueError(
            f"{where} flags must be a string of {list(FLAGS)}, got {flags!r}"
        )


@Op.registry.register(PARSE_OP)
class ParseOp(FunctionalOp):
    """Parse each row's value with ``format`` or ``pattern``."""

    def __init__(
        self,
        inputs: list[str] | list[Op] | Op,
        format: str = "json",
        pattern: str | None = None,
        require: list[str] | None = None,
        flags: str | None = None,
        on_fail: str = MARKER,
    ) -> None:
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
        validate_parse(format, pattern, require, flags, on_fail, PARSE_OP)
        super().__init__(input_ops)
        self.format = format
        self.pattern = pattern
        self.require = require
        self.flags = flags
        self.on_fail = on_fail

    @property
    def row_source(self) -> Op:
        return self.inputs[0]

    @property
    def emits_markers(self) -> bool:
        """Whether an unparsable row becomes a failed-record marker."""
        return self.on_fail == MARKER

    @property
    def parse_spec(self) -> dict[str, Any]:
        """The fields the generated step reads."""
        spec: dict[str, Any] = {"format": self.format, "on_fail": self.on_fail}
        if self.pattern is not None:
            spec["pattern"] = self.pattern
        if self.require is not None:
            spec["require"] = self.require
        if self.flags is not None:
            spec["flags"] = self.flags
        return spec

    def _serialize(self) -> dict[str, Any]:
        return self.parse_spec

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "ParseOp":
        input_ops = [other_ops[inp] for inp in data["_inputs"]]
        return cls(
            inputs=input_ops,
            format=data["format"],
            pattern=data.get("pattern"),
            require=data.get("require"),
            flags=data.get("flags"),
            on_fail=data.get("on_fail", MARKER),
        )


def parse_op(
    inputs: list[str] | list[Op] | Op,
    format: str = "json",
    pattern: str | None = None,
    require: list[str] | None = None,
    flags: str | None = None,
    on_fail: str = MARKER,
) -> ParseOp:
    return ParseOp(inputs, format, pattern, require, flags, on_fail)
