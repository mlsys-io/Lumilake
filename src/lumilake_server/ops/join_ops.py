"""``JoinOp``: merge keyed row columns and 1:1 outputs onto the rows they came from."""

import re
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)

from lumilake_server.data_specs import JoinCardinality, JoinHow
from lumilake_server.ops.ops import Op
from lumilake_server.ops.spec_validation import validate_fields

JOIN_OP = "JoinOp"

_FIELD_RE = re.compile(r"^\w+$")
_PATH_RE = re.compile(r"^\w+(\.\w+)*$")


class JoinSpec(BaseModel):
    """The YAML `JoinOp` entry fields, built by every source of the op."""

    model_config = ConfigDict(extra="ignore")

    by: list[StrictStr]
    row_key: list[StrictStr] | None = None
    how: JoinHow = JoinHow.LEFT
    require_match: StrictBool = False
    cardinality: JoinCardinality = JoinCardinality.ONE_TO_ONE
    attach: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    on_missing: Literal["null_fill", "fail_row"] = "null_fill"

    @model_validator(mode="before")
    @classmethod
    def _dotted_paths(cls, data: Any) -> Any:
        """Take each key as its paths: one dotted field name, or several."""
        if not isinstance(data, dict):
            return data
        paths_by_field = data.copy()
        for field in ("by", "row_key"):
            value = data.get(field)
            if value is None and field == "row_key":
                continue
            paths = [value] if isinstance(value, str) else value
            if not isinstance(paths, list) or not paths:
                raise ValueError(
                    f"requires '{field}', a dotted field name or a list of them"
                )
            for path in paths:
                if not isinstance(path, str) or not _PATH_RE.match(path):
                    raise ValueError(
                        f"{field} must be dotted field names, got {path!r}"
                    )
            paths_by_field[field] = paths
        return paths_by_field

    @field_validator("attach")
    @classmethod
    def _field_names(cls, value: dict[str, str]) -> dict[str, str]:
        for field in value:
            if not _FIELD_RE.match(field):
                raise ValueError(f"key {field!r} must be a field name")
        return value

    @model_validator(mode="after")
    def _row_key_names_every_by_path(self) -> Self:
        """The first-input record key, refused unless it names every ``by``."""
        if self.row_key is not None:
            missing = [path for path in self.by if path not in self.row_key]
            if missing:
                raise ValueError(
                    f"row_key must name every by path, got {self.row_key!r} without"
                    f" {missing!r}"
                )
        return self


def build_join_spec(fields: dict[str, Any], where: str) -> JoinSpec:
    """Validate one `JoinOp` entry, naming the op and field in the error."""
    return validate_fields(JoinSpec, fields, where)


@Op.registry.register(JOIN_OP)
class JoinOp(Op):
    """A keyed join by ``by``, attaches joined back, an optional ``row_key`` per row."""

    column_indices: list[int]
    attach_indices: dict[str, int]

    def __init__(
        self,
        inputs: list[Op],
        spec: JoinSpec,
        attach: dict[str, Op] | None = None,
    ) -> None:
        if not inputs:
            raise ValueError("JoinOp requires at least one input")
        attach = {} if attach is None else attach
        self.spec = spec
        joined: list[Op] = []
        for op in inputs:
            if op not in joined:
                joined.append(op)
        super().__init__(joined)
        self.attach_indices = {}
        target_indices = set()
        for field, target in attach.items():
            if target not in joined:
                raise ValueError(
                    f"JoinOp attach field {field!r} reads '{target.id}', which is not"
                    " one of the op's inputs; list it there so its producer is built"
                    " before the join"
                )
            self.attach_indices[field] = joined.index(target)
            target_indices.add(self.attach_indices[field])
        self.column_indices = [
            index for index in range(len(joined)) if index not in target_indices
        ]
        if not self.column_indices:
            raise ValueError("JoinOp needs a row column besides its attach targets")

    @property
    def by(self) -> list[str]:
        return self.spec.by

    @property
    def row_key(self) -> list[str] | None:
        """The record key of the first input's rows, when several share a key."""
        return self.spec.row_key

    @property
    def how(self) -> JoinHow:
        return self.spec.how

    @property
    def require_match(self) -> bool:
        return self.spec.require_match

    @property
    def cardinality(self) -> JoinCardinality:
        return self.spec.cardinality

    @property
    def on_missing(self) -> Literal["null_fill", "fail_row"]:
        return self.spec.on_missing

    @property
    def columns(self) -> list[Op]:
        """The row columns, in declaration order."""
        return [self.inputs[index] for index in self.column_indices]

    @property
    def column_ids(self) -> list[str]:
        """The row columns' op ids, in declaration order."""
        return [op.id for op in self.columns]

    @property
    def attach_ops(self) -> dict[str, Op]:
        """The attach targets, keyed by the field they fill."""
        return {
            field: self.inputs[index] for field, index in self.attach_indices.items()
        }

    @property
    def row_source(self) -> Op:
        """The first input, whose rows fix the output."""
        return self.columns[0]

    def _serialize(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "by": self.by,
            "attach": {
                field: self.inputs[index].id
                for field, index in self.attach_indices.items()
            },
            "how": self.how,
            "require_match": self.require_match,
            "cardinality": self.cardinality,
            "on_missing": self.on_missing,
        }
        if self.row_key is not None:
            data["row_key"] = self.row_key
        return data

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, Op]) -> Self:
        spec = build_join_spec(data, f"JoinOp '{data['_id']}'")
        joined = [_built_before(other_ops, name, "input") for name in data["_inputs"]]
        attach = {
            field: _built_before(other_ops, op_id, f"attach field {field!r}")
            for field, op_id in data["attach"].items()
        }
        return cls(joined, spec, attach)


def _built_before(other_ops: dict[str, Op], op_id: str, role: str) -> Op:
    """The op a join reads, from the ops the payload already built."""
    op = other_ops.get(op_id)
    if op is None:
        raise ValueError(
            f"JoinOp {role} references '{op_id}', which is not built before it; list it"
            " in the op's inputs"
        )
    return op
