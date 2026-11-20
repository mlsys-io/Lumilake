"""``JoinOp``: merge row columns and 1:1 outputs back onto keyed rows.

The first input fixes the output: one row per first-input row, in its order.
Every further input is merged into the row carrying the same key, and each
``attach`` entry joins one 1:1 op's per-row output to the row it was computed
from. An attach target's row source is resolved when the runtime graph is
built, so no user code positions values; a field a row is missing is either
left out or fails the job.
"""

import re
from typing import Any

from lumilake_server.ops.ops import Op

JOIN_OP = "JoinOp"

_FIELD_RE = re.compile(r"^\w+$")
_PATH_RE = re.compile(r"^\w+(\.\w+)*$")
_MISSING = ("null", "error")


@Op.registry.register(JOIN_OP)
class JoinOp(Op):
    """A keyed join: one row per first-input row, further inputs merged by
    ``by``, and ``attach`` outputs joined to the rows they were computed from."""

    column_indices: list[int]
    attach_indices: dict[str, int]
    by: list[str]
    missing: str

    def __init__(
        self,
        inputs: list[Op],
        by: str | list[str],
        attach: dict[str, Op] | None = None,
        missing: str = "null",
    ) -> None:
        if not inputs:
            raise ValueError("JoinOp requires at least one input")
        attach = {} if attach is None else attach
        self.by = _as_path(by)
        if missing not in _MISSING:
            raise ValueError(
                f"JoinOp missing must be one of {_MISSING}, got {missing!r}"
            )
        self.missing = missing
        joined: list[Op] = []
        for op in inputs:
            if op not in joined:
                joined.append(op)
        self.attach_indices = {}
        target_indices = set()
        for field, target in attach.items():
            _validate_field(field)
            if target not in joined:
                raise ValueError(
                    f"JoinOp attach field {field!r} reads '{target.id}', which"
                    " is not one of the op's inputs; list it there so its"
                    " producer is built before the join"
                )
            self.attach_indices[field] = joined.index(target)
            target_indices.add(self.attach_indices[field])
        self.column_indices = [
            index for index in range(len(joined)) if index not in target_indices
        ]
        if not self.column_indices:
            raise ValueError("JoinOp needs a row column besides its attach targets")
        super().__init__(joined)

    @property
    def columns(self) -> list[Op]:
        """The joined row columns, in declaration order."""
        return [self.inputs[index] for index in self.column_indices]

    @property
    def column_ids(self) -> list[str]:
        """The joined row columns' op ids, in declaration order."""
        return [op.id for op in self.columns]

    @property
    def attach_ops(self) -> dict[str, Op]:
        """The attach targets, keyed by the field they fill."""
        return {
            field: self.inputs[index] for field, index in self.attach_indices.items()
        }

    @property
    def row_source(self) -> Op:
        """The first input fixes the output: one row per its row."""
        return self.columns[0]

    def _serialize(self) -> dict[str, Any]:
        return {
            "by": self.by,
            "attach": {
                field: self.inputs[index].id
                for field, index in self.attach_indices.items()
            },
            "missing": self.missing,
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "JoinOp":
        joined = [_built_before(other_ops, name, "input") for name in data["_inputs"]]
        attach = {
            field: _built_before(other_ops, op_id, f"attach field {field!r}")
            for field, op_id in data["attach"].items()
        }
        return cls(joined, data["by"], attach, data["missing"])


def _built_before(other_ops: dict[str, "Op"], op_id: str, role: str) -> "Op":
    """The op a join reads, which the graph payload must carry before it."""
    op = other_ops.get(op_id)
    if op is None:
        raise ValueError(
            f"JoinOp {role} references '{op_id}', which is not built before it;"
            " list it in the op's inputs"
        )
    return op


def _as_path(by: str | list[str]) -> list[str]:
    """The key path as parts, refusing anything but a dotted field name."""
    parts = by.split(".") if isinstance(by, str) else by.copy()
    if not parts or not all(isinstance(part, str) for part in parts):
        raise ValueError(f"JoinOp by must be a dotted field name, got {by!r}")
    if not _PATH_RE.match(".".join(parts)):
        raise ValueError(f"JoinOp by must be a dotted field name, got {by!r}")
    return parts


def _validate_field(field: str) -> None:
    """Refuse an attach name that is not a single field of the joined row."""
    if not _FIELD_RE.match(field):
        raise ValueError(f"JoinOp attach field must be a field name, got {field!r}")
