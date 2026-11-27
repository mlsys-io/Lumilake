"""YAML emitter for the keyed join op."""

from typing import Any

from lumilake_server.ops.join_ops import build_join_spec

from .common import EmitContext


def emit_join_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    """Fill in a join's keys, how, match rule, attach targets and per-row policy."""
    entry = ctx.entry
    user_id_to_internal = ctx.user_id_to_internal
    if not entry.inputs:
        raise ValueError(f"JoinOp '{entry.id}' requires at least one input")
    spec = build_join_spec(
        {
            "by": entry.fields.get("by"),
            "row_key": entry.fields.get("row_key"),
            "how": entry.fields.get("how") or "left",
            "require_match": entry.fields.get("require_match", False),
            "cardinality": entry.fields.get("cardinality") or "one_to_one",
            "attach": entry.fields.get("attach") or {},
            # ``on_missing: null_fill`` spells the default in YAML.
            "on_missing": entry.fields.get("on_missing") or "null_fill",
        },
        f"JoinOp '{entry.id}'",
    )
    resolved: dict[str, str] = {}
    for field, target in spec.attach.items():
        if target not in user_id_to_internal:
            raise ValueError(
                f"JoinOp '{entry.id}' attach field {field!r} references unknown id"
                f" {target!r}"
            )
        if target not in entry.inputs:
            raise ValueError(
                f"JoinOp '{entry.id}' attach field {field!r} reads '{target}', which is"
                " not listed in the op's inputs; add it there so its producer is built"
                " before the join"
            )
        resolved[field] = user_id_to_internal[target]
    op_dict["by"] = spec.by
    op_dict["how"] = spec.how
    op_dict["require_match"] = spec.require_match
    op_dict["cardinality"] = spec.cardinality
    if spec.row_key is not None:
        op_dict["row_key"] = spec.row_key
    op_dict["attach"] = resolved
    op_dict["on_missing"] = spec.on_missing
