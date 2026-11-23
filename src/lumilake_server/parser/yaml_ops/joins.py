"""YAML emitter for the keyed join op."""

from typing import Any

from lumilake_server.ops.join_ops import build_join_spec

from .common import EmitContext


def emit_join_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    """Fill in a join's key path, attach targets and missing-field policy."""
    entry = ctx.entry
    user_id_to_internal = ctx.user_id_to_internal
    if not entry.inputs:
        raise ValueError(f"JoinOp '{entry.id}' requires at least one input")
    spec = build_join_spec(
        {
            "by": entry.fields.get("by"),
            "attach": entry.fields.get("attach") or {},
            # ``missing: null`` spells the default in YAML.
            "missing": entry.fields.get("missing") or "null",
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
    op_dict["attach"] = resolved
    op_dict["missing"] = spec.missing
