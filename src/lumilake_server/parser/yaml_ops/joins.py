"""YAML emitter for the keyed join op."""

from typing import Any

from lumilake_server.ops.join_ops import join_key_paths

from .common import EmitContext


def emit_join_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    """Fill in a join's key paths, attach targets and missing-field policy."""
    entry = ctx.entry
    user_id_to_internal = ctx.user_id_to_internal
    by = join_key_paths(entry.fields.get("by"), entry.id)
    if not entry.inputs:
        raise ValueError(f"JoinOp '{entry.id}' requires at least one input")
    attach = entry.fields.get("attach", {})
    if not isinstance(attach, dict):
        raise ValueError(
            f"JoinOp '{entry.id}' attach must map a field name to an op id"
        )
    # ``on_missing: null_fill`` spells the default in YAML.
    on_missing = entry.fields.get("on_missing")
    if on_missing is None:
        on_missing = "null_fill"
    if on_missing not in ("null_fill", "fail_row"):
        raise ValueError(
            f"JoinOp '{entry.id}' on_missing must be 'null_fill' or 'fail_row',"
            f" got {on_missing!r}"
        )
    resolved: dict[str, str] = {}
    for field, target in attach.items():
        if not isinstance(field, str) or not isinstance(target, str):
            raise ValueError(
                f"JoinOp '{entry.id}' attach must map a field name to an op id"
            )
        if target not in user_id_to_internal:
            raise ValueError(
                f"JoinOp '{entry.id}' attach field {field!r} references unknown"
                f" id {target!r}"
            )
        if target not in entry.inputs:
            raise ValueError(
                f"JoinOp '{entry.id}' attach field {field!r} reads '{target}',"
                " which is not listed in the op's inputs; add it there so its"
                " producer is built before the join"
            )
        resolved[field] = user_id_to_internal[target]
    op_dict["by"] = by
    op_dict["attach"] = resolved
    op_dict["on_missing"] = on_missing
