"""YAML emitter for the keyed join op."""

from typing import Any

from .common import EmitContext


def emit_join_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    """Fill in a join's key path, attach targets and missing-field policy."""
    entry = ctx.entry
    user_id_to_internal = ctx.user_id_to_internal
    by = entry.fields.get("by")
    if not isinstance(by, str) or not by.strip():
        raise ValueError(f"JoinOp '{entry.id}' requires 'by', a dotted field name")
    if not entry.inputs:
        raise ValueError(f"JoinOp '{entry.id}' requires at least one input")
    attach = entry.fields.get("attach", {})
    if not isinstance(attach, dict):
        raise ValueError(
            f"JoinOp '{entry.id}' attach must map a field name to an op id"
        )
    # ``missing: null`` spells the default in YAML.
    missing = entry.fields.get("missing")
    if missing is None:
        missing = "null"
    if missing not in ("null", "error"):
        raise ValueError(
            f"JoinOp '{entry.id}' missing must be 'null' or 'error', got {missing!r}"
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
    op_dict["by"] = by.split(".")
    op_dict["attach"] = resolved
    op_dict["missing"] = missing
