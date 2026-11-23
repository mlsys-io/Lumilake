"""YAML emitter for the parse op."""

from typing import Any

from lumilake_server.ops.parse_ops import build_parse_spec

from .common import EmitContext


def emit_parse_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    """Fill in a ``ParseOp`` entry from its YAML fields."""
    entry = ctx.entry
    spec = build_parse_spec(entry.fields, f"ParseOp '{entry.id}'")
    op_dict.update(spec.model_dump(exclude_none=True))
