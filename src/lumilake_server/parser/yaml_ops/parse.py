"""YAML emitter for the parse op."""

from typing import Any

from lumilake_server.ops.parse_ops import validate_parse

from .common import EmitContext


def emit_parse_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    """Fill in a ``ParseOp`` entry from its YAML fields."""
    entry = ctx.entry
    form = entry.fields.get("format", "json")
    pattern = entry.fields.get("pattern")
    require = entry.fields.get("require")
    flags = entry.fields.get("flags")
    unknown = set(entry.fields) - {"format", "pattern", "require", "flags"}
    if unknown:
        raise ValueError(f"ParseOp '{entry.id}' has unknown fields: {sorted(unknown)}")
    validate_parse(form, pattern, require, flags, f"ParseOp '{entry.id}'")
    op_dict["format"] = form
    if pattern is not None:
        op_dict["pattern"] = pattern
    if require is not None:
        op_dict["require"] = require
    if flags is not None:
        op_dict["flags"] = flags
