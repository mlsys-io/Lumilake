"""YAML emitter for the parse op."""

from typing import Any

from lumilake_server.ops.parse_ops import validate_parse

from .common import EmitContext


def emit_parse_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    """Fill in a ``ParseOp`` entry: its format, its failed-row policy, and the
    pattern, required keys and flags that format takes."""
    entry = ctx.entry
    form = entry.fields.get("format", "json")
    pattern = entry.fields.get("pattern")
    require = entry.fields.get("require")
    flags = entry.fields.get("flags")
    on_fail = entry.fields.get("on_fail", "marker")
    unknown = set(entry.fields) - {"format", "pattern", "require", "flags", "on_fail"}
    if unknown:
        raise ValueError(f"ParseOp '{entry.id}' has unknown fields: {sorted(unknown)}")
    validate_parse(form, pattern, require, flags, on_fail, f"ParseOp '{entry.id}'")
    op_dict["format"] = form
    op_dict["on_fail"] = on_fail
    if pattern is not None:
        op_dict["pattern"] = pattern
    if require is not None:
        op_dict["require"] = require
    if flags is not None:
        op_dict["flags"] = flags
