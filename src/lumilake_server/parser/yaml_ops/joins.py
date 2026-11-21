"""YAML emitter for the keyed join op."""

import re
from typing import Any

from .common import EmitContext

_PATH_RE = re.compile(r"^\w+(\.\w+)*$")


def emit_join_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    """Fill in a join's key paths, attach targets and missing-field policy."""
    entry = ctx.entry
    user_id_to_internal = ctx.user_id_to_internal
    by = _key_paths(entry.fields.get("by"), entry.id)
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
    op_dict["by"] = by
    op_dict["attach"] = resolved
    op_dict["missing"] = missing


def _key_paths(by: Any, entry_id: str) -> list[str]:
    """The join key as its paths: one dotted field name, or a list of them for
    a composite key."""
    paths = [by] if isinstance(by, str) else by
    if not isinstance(paths, list) or not paths:
        raise ValueError(
            f"JoinOp '{entry_id}' requires 'by', a dotted field name or a list"
            " of them"
        )
    for path in paths:
        if not isinstance(path, str) or not _PATH_RE.match(path):
            raise ValueError(
                f"JoinOp '{entry_id}' by must be dotted field names, got {path!r}"
            )
    return paths
