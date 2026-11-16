"""YAML emitter for the image-generation op."""

from typing import Any

from lumilake_server.parser.common import make_id

from .common import EmitContext, build_generation_config


def emit_image_generation_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    entry = ctx.entry
    user_id_to_internal = ctx.user_id_to_internal
    config = entry.fields.get("config")
    if not isinstance(config, dict) or "model" not in config:
        raise ValueError(
            f"ImageGenerationOp '{entry.id}' requires 'config' with a 'model' field"
        )

    # Either reference an existing FormatOp/content op directly, or provide an
    # inline `content` string that is a bare user-facing id (mirroring how n8n
    # wraps `{{ $('Some Node') }}` prompts through an implicit FormatOp).
    content_ref = entry.fields.get("content")
    content_id: str | None = None
    if isinstance(content_ref, str) and content_ref.strip():
        target = content_ref.strip()
        if target not in user_id_to_internal:
            raise ValueError(
                f"ImageGenerationOp '{entry.id}' content references unknown id "
                f"'{target}'"
            )
        target_internal = user_id_to_internal[target]
        referenced = ctx.graph_ops.get(target_internal)
        if referenced and referenced.get("_op") == "FormatOp":
            content_id = target_internal
        else:
            fmt_id = make_id(ctx.scope, "format", entry.id)
            fmt_op = {
                "_id": fmt_id,
                "_op": "FormatOp",
                "_max_iter": None,
                "_inputs": [target_internal],
                "template": "{ref0}",
                "format_args": [],
                "format_kwargs": {"ref0": target_internal},
            }
            op_dict.setdefault("__implicit_ops__", []).append(fmt_op)
            content_id = fmt_id
    else:
        raise ValueError(
            f"ImageGenerationOp '{entry.id}' requires a 'content' user-id reference"
        )

    op_dict["content"] = content_id
    op_dict["config"] = build_generation_config(
        config, entry.id, op_kind="ImageGenerationOp"
    )
    op_dict["cacheable"] = bool(entry.fields.get("cacheable", False))
    op_dict["_inputs"] = [content_id]
