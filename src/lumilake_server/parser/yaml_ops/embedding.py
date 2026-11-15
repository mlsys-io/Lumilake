"""YAML emitter for the embedding op."""

from typing import Any

from .common import EmitContext, build_generation_config


def emit_embedding_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    entry = ctx.entry
    user_id_to_internal = ctx.user_id_to_internal
    config = entry.fields.get("config")
    if not isinstance(config, dict) or "model" not in config:
        raise ValueError(
            f"EmbeddingOp '{entry.id}' requires 'config' with a 'model' field"
        )

    content_ref = entry.fields.get("input")
    if not isinstance(content_ref, str) or not content_ref.strip():
        raise ValueError(
            f"EmbeddingOp '{entry.id}' requires an 'input' reference to a text op "
            "or workflow input id"
        )
    target = content_ref.strip()
    if target not in user_id_to_internal:
        raise ValueError(
            f"EmbeddingOp '{entry.id}' input references unknown id '{target}'"
        )
    content_id = user_id_to_internal[target]

    op_dict["content"] = content_id
    op_dict["config"] = build_generation_config(config, entry.id, op_kind="EmbeddingOp")
    op_dict["cacheable"] = bool(entry.fields.get("cacheable", False))
    op_dict["_inputs"] = [content_id]
