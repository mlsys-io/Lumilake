"""YAML emitter for the classification op."""

from typing import Any

from lumilake_server.ops.classification_ops import ClassificationConfig

from .common import EmitContext, validate_op_config


def emit_classification_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    entry = ctx.entry
    user_id_to_internal = ctx.user_id_to_internal
    query_ref = entry.fields.get("query")
    candidates_ref = entry.fields.get("candidates")
    config = entry.fields.get("config")
    if not isinstance(config, dict) or not config.get("model"):
        raise ValueError(f"ClassificationOp '{entry.id}' requires config.model")
    if not isinstance(query_ref, str) or query_ref not in user_id_to_internal:
        raise ValueError(f"ClassificationOp '{entry.id}' requires a 'query' reference")
    if not isinstance(candidates_ref, str) or candidates_ref not in user_id_to_internal:
        raise ValueError(
            f"ClassificationOp '{entry.id}' requires a 'candidates' reference"
        )
    validate_op_config(ClassificationConfig, config, entry.id, "ClassificationOp")

    op_dict["query"] = user_id_to_internal[query_ref]
    op_dict["candidates"] = user_id_to_internal[candidates_ref]
    op_dict["config"] = config
    op_dict["_inputs"] = [op_dict["query"], op_dict["candidates"]]
