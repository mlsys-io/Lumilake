"""YAML emitters for the data-plane ops."""

from collections.abc import Mapping
from typing import Any

from lumilake_server.utils.retrieval_spec import RetrievalParam, RetrievalSpec

from .common import EmitContext


def emit_data_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    entry = ctx.entry
    data = entry.fields.get("data")
    if not isinstance(data, list) or not all(isinstance(v, str) for v in data):
        raise ValueError(f"DataOp '{entry.id}' requires 'data: list[str]'")
    op_dict["data"] = list(data)
    op_dict["_inputs"] = []  # DataOps have no inputs


def emit_data_retrieval_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    entry = ctx.entry
    data_spec = entry.fields.get("data_spec")
    if data_spec is None:
        raise ValueError(f"DataRetrievalOp '{entry.id}' requires 'data_spec'")
    spec = RetrievalSpec.load(
        data_spec, f"DataRetrievalOp '{entry.id}' data_spec", check_order=False
    )
    op_dict["data_spec"] = _resolve_param_nodes(spec, entry.id, ctx.user_id_to_internal)


def _resolve_param_nodes(
    spec: RetrievalSpec, entry_id: str, user_id_to_internal: Mapping[str, str]
) -> dict[str, Any]:
    """Point each param's user-facing node id at its internal op id."""
    params: list[RetrievalParam] = []
    for i, param in enumerate(spec.params):
        if param.node is None:
            params.append(param)
            continue
        if param.node not in user_id_to_internal:
            raise ValueError(
                f"DataRetrievalOp '{entry_id}' data_spec.params[{i}].node "
                f"references unknown input/op id '{param.node}'"
            )
        params.append(
            param.model_copy(update={"node": user_id_to_internal[param.node]})
        )
    return spec.model_copy(update={"params": params}).model_dump(exclude_none=True)


def emit_message_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    entry = ctx.entry
    user_id_to_internal = ctx.user_id_to_internal
    messages = entry.fields.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"MessageOp '{entry.id}' requires non-empty 'messages'")
    serialized: list[dict[str, str]] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"MessageOp '{entry.id}' messages[{i}] must be a mapping")
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(
                f"MessageOp '{entry.id}' messages[{i}] requires string role + content"
            )
        # Allow referencing an upstream op by id in `content` — mirror n8n parser
        # behaviour, which stores the op id directly in the serialized message.
        resolved_content = user_id_to_internal.get(content, content)
        serialized.append({"role": role, "content": resolved_content})
    op_dict["messages"] = serialized


def emit_format_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    entry = ctx.entry
    user_id_to_internal = ctx.user_id_to_internal
    template = entry.fields.get("template")
    if not isinstance(template, str):
        raise ValueError(f"FormatOp '{entry.id}' requires 'template: str'")

    fmt_args = entry.fields.get("format_args", []) or []
    fmt_kwargs = entry.fields.get("format_kwargs", {}) or {}

    if not isinstance(fmt_args, list) or not all(isinstance(a, str) for a in fmt_args):
        raise ValueError(f"FormatOp '{entry.id}' format_args must be list[str]")
    if not isinstance(fmt_kwargs, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in fmt_kwargs.items()
    ):
        raise ValueError(f"FormatOp '{entry.id}' format_kwargs must be dict[str, str]")

    resolved_args = [user_id_to_internal.get(a, a) for a in fmt_args]
    resolved_kwargs = {k: user_id_to_internal.get(v, v) for k, v in fmt_kwargs.items()}

    op_dict["template"] = template
    op_dict["format_args"] = resolved_args
    op_dict["format_kwargs"] = resolved_kwargs
