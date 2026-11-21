"""YAML emitters for the data-plane ops."""

from typing import Any

from lumilake_server.utils.retrieval_key import retrieval_key_problem

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
    if not isinstance(data_spec, dict):
        raise ValueError(f"DataRetrievalOp '{entry.id}' requires 'data_spec'")
    problem = retrieval_key_problem(data_spec, entry.id)
    if problem is not None:
        raise ValueError(f"DataRetrievalOp '{entry.id}' {problem}")
    op_dict["data_spec"] = _resolve_data_spec(
        data_spec, entry.id, ctx.user_id_to_internal
    )


def _resolve_data_spec(
    data_spec: dict[str, Any],
    entry_id: str,
    user_id_to_internal: dict[str, str],
    *,
    op_label: str = "DataRetrievalOp",
    field_label: str = "data_spec",
) -> dict[str, Any]:
    """Resolve ``params[*].node`` refs in a spec.

    ``op_label`` and ``field_label`` are threaded through every error
    message so the caller sees the op and field they actually authored.
    """
    resolved = dict(data_spec)
    params = resolved.get("params")
    if params is None:
        return resolved
    if not isinstance(params, list):
        raise ValueError(f"{op_label} '{entry_id}' {field_label}.params must be a list")
    new_params: list[dict[str, Any]] = []
    for i, param in enumerate(params):
        if not isinstance(param, dict):
            raise ValueError(
                f"{op_label} '{entry_id}' {field_label}.params[{i}] must be a mapping"
            )
        resolved_param = dict(param)
        if "node" in resolved_param:
            node_ref = resolved_param["node"]
            if not isinstance(node_ref, str):
                raise ValueError(
                    f"{op_label} '{entry_id}' {field_label}.params[{i}].node "
                    "must be a string"
                )
            if node_ref not in user_id_to_internal:
                raise ValueError(
                    f"{op_label} '{entry_id}' {field_label}.params[{i}].node "
                    f"references unknown input/op id '{node_ref}'"
                )
            resolved_param["node"] = user_id_to_internal[node_ref]
        new_params.append(resolved_param)
    resolved["params"] = new_params
    return resolved


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
