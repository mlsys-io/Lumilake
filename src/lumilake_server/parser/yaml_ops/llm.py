"""YAML emitters for the LLM ops."""

from typing import Any

from lumilake_server.ops.router_ops import MessagesColumn, RowGate
from lumilake_server.parser.common import make_id

from .common import EmitContext, OpEntry, build_generation_config, require_str
from .data import emit_message_op


def emit_llm_chat_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    _emit_llm_like_op(op_dict, ctx, op_kind="LLMChatOp")


def emit_llm_completion_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    entry = ctx.entry
    user_id_to_internal = ctx.user_id_to_internal
    config = entry.fields.get("config")
    if not isinstance(config, dict) or not config.get("model"):
        raise ValueError(f"LLMCompletionOp '{entry.id}' requires config.model")
    prompt_ref = entry.fields.get("prompt")
    if not isinstance(prompt_ref, str) or prompt_ref not in user_id_to_internal:
        raise ValueError(
            f"LLMCompletionOp '{entry.id}' requires a prompt reference to an op "
            "or input"
        )
    op_dict["prompt"] = user_id_to_internal[prompt_ref]
    op_dict["config"] = build_generation_config(
        config, entry.id, op_kind="LLMCompletionOp"
    )
    op_dict["cacheable"] = bool(entry.fields.get("cacheable", False))
    op_dict["_inputs"] = [user_id_to_internal[prompt_ref]]


def emit_llm_vision_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    entry = ctx.entry
    user_id_to_internal = ctx.user_id_to_internal
    image_source = entry.fields.get("image_source")
    if not isinstance(image_source, str) or not image_source.strip():
        raise ValueError(
            f"LLMVisionOp '{entry.id}' requires 'image_source' referencing an op id"
        )
    if image_source not in user_id_to_internal:
        raise ValueError(
            f"LLMVisionOp '{entry.id}' image_source references unknown id "
            f"'{image_source}'"
        )
    image_path = entry.fields.get("image_path", "images")
    if not isinstance(image_path, str) or not image_path:
        raise ValueError(f"LLMVisionOp '{entry.id}' image_path must be a string")

    _emit_llm_like_op(op_dict, ctx, op_kind="LLMVisionOp")

    op_dict["image_source"] = user_id_to_internal[image_source]
    op_dict["image_path"] = image_path


def _emit_llm_like_op(
    op_dict: dict[str, Any], ctx: EmitContext, *, op_kind: str
) -> None:
    """Shared core for :class:`LLMChatOp` and :class:`LLMVisionOp`.

    Both op types consume a :class:`MessageOp` and carry a generation ``config``
    plus optional ``aggregate_table`` / ``rowwise_*`` / ``system_messages``
    fields. They differ in:

      * ``_inputs`` composition: ``LLMChatOp`` adds aggregate/rowwise node
        references to ``_inputs`` alongside the :class:`MessageOp`; the n8n
        compiler does *not* do this for :class:`LLMVisionOp`, so we mirror that
        asymmetry here.
      * Extra vision-only fields (``image_source``, ``image_path``) are set by
        the caller after this helper returns.
      * ``model`` requirement: both op kinds require ``config.model``; setting
        ``config.api`` routes the op to an external endpoint but does not relax
        the model requirement, so the workflow spec reads the same either way.
    """
    entry = ctx.entry
    user_id_to_internal = ctx.user_id_to_internal
    config = entry.fields.get("config")
    if not isinstance(config, dict):
        raise ValueError(
            f"{op_kind} '{entry.id}' requires 'config' with a 'model' field"
        )
    if "model" not in config:
        raise ValueError(
            f"{op_kind} '{entry.id}' requires 'config' with a 'model' field"
        )

    prompt_spec = entry.fields.get("prompt")
    messages_ref = entry.fields.get("messages_ref")
    inline_messages = entry.fields.get("messages")

    if messages_ref is not None:
        if not isinstance(messages_ref, str):
            raise ValueError(f"{op_kind} '{entry.id}' messages_ref must be a string id")
        if messages_ref not in user_id_to_internal:
            raise ValueError(
                f"{op_kind} '{entry.id}' messages_ref references unknown id "
                f"'{messages_ref}'"
            )
        msg_internal_id = user_id_to_internal[messages_ref]
        referenced_op = ctx.graph_ops.get(msg_internal_id)
        if referenced_op is None or referenced_op.get("_op") != "MessageOp":
            referenced_type = (
                referenced_op.get("_op") if referenced_op is not None else "<pending>"
            )
            raise ValueError(
                f"{op_kind} '{entry.id}' messages_ref '{messages_ref}' must "
                f"reference a MessageOp, got '{referenced_type}'"
            )
    elif inline_messages is not None:
        prepared_messages, extra_format_ops = _build_inline_messages(
            inline_messages,
            prompt_spec=prompt_spec,
            ctx=ctx,
        )
        for fmt_op in extra_format_ops:
            op_dict.setdefault("__implicit_ops__", []).append(fmt_op)

        msg_internal_id = _queue_inline_message_op(
            op_dict, prepared_messages, extra_format_ops, ctx
        )
    else:
        raise ValueError(
            f"{op_kind} '{entry.id}' requires 'messages' (inline) or 'messages_ref'"
        )

    op_dict["messages"] = msg_internal_id
    op_dict["config"] = build_generation_config(config, entry.id, op_kind=op_kind)
    op_dict["return_history"] = bool(entry.fields.get("return_history", False))
    op_dict["cacheable"] = bool(entry.fields.get("cacheable", False))

    aggregate_table = entry.fields.get("aggregate_table")
    rowwise_columns = entry.fields.get("rowwise_columns")
    resolved_aggregate = _resolve_table_nodes(aggregate_table, user_id_to_internal)
    resolved_rowwise = _resolve_table_nodes(rowwise_columns, user_id_to_internal)

    if "structural_outputs" in entry.fields:
        op_dict["structural_outputs"] = entry.fields["structural_outputs"]
    if resolved_aggregate is not None:
        op_dict["aggregate_table"] = resolved_aggregate
    if "rowwise_template" in entry.fields:
        op_dict["rowwise_template"] = entry.fields["rowwise_template"]
    if resolved_rowwise is not None:
        op_dict["rowwise_columns"] = resolved_rowwise
    rowwise_messages_column = entry.fields.get("rowwise_messages_column")
    if rowwise_messages_column is not None:
        if not isinstance(rowwise_messages_column, dict):
            raise ValueError(
                f"{op_kind} '{entry.id}' rowwise_messages_column must be a mapping"
            )
        column_node = rowwise_messages_column.get("node")
        if not isinstance(column_node, str) or column_node not in user_id_to_internal:
            raise ValueError(
                f"{op_kind} '{entry.id}' rowwise_messages_column requires a known node"
            )
        op_dict["rowwise_messages_column"] = MessagesColumn(
            node=user_id_to_internal[column_node],
            path=require_str(
                rowwise_messages_column, "path", f"{op_kind} '{entry.id}'"
            ),
            label=require_str(
                rowwise_messages_column, "label", f"{op_kind} '{entry.id}'"
            ),
        ).to_dict()
    row_gate = entry.fields.get("row_gate")
    if row_gate is not None:
        if not isinstance(row_gate, dict):
            raise ValueError(f"{op_kind} '{entry.id}' row_gate must be a mapping")
        gate_node = row_gate.get("node")
        if not isinstance(gate_node, str) or gate_node not in user_id_to_internal:
            raise ValueError(f"{op_kind} '{entry.id}' row_gate requires a known node")
        gate = RowGate(
            node=user_id_to_internal[gate_node],
            branch=require_str(row_gate, "branch", f"{op_kind} '{entry.id}' row_gate"),
        )
        if "rowwise_messages_column" in op_dict:
            raise ValueError(
                f"{op_kind} '{entry.id}' combines row_gate with"
                " rowwise_messages_column; the gated path builds its prompts"
                " from rowwise_template, so a messages column would be ignored."
                " Use rowwise_template, or drop the row gate."
            )
        op_dict["row_gate"] = gate.to_dict()
        if gate.node not in op_dict["_inputs"]:
            op_dict["_inputs"].append(gate.node)
    if "system_messages" in entry.fields:
        op_dict["system_messages"] = entry.fields["system_messages"]
    elif "rowwise_template" in entry.fields and isinstance(inline_messages, list):
        sys_msgs = [
            m["content"]
            for m in inline_messages
            if isinstance(m, dict)
            and m.get("role") == "system"
            and isinstance(m.get("content"), str)
        ]
        if sys_msgs:
            op_dict["system_messages"] = sys_msgs

    # For LLMChatOp, n8n adds aggregate/rowwise node refs into `_inputs`.
    # LLMVisionOp keeps `_inputs` as [msg_op] only (those refs flow via
    # rowwise_columns / image_source separately).
    llm_inputs = [msg_internal_id]
    if op_kind == "LLMChatOp":
        for table in (resolved_aggregate, resolved_rowwise):
            if not table:
                continue
            for item in table:
                if not isinstance(item, dict):
                    continue
                node_id = item.get("node")
                if isinstance(node_id, str) and node_id not in llm_inputs:
                    llm_inputs.append(node_id)
        row_gate = op_dict.get("row_gate")
        if isinstance(row_gate, dict):
            gate_node = row_gate.get("node")
            if isinstance(gate_node, str) and gate_node not in llm_inputs:
                llm_inputs.append(gate_node)
    op_dict["_inputs"] = llm_inputs


def _queue_inline_message_op(
    op_dict: dict[str, Any],
    prepared_messages: list[Any],
    extra_format_ops: list[dict[str, Any]],
    ctx: EmitContext,
) -> str:
    """Queue the implicit MessageOp an inline-messages LLM depends on."""
    entry = ctx.entry
    user_id_to_internal = ctx.user_id_to_internal
    msg_entry = OpEntry(
        id=entry.id,
        op="MessageOp",
        inputs=[],
        fields={"messages": prepared_messages},
    )
    msg_internal_id = make_id(ctx.scope, "message", entry.id)
    # `_inputs` on a MessageOp mirrors n8n: only ops whose ids appear in
    # message contents count.
    seen: list[str] = []
    for ref in entry.inputs:
        dep = user_id_to_internal[ref]
        if dep not in seen:
            seen.append(dep)
    for fmt_op in extra_format_ops:
        if fmt_op["_id"] not in seen:
            seen.append(fmt_op["_id"])
    resolved_contents: set[str] = set()
    for msg in prepared_messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        resolved_contents.add(user_id_to_internal.get(content, content))
    msg_inputs = [dep for dep in seen if dep in resolved_contents]
    msg_dict = {
        "_id": msg_internal_id,
        "_op": "MessageOp",
        "_max_iter": None,
        "_inputs": msg_inputs,
    }
    emit_message_op(
        msg_dict,
        EmitContext(
            entry=msg_entry,
            user_id_to_internal=user_id_to_internal,
            scope=ctx.scope,
            graph_ops=ctx.graph_ops,
        ),
    )
    op_dict.setdefault("__implicit_ops__", []).append(msg_dict)
    return msg_internal_id


def _resolve_table_nodes(
    table: Any, user_id_to_internal: dict[str, str]
) -> list[dict[str, Any]] | None:
    """Resolve ``node:`` user-ids inside aggregate/rowwise table specs.

    Each row is copied, and a ``node`` entry that matches a known user-facing
    id gets rewritten to the corresponding internal op id. Rows without a
    ``node`` field (e.g. ``data``-only entries) pass through untouched.
    """
    if table is None:
        return None
    if not isinstance(table, list):
        raise ValueError("aggregate_table/rowwise_columns must be a list")
    out: list[dict[str, Any]] = []
    for item in table:
        if not isinstance(item, dict):
            raise ValueError("aggregate_table/rowwise_columns entries must be mappings")
        resolved = dict(item)
        node_ref = resolved.get("node")
        if isinstance(node_ref, str) and node_ref in user_id_to_internal:
            resolved["node"] = user_id_to_internal[node_ref]
        out.append(resolved)
    return out


def _build_inline_messages(
    messages: list[Any],
    *,
    prompt_spec: Any,
    ctx: EmitContext,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Return (prepared_messages, implicit_format_ops).

    Two routes:

    1. ``prompt_spec`` is a mapping with ``template`` / ``format_kwargs``. The
       helper builds one implicit :class:`FormatOp` from that spec and rewrites
       every user-role message's ``content`` to the FormatOp's internal id.
       This is how aggregate / multi-ref prompts are expressed: the YAML
       author provides the raw template (including ``{df}`` / ``{ref0}``) once
       and this helper reproduces the ``FormatOp → MessageOp → LLMChatOp``
       shape n8n emits for the equivalent chain node.

    2. Otherwise fall back to :func:`_wrap_user_content_refs`, which only
       wraps bare user-id content through a trivial ``"{ref0}"`` FormatOp.
    """
    entry = ctx.entry
    user_id_to_internal = ctx.user_id_to_internal
    if isinstance(prompt_spec, dict):
        template = prompt_spec.get("template")
        format_kwargs = prompt_spec.get("format_kwargs", {}) or {}
        if not isinstance(template, str):
            raise ValueError(
                f"LLMChat/Vision '{entry.id}' prompt.template must be a string"
            )
        if not isinstance(format_kwargs, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in format_kwargs.items()
        ):
            raise ValueError(
                f"LLMChat/Vision '{entry.id}' prompt.format_kwargs must be "
                "dict[str, str]"
            )
        resolved_kwargs: dict[str, str] = {}
        fmt_inputs: list[str] = []
        for key, ref in format_kwargs.items():
            if ref not in user_id_to_internal:
                raise ValueError(
                    f"LLMChat/Vision '{entry.id}' prompt.format_kwargs "
                    f"references unknown id '{ref}'"
                )
            internal = user_id_to_internal[ref]
            resolved_kwargs[key] = internal
            if internal not in fmt_inputs:
                fmt_inputs.append(internal)
        fmt_id = make_id(ctx.scope, "format", entry.id)
        fmt_op = {
            "_id": fmt_id,
            "_op": "FormatOp",
            "_max_iter": None,
            "_inputs": fmt_inputs,
            "template": template,
            "format_args": [],
            "format_kwargs": resolved_kwargs,
        }
        rewritten: list[Any] = []
        for msg in messages:
            if not isinstance(msg, dict):
                rewritten.append(msg)
                continue
            if msg.get("role") == "user":
                new_msg = dict(msg)
                new_msg["content"] = fmt_id
                rewritten.append(new_msg)
            else:
                rewritten.append(msg)
        return rewritten, [fmt_op]

    return _wrap_user_content_refs(messages, ctx=ctx)


def _wrap_user_content_refs(
    messages: list[Any],
    *,
    ctx: EmitContext,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Mirror n8n's prompt-wrapping for a bare user-role reference.

    When a user-role message's ``content`` is a single user-facing id that
    resolves to a non-FormatOp (e.g. an InputOp, LLMChatOp, DataRetrievalOp),
    emit an implicit FormatOp whose ``_id`` uses ``entry.id`` as the name
    component — the same scope/name pair n8n's ``_build_prompt_content`` uses
    for the equivalent chain node — and rewrite the message content to the
    FormatOp's ``_id``.

    No-ops for:
      - Contents that aren't user ids (plain strings / templates).
      - Contents that resolve to a FormatOp (already formatted upstream).
      - Non-user roles.
    """
    entry = ctx.entry
    user_id_to_internal = ctx.user_id_to_internal
    graph_ops = ctx.graph_ops

    if not isinstance(messages, list) or not messages:
        return messages, []

    format_op: dict[str, Any] | None = None
    prepared: list[Any] = []
    for msg in messages:
        if not isinstance(msg, dict):
            prepared.append(msg)
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role != "user" or not isinstance(content, str):
            prepared.append(msg)
            continue
        op_id = user_id_to_internal.get(content)
        if op_id is None:
            prepared.append(msg)
            continue
        referenced = graph_ops.get(op_id)
        if referenced and referenced.get("_op") == "FormatOp":
            prepared.append(msg)
            continue
        # Reuse a single implicit FormatOp if multiple user messages reference
        # the same or different upstream ids. n8n only ever produces one per
        # chain node, so we follow that convention.
        if format_op is None:
            fmt_id = make_id(ctx.scope, "format", entry.id)
            format_op = {
                "_id": fmt_id,
                "_op": "FormatOp",
                "_max_iter": None,
                "_inputs": [op_id],
                "template": "{ref0}",
                "format_args": [],
                "format_kwargs": {"ref0": op_id},
            }
        else:
            fmt_kwargs = format_op["format_kwargs"]
            key = f"ref{len(fmt_kwargs)}"
            fmt_kwargs[key] = op_id
            if op_id not in format_op["_inputs"]:
                format_op["_inputs"].append(op_id)
            # For a second distinct ref, template becomes "{ref0}{ref1}" — a
            # simple concatenation. Callers needing richer templates should
            # declare an explicit FormatOp instead.
            format_op["template"] = "".join("{" + k + "}" for k in fmt_kwargs)
        rewritten = dict(msg)
        rewritten["content"] = format_op["_id"]
        prepared.append(rewritten)

    if format_op is None:
        return messages, []
    return prepared, [format_op]
