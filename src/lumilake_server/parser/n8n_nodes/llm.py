"""n8n LLM nodes: the chain and agent translators, plus the prompt, message
and model-config helpers they use.
"""

import re
from typing import Any

from lumilake_server.common import retrieval_items_path
from lumilake_server.data_specs import NodeColumn, build_column_table

from ..common import make_id as _make_id
from . import references, shared
from .constants import TABLE_BLOCK_PATTERN
from .entries import ChainPlan, NodeDict, OpContext


def chain_plan(node: NodeDict, ctx: OpContext, node_name: str) -> ChainPlan:
    params = node.get("parameters", {})
    messages = params.get("messages", {}).get("messageValues", [])
    notes = shared.parse_notes(node.get("notes", ""))
    op_type = notes.get("op-type")
    if not op_type:
        raise ValueError(f"Chain node '{node_name}' missing op-type in notes")
    model_config, vlm_config = _resolve_model_config(
        node_name, ctx.incoming, ctx.model_specs
    )
    if op_type == "data-retrieval":
        raise ValueError(
            "op-type 'data-retrieval' is unsupported. "
            "Use op-type 'text-generation' with notes.mode='rowwise' instead."
        )
    return ChainPlan(
        node=node,
        node_name=node_name,
        ctx=ctx,
        prompt_text=params.get("text", ""),
        system_messages=[m.get("message") for m in messages if m.get("message")],
        notes=notes,
        upstream_main=references.find_main_upstream(
            node_name, ctx.incoming, ctx.node_map
        ),
        model_config=model_config,
        vlm_config=vlm_config,
        return_history=bool(notes.get("return-history")),
    )


def build_text_generation(plan: ChainPlan) -> tuple[list[NodeDict], str]:
    return _build_chat_chain(plan, plan.prompt_text, None)


def build_aggregate(plan: ChainPlan) -> tuple[list[NodeDict], str]:
    ctx = plan.ctx
    prompt_text, aggregate_table = _build_aggregate_prompt_content(
        plan.prompt_text, ctx.op_ids, plan.upstream_main, ctx.node_map
    )
    return _build_chat_chain(plan, prompt_text, aggregate_table)


def build_rowwise(plan: ChainPlan) -> tuple[list[NodeDict], str]:
    ctx = plan.ctx
    msg_content = _build_rowwise_prompt_content(plan.prompt_text)
    rowwise_columns = _extract_rowwise_columns(
        plan.prompt_text, ctx.node_map, ctx.op_ids, ctx.inputs
    )
    msg_op = _build_message_op(
        ctx.scope, plan.node_name, plan.system_messages, msg_content
    )
    llm_op = _build_llm_chat_op(
        scope=ctx.scope,
        node_name=plan.node_name,
        msg_op_id=msg_op["_id"],
        model_config=plan.model_config,
        return_history=plan.return_history,
        rowwise_template=msg_content,
        rowwise_columns=rowwise_columns,
        system_messages=plan.system_messages,
    )
    return [msg_op, llm_op], llm_op["_id"]


def build_image_generation(plan: ChainPlan) -> tuple[list[NodeDict], str]:
    ctx = plan.ctx
    prompt_text = plan.prompt_text
    if plan.system_messages:
        prompt_text = "\n\n".join(plan.system_messages) + "\n\n" + (prompt_text or "")
    msg_content, extra_ops = _build_prompt_content(
        prompt_text,
        plan.node_name,
        ctx.op_ids,
        plan.upstream_main,
        ctx.scope,
        escape=False,
    )
    ops = list(extra_ops)
    content_op_id = msg_content if shared.is_op_id(msg_content) else None
    if content_op_id is None:
        data_op = _build_data_op(ctx.scope, plan.node_name, msg_content)
        ops.append(data_op)
        content_op_id = data_op["_id"]

    img_config = plan.model_config
    if "model" in plan.notes:
        img_config = {"model": plan.notes["model"]}

    llm_op = build_image_generation_op(
        ctx.scope, plan.node_name, content_op_id, img_config
    )
    ops.append(llm_op)
    return ops, llm_op["_id"]


def build_image_understanding(plan: ChainPlan) -> tuple[list[NodeDict], str]:
    ctx = plan.ctx
    image_ref = plan.notes.get("image-input")
    if not image_ref:
        raise ValueError(f"Chain node '{plan.node_name}' missing image-input in notes")
    vision_config = (
        plan.vlm_config if plan.vlm_config is not None else plan.model_config
    )
    image_source, _ = _resolve_image_source(image_ref, ctx.op_ids, plan.node_name)
    image_path = plan.notes.get("image-path")
    if not isinstance(image_path, str) or not image_path:
        image_path = "images"
    rowwise_columns = _extract_rowwise_columns(
        plan.prompt_text, ctx.node_map, ctx.op_ids, ctx.inputs
    )
    if rowwise_columns:
        msg_content = _build_rowwise_prompt_content(plan.prompt_text)
        extra_ops: list[dict[str, Any]] = []
        rowwise_template = msg_content
    else:
        msg_content, extra_ops = _build_prompt_content(
            plan.prompt_text,
            plan.node_name,
            ctx.op_ids,
            plan.upstream_main,
            ctx.scope,
        )
        rowwise_template = None
    ops = list(extra_ops)
    msg_op = _build_message_op(
        ctx.scope, plan.node_name, plan.system_messages, msg_content
    )
    ops.append(msg_op)
    llm_op = _build_llm_vision_op(
        ctx.scope,
        plan.node_name,
        msg_op["_id"],
        vision_config,
        image_source,
        str(image_path),
        plan.return_history,
        rowwise_template=rowwise_template,
        rowwise_columns=rowwise_columns if rowwise_columns else None,
        system_messages=plan.system_messages or None,
    )
    ops.append(llm_op)
    return ops, llm_op["_id"]


def _build_chat_chain(
    plan: ChainPlan,
    prompt_text: str,
    aggregate_table: list[dict[str, Any]] | None,
) -> tuple[list[NodeDict], str]:
    ctx = plan.ctx
    msg_content, extra_ops = _build_prompt_content(
        prompt_text,
        plan.node_name,
        ctx.op_ids,
        plan.upstream_main,
        ctx.scope,
        already_escaped=aggregate_table is not None,
    )
    ops = list(extra_ops)
    msg_op = _build_message_op(
        ctx.scope, plan.node_name, plan.system_messages, msg_content
    )
    ops.append(msg_op)
    llm_op = _build_llm_chat_op(
        ctx.scope,
        plan.node_name,
        msg_op["_id"],
        plan.model_config,
        plan.return_history,
        structural_outputs=plan.notes.get("structural-outputs"),
        aggregate_table=aggregate_table,
    )
    ops.append(llm_op)
    return ops, llm_op["_id"]


def make_agent(
    node: NodeDict, ctx: OpContext, node_name: str
) -> tuple[list[NodeDict], str]:
    params = node.get("parameters", {})
    text = params.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            f"n8n agent node '{node_name}' missing parameters.text (the"
            " natural-language retrieval task)."
        )
    options = params.get("options") or {}
    if options and not isinstance(options, dict):
        raise ValueError(
            f"n8n agent node '{node_name}' parameters.options must be an object."
        )

    notes = shared.parse_notes(node.get("notes", ""))
    schema_scope = notes.get("schema-scope") or notes.get("schema_scope")
    if not isinstance(schema_scope, str) or not schema_scope.strip():
        raise ValueError(
            f"n8n agent node '{node_name}' notes missing 'schema-scope'"
            " (comma-separated list of demo.* tables the agent may query)."
        )
    output_format = notes.get("output-format") or notes.get("output_format")
    if output_format is None:
        output_format = "jsonl"
    if not isinstance(output_format, str):
        raise ValueError(
            f"n8n agent node '{node_name}' notes 'output-format' must be a string."
        )

    raw_max_steps = (
        notes.get("max-steps")
        or notes.get("max_steps")
        or (options.get("maxIterations") if isinstance(options, dict) else None)
    )
    max_steps: int | None
    if raw_max_steps is None:
        max_steps = None
    else:
        try:
            max_steps = int(raw_max_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"n8n agent node '{node_name}' max_steps must be an integer."
            ) from exc

    verify_raw = notes.get("verify")
    verify: bool | None
    if verify_raw is None:
        verify = None
    elif isinstance(verify_raw, bool):
        verify = verify_raw
    elif isinstance(verify_raw, str):
        verify = verify_raw.strip().lower() in {"1", "true", "yes"}
    else:
        raise ValueError(f"n8n agent node '{node_name}' notes 'verify' must be a bool.")

    description, agent_params, extra_deps = (
        shared.build_execute_query_template_and_params(
            query=text, op_ids=ctx.op_ids, inputs=ctx.inputs, node_map=ctx.node_map
        )
    )

    main_inputs = [
        ctx.op_ids[dep]
        for dep in references.main_incoming(node_name, ctx.incoming, ctx.node_map)
        if dep in ctx.op_ids
    ]
    dep_inputs = [
        param["node"]
        for param in agent_params
        if isinstance(param, dict) and isinstance(param.get("node"), str)
    ]

    data_spec: dict[str, Any] = {
        "type": "lumid",
        "mode": "agent",
        "schema_scope": schema_scope,
        "output_format": output_format,
        "verify": bool(verify) if verify is not None else False,
        "description": description,
        "params": agent_params,
    }
    if max_steps is not None:
        data_spec["max_steps"] = max_steps

    op: dict[str, Any] = {
        "_id": _make_id(ctx.scope, "retrieval", node_name),
        "_op": "DataRetrievalOp",
        "_max_iter": None,
        "_inputs": list(dict.fromkeys(extra_deps + dep_inputs + main_inputs)),
        "data_spec": data_spec,
    }
    return [op], op["_id"]


def chain_dependencies(node_name: str, ctx: OpContext) -> list[str]:
    node = ctx.node_map[node_name]
    deps: list[str] = []
    upstream_main = references.find_main_upstream(node_name, ctx.incoming, ctx.node_map)
    if upstream_main:
        deps.append(upstream_main)
    params = node.get("parameters", {})
    prompt_text = params.get("text", "")
    refs = shared.extract_refs(shared.strip_expr(prompt_text or ""))
    notes = shared.parse_notes(node.get("notes", ""))
    mode = notes.get("mode")
    if mode in {"rowwise", "aggregate"}:
        deps.extend(ref for _, ref in refs)
    else:
        deps.extend(
            ref
            for _, ref in refs
            if not references.is_data_source_node(ctx.node_map, ref)
        )
    if notes.get("op-type") == "text-generation" and notes.get("mode") == "rowwise":
        try:
            postgres_node, _ = references.resolve_data_nodes(
                node_name, ctx.incoming, ctx.node_map
            )
            pg_params = postgres_node.get("parameters", {})
            deps.extend(shared.extract_ref_names(pg_params.get("limit")))
            where = pg_params.get("where", {})
            if isinstance(where, dict):
                deps.extend(shared.extract_ref_names(where))
        except Exception:
            # Keep dep resolution permissive; strict validation happens later.
            pass
    deps.extend(shared.extract_ref_names(notes.get("filters")))
    deps.extend(shared.extract_ref_names(notes.get("input")))
    image_ref = notes.get("image-input")
    deps.extend(shared.extract_ref_names(image_ref))
    if isinstance(image_ref, str) and "$(" not in image_ref:
        deps.append(image_ref)
    return [dep for dep in deps if isinstance(dep, str)]


def agent_dependencies(node_name: str, ctx: OpContext) -> list[str]:
    node = ctx.node_map[node_name]
    deps: list[str] = []
    params = node.get("parameters", {})
    deps.extend(shared.extract_ref_names(params.get("text")))
    for edge in ctx.incoming.get(node_name, []):
        if edge.get("type") == "main" and edge.get("node"):
            deps.append(edge["node"])
    return [dep for dep in deps if isinstance(dep, str)]


_ESCAPE_BRACES_PATTERN = re.compile(rf"({shared.REF_PATTERN.pattern})|([{{}}])")


def _escape_literal_braces(text: str) -> str:
    """Escape literal ``{``/``}`` to ``{{``/``}}``, leaving n8n ``{{ ... }}``
    expression wrappers intact so they can be swapped for placeholders."""

    def _repl(match: re.Match[str]) -> str:
        if match.group(1) is not None:
            return match.group(1)
        return match.group(0) * 2

    return _ESCAPE_BRACES_PATTERN.sub(_repl, text)


def _build_prompt_content(
    prompt_text: str,
    node_name: str,
    op_ids: dict[str, str],
    upstream_main: str | None,
    scope: str,
    escape: bool = True,
    already_escaped: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    stripped = shared.strip_expr(prompt_text or "")
    if not stripped and upstream_main and upstream_main in op_ids:
        return op_ids[upstream_main], []

    refs = shared.extract_refs(stripped)
    if refs:
        template = stripped if already_escaped else _escape_literal_braces(stripped)
        format_kwargs: dict[str, str] = {}
        format_paths: dict[str, str] = {}
        for idx, (full, ref_name) in enumerate(refs):
            if ref_name not in op_ids:
                raise ValueError(f"Unknown reference '{ref_name}' in prompt")
            key = f"ref{idx}"
            template = template.replace(full, f"{{{key}}}")
            format_kwargs[key] = op_ids[ref_name]
            detail_matches = shared.extract_ref_details(full)
            ref_path = detail_matches[0][1] if detail_matches else None
            runtime_path = references.to_runtime_output_path(ref_path)
            if runtime_path and runtime_path != "items.output":
                format_paths[key] = runtime_path
        fmt_op = _build_format_op(
            scope, node_name, template, format_kwargs, format_paths
        )
        return fmt_op["_id"], [fmt_op]

    if already_escaped or not escape:
        return stripped, []
    return _escape_literal_braces(stripped), []


def _build_message_op(
    scope: str, node_name: str, system_messages: list[str], user_content: str
) -> dict[str, Any]:
    messages = []
    for msg in system_messages:
        messages.append({"role": "system", "content": msg})
    messages.append({"role": "user", "content": user_content})

    inputs = [msg["content"] for msg in messages if shared.is_op_id(msg["content"])]
    return {
        "_id": _make_id(scope, "message", node_name),
        "_op": "MessageOp",
        "_max_iter": None,
        "_inputs": inputs,
        "messages": messages,
    }


def _build_format_op(
    scope: str,
    node_name: str,
    template: str,
    format_kwargs: dict[str, str],
    format_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    op: dict[str, Any] = {
        "_id": _make_id(scope, "format", node_name),
        "_op": "FormatOp",
        "_max_iter": None,
        "_inputs": list(format_kwargs.values()),
        "template": template,
        "format_args": [],
        "format_kwargs": format_kwargs,
    }
    if format_paths:
        op["format_paths"] = format_paths
    return op


def _build_data_op(scope: str, node_name: str, content: str) -> dict[str, Any]:
    return {
        "_id": _make_id(scope, "data", node_name),
        "_op": "DataOp",
        "_max_iter": None,
        "_inputs": [],
        "data": [content],
    }


def build_image_generation_op(
    scope: str, node_name: str, content_op_id: str, model_config: dict[str, Any]
) -> dict[str, Any]:
    return {
        "_id": _make_id(scope, "imagegen", node_name),
        "_op": "ImageGenerationOp",
        "_max_iter": None,
        "_inputs": [content_op_id],
        "content": content_op_id,
        "config": model_config,
        "cacheable": False,
    }


def _build_llm_chat_op(
    scope: str,
    node_name: str,
    msg_op_id: str,
    model_config: dict[str, Any],
    return_history: bool = False,
    structural_outputs: Any = None,
    aggregate_table: list[dict[str, Any]] | None = None,
    rowwise_template: str | None = None,
    rowwise_columns: list[dict[str, Any]] | None = None,
    system_messages: list[str] | None = None,
) -> dict[str, Any]:
    aggregate_columns = build_column_table(aggregate_table, "aggregate_table")
    rowwise_specs = build_column_table(rowwise_columns, "rowwise_columns")
    llm_inputs = list(
        dict.fromkeys(
            [
                msg_op_id,
                *(
                    column.node
                    for table in (aggregate_columns, rowwise_specs)
                    for column in table or []
                    if isinstance(column, NodeColumn)
                ),
            ]
        )
    )

    op: dict[str, Any] = {
        "_id": _make_id(scope, "llm", node_name),
        "_op": "LLMChatOp",
        "_max_iter": None,
        "_inputs": llm_inputs,
        "messages": msg_op_id,
        "config": model_config,
        "return_history": return_history,
        "cacheable": False,
    }
    if structural_outputs is not None:
        if not isinstance(structural_outputs, list) or not all(
            isinstance(item, dict) for item in structural_outputs
        ):
            raise ValueError(
                f"Chain node '{node_name}' structural-outputs must be a list of objects"
            )
        op["structural_outputs"] = structural_outputs
    if aggregate_columns is not None:
        op["aggregate_table"] = [
            column.model_dump(exclude_none=True) for column in aggregate_columns
        ]
    if rowwise_template is not None:
        op["rowwise_template"] = rowwise_template
    if rowwise_specs is not None:
        op["rowwise_columns"] = [
            column.model_dump(exclude_none=True) for column in rowwise_specs
        ]
    if system_messages is not None:
        op["system_messages"] = system_messages
    return op


def _build_llm_vision_op(
    scope: str,
    node_name: str,
    msg_op_id: str,
    model_config: dict[str, Any],
    image_source: str,
    image_path: str,
    return_history: bool = False,
    rowwise_template: str | None = None,
    rowwise_columns: list[dict[str, Any]] | None = None,
    system_messages: list[str] | None = None,
) -> dict[str, Any]:
    op: dict[str, Any] = {
        "_id": _make_id(scope, "llmvision", node_name),
        "_op": "LLMVisionOp",
        "_max_iter": None,
        "_inputs": [msg_op_id],
        "messages": msg_op_id,
        "config": model_config,
        "return_history": return_history,
        "cacheable": False,
        "image_source": image_source,
        "image_path": image_path,
    }
    if rowwise_template is not None:
        op["rowwise_template"] = rowwise_template
    vision_columns = build_column_table(rowwise_columns, "rowwise_columns")
    if vision_columns is not None:
        op["rowwise_columns"] = [
            column.model_dump(exclude_none=True) for column in vision_columns
        ]
    if system_messages is not None:
        op["system_messages"] = system_messages
    return op


def _resolve_image_source(
    value: Any, op_ids: dict[str, str], node_name: str
) -> tuple[str, str]:
    if isinstance(value, str):
        refs = shared.extract_refs(value)
        if refs:
            _, name = refs[0]
            if name in op_ids:
                return op_ids[name], name
            raise ValueError(
                f"Chain node '{node_name}' image-input references unknown node '{name}'"
            )
        if value in op_ids:
            return op_ids[value], value
    raise ValueError(f"Chain node '{node_name}' image-input must reference a node name")


def _build_rowwise_prompt_content(prompt_text: str) -> str:
    stripped = shared.strip_expr(prompt_text or "")
    refs = _extract_rowwise_ref_bindings(stripped)
    template = _escape_literal_braces(stripped)
    for node_name, path, label in refs:
        escaped_name = re.escape(node_name)
        full_pattern = None
        if path:
            # Keep original formatting as much as possible.
            full_pattern = re.compile(
                r"\{\{\s*\$\('" + escaped_name + r"'\)\." + re.escape(path) + r"\s*\}\}"
            )
        if full_pattern:
            template = full_pattern.sub(f"{{{label}}}", template, count=1)
        else:
            template = re.sub(
                r"\{\{\s*\$\('" + escaped_name + r"'\)(?:\.[^}]*)?\s*\}\}",
                f"{{{label}}}",
                template,
                count=1,
            )
    return template


def _extract_rowwise_columns(
    prompt_text: str,
    node_map: dict[str, Any],
    op_ids: dict[str, str],
    inputs: dict[str, list[str]],
) -> list[dict[str, Any]]:
    stripped = shared.strip_expr(prompt_text or "")
    columns: list[dict[str, Any]] = []
    for node_name, path, label in _extract_rowwise_ref_bindings(stripped):
        if node_name not in op_ids:
            continue
        if not node_map.get(node_name):
            continue
        columns.append(
            references.column(node_map, op_ids, inputs, node_name, label, path)
        )
    return columns


def _extract_rowwise_ref_bindings(
    prompt_text: str,
) -> list[tuple[str, str | None, str]]:
    bindings: list[tuple[str, str | None, str]] = []
    label_by_ref: dict[tuple[str, str | None], str] = {}
    seen_labels: set[str] = set()

    for node_name, path in shared.extract_ref_details(prompt_text):
        key = (node_name, path)
        label = label_by_ref.get(key)
        if label is None:
            base_label = references.path_to_label(path)
            if not base_label:
                base_label = (
                    re.sub(r"[^A-Za-z0-9_]+", "_", node_name).strip("_") or "value"
                )
            label = shared.ensure_unique_label(base_label, seen_labels)
            label_by_ref[key] = label
        bindings.append((node_name, path, label))
    return bindings


def _build_aggregate_prompt_content(
    prompt_text: str,
    op_ids: dict[str, str],
    upstream_main: str | None,
    node_map: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    stripped = shared.strip_expr(prompt_text or "")
    table_spec: list[dict[str, Any]] = []
    table_match = TABLE_BLOCK_PATTERN.search(stripped)
    if not table_match:
        return _escape_literal_braces(stripped), table_spec

    block = table_match.group(1)
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        column, value = line.split(":", 1)
        column = column.strip()
        value = value.strip()
        ref_detail = shared.extract_ref_details(value)
        if not ref_detail:
            continue
        ref_node_name, path = ref_detail[0]
        if ref_node_name in op_ids:
            table_spec.append(
                references.table_column(node_map, op_ids, ref_node_name, column, path)
            )
            continue

        if upstream_main and ref_node_name in node_map:
            upstream_id = op_ids.get(upstream_main)
            reference = references.retrieval_mode(node_map, ref_node_name)
            if upstream_id and reference == "sql":
                suffix = references.path_to_label(path) or column
                table_spec.append(
                    {
                        "label": column,
                        "node": upstream_id,
                        "path": f"{retrieval_items_path('sql')}.{suffix}",
                    }
                )

    escaped = _escape_literal_braces(stripped)
    rewritten = TABLE_BLOCK_PATTERN.sub("{df}", escaped)
    return rewritten, table_spec


def _resolve_model_config(
    node_name: str,
    incoming: dict[str, list[dict[str, str]]],
    model_specs: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    sources: list[str] = []
    for edge in incoming.get(node_name, []):
        if edge.get("type") != "ai_languageModel":
            continue
        model_node = edge.get("node")
        if not model_node:
            continue
        if model_node not in model_specs:
            raise ValueError(
                f"Chain node '{node_name}' has unknown model source '{model_node}'"
            )
        sources.append(model_node)
    if not sources:
        raise ValueError(f"Chain node '{node_name}' missing model connection")
    if len(sources) != 1:
        raise ValueError(
            f"Chain node '{node_name}' has multiple model connections: {sources}"
        )
    spec = model_specs[sources[0]]
    return spec["config"], spec.get("vlm_config")
