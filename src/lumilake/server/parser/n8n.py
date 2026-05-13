import json
import re
from dataclasses import dataclass
from typing import Any

from lumilake import envs

from .common import make_id as _make_id

# n8n wire-format node type identifiers. Private to this module — anything
# outside that needs to classify n8n nodes should go through a helper
# exported here (e.g. :func:`extract_chat_trigger_inputs`) instead of
# comparing raw type strings.
N8N_CHAIN_TYPE = "@n8n/n8n-nodes-langchain.chainLlm"
N8N_CHAT_TRIGGER = "@n8n/n8n-nodes-langchain.chatTrigger"
N8N_MODEL_SPEC = "@n8n/n8n-nodes-langchain.lmOpenHuggingFaceInference"
N8N_CODE_NODE = "n8n-nodes-base.code"
N8N_POSTGRES_NODE = "n8n-nodes-base.postgres"
N8N_S3_NODE = "n8n-nodes-base.s3"


def extract_chat_trigger_inputs(workflow: dict[str, Any]) -> dict[str, list[str]]:
    """Return ``{chat_trigger_name: []}`` for every ChatTrigger in ``workflow``.

    Tests and test harnesses use this to synthesize a minimal ``inputs``
    payload for a workflow without needing to know the n8n type string.
    """
    inputs: dict[str, list[str]] = {}
    for node in workflow.get("nodes", []):
        if not isinstance(node, dict):
            continue
        if node.get("type") != N8N_CHAT_TRIGGER:
            continue
        name = node.get("name")
        if isinstance(name, str) and name:
            inputs[name] = []
    return inputs


REF_PATTERN = re.compile(
    r"\{\{\s*\$\('([^']+)'\)(?:\.[^}]*)?\s*\}\}|\$\('([^']+)'\)(?:\.[\w\.\[\]]+)?"
)
REF_DETAIL_PATTERN = re.compile(r"\$\('([^']+)'\)(?:\.([\w\.\[\]]+))?")
QUERY_EXPR_PATTERN = re.compile(r"\{\{\s*(.*?)\s*\}\}")
REF_TOKEN_PREFIX = "__lumilake_ref__:"
TABLE_BLOCK_PATTERN = re.compile(r"```table\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass
class N8NGraphSpec:
    graph: dict[str, Any]
    inputs: dict[str, list[str]]


def parse_n8n_payload(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Parse an n8n request payload into Lumilake-style graph specs.

    Expected payload (flexible):
    {
      "graphs": [
        {
          "workflow": { ... n8n workflow json ... },
          "inputs": { "stock": ["NVDA"] }
        },
        ...
      ]
    }
    """
    graphs = payload.get("graphs")
    if not isinstance(graphs, list):
        raise ValueError("n8n payload must contain 'graphs' list")

    graph_specs: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(graphs):
        if not isinstance(item, dict):
            raise ValueError("each graph must be an object")
        workflow = item.get("workflow") or item.get("graph") or item.get("template")
        inputs = item.get("inputs", {})
        if not isinstance(workflow, dict):
            raise ValueError(f"graph {idx} missing workflow json")
        if not isinstance(inputs, dict):
            raise ValueError(f"graph {idx} inputs must be an object")

        graph_name = item.get("name") or f"graph_{idx}"
        compiled = _parse_n8n_workflow(workflow, inputs, graph_name)
        graph_specs[graph_name] = {
            "graph": compiled.graph,
            "inputs": compiled.inputs,
        }

    return graph_specs


def _parse_n8n_workflow(
    workflow: dict[str, Any], inputs: dict[str, list[str]], scope: str
) -> N8NGraphSpec:
    nodes, connections = _parse_and_validate_nodes(workflow)
    if not nodes:
        raise ValueError("n8n workflow has no nodes")

    node_map = _index_nodes(nodes)
    incoming = _build_incoming_edges(connections)
    model_specs = _index_model_specs(node_map)

    allowed_types = {
        N8N_CHAIN_TYPE,
        N8N_CHAT_TRIGGER,
        N8N_MODEL_SPEC,
        N8N_CODE_NODE,
        N8N_POSTGRES_NODE,
        N8N_S3_NODE,
    }
    unsupported = [
        name for name, node in node_map.items() if node.get("type") not in allowed_types
    ]
    if unsupported:
        raise ValueError(f"Unsupported n8n node types: {unsupported}")

    ordered_ops: list[dict[str, Any]] = []
    graph_ops: dict[str, dict[str, Any]] = {}
    op_ids: dict[str, str] = {}

    def add_op(op: dict[str, Any] | None) -> None:
        if not op:
            return
        op_id = op["_id"]
        if op_id in graph_ops:
            return
        graph_ops[op_id] = op
        ordered_ops.append(op)

    # Inputs first
    for node_name, node in node_map.items():
        if node["type"] == N8N_CHAT_TRIGGER:
            op = _make_input_op(scope, node_name, inputs)
            add_op(op)
            op_ids[node_name] = op["_id"]

    # Build workflow nodes in dependency order.
    chain_nodes = {
        name: node for name, node in node_map.items() if node["type"] == N8N_CHAIN_TYPE
    }
    code_nodes = {
        name: node for name, node in node_map.items() if node["type"] == N8N_CODE_NODE
    }
    postgres_nodes = {
        name: node
        for name, node in node_map.items()
        if node["type"] == N8N_POSTGRES_NODE
    }
    s3_nodes = {
        name: node for name, node in node_map.items() if node["type"] == N8N_S3_NODE
    }

    pending = (
        set(chain_nodes.keys())
        | set(code_nodes.keys())
        | set(postgres_nodes.keys())
        | set(s3_nodes.keys())
    )
    while pending:
        progress = False
        for node_name in list(pending):
            node_raw = (
                chain_nodes.get(node_name)
                or code_nodes.get(node_name)
                or postgres_nodes.get(node_name)
                or s3_nodes.get(node_name)
            )
            if not node_raw:
                raise ValueError(f"Unknown node '{node_name}'")
            node = node_raw
            deps = _resolve_dependencies(node_name, node, incoming, node_map)
            if any(dep in pending for dep in deps):
                continue
            if any(dep not in op_ids for dep in deps):
                continue

            if node_name in chain_nodes:
                ops, llm_id = _make_chain_ops(
                    node=chain_nodes[node_name],
                    node_name=node_name,
                    incoming=incoming,
                    node_map=node_map,
                    op_ids=op_ids,
                    inputs=inputs,
                    model_specs=model_specs,
                    scope=scope,
                )
                for op in ops:
                    add_op(op)
                op_ids[node_name] = llm_id
            elif node_name in postgres_nodes:
                op = _make_postgres_retrieval_op(
                    scope=scope,
                    node_name=node_name,
                    node=postgres_nodes[node_name],
                    incoming=incoming,
                    node_map=node_map,
                    op_ids=op_ids,
                    inputs=inputs,
                )
                add_op(op)
                op_ids[node_name] = op["_id"]
            elif node_name in s3_nodes:
                if _is_lake_retrieval_node(s3_nodes[node_name]):
                    op = _make_lake_retrieval_op(
                        scope=scope,
                        node_name=node_name,
                        node=s3_nodes[node_name],
                        incoming=incoming,
                        node_map=node_map,
                        op_ids=op_ids,
                        inputs=inputs,
                    )
                else:
                    op = _make_s3_retrieval_op(
                        scope=scope,
                        node_name=node_name,
                        node=s3_nodes[node_name],
                        incoming=incoming,
                        node_map=node_map,
                        op_ids=op_ids,
                        inputs=inputs,
                    )
                add_op(op)
                op_ids[node_name] = op["_id"]
            else:
                ops, op_id = _make_code_ops(
                    node=code_nodes[node_name],
                    node_name=node_name,
                    incoming=incoming,
                    node_map=node_map,
                    op_ids=op_ids,
                    scope=scope,
                )
                for op in ops:
                    add_op(op)
                op_ids[node_name] = op_id
            pending.remove(node_name)
            progress = True
        if not progress:
            missing = ", ".join(sorted(pending))
            raise ValueError(f"Unresolved n8n dependencies: {missing}")

    # Outputs last
    for node_name, node in chain_nodes.items():
        notes = _parse_notes(node.get("notes", ""))
        if notes.get("is-output", False):
            output_name = notes.get("output-field") or node_name
            upstream = op_ids.get(node_name)
            if not upstream:
                raise ValueError(f"Output node {node_name} missing op id")
            output_op = _make_output_op(scope, output_name, upstream)
            add_op(output_op)

    graph = {op["_id"]: op for op in ordered_ops}
    return N8NGraphSpec(graph=graph, inputs=inputs)


def _parse_and_validate_nodes(
    workflow: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nodes = workflow.get("nodes")
    connections = workflow.get("connections", {})
    if not isinstance(nodes, list):
        raise ValueError("n8n workflow must contain nodes list")
    if not isinstance(connections, dict):
        raise ValueError("n8n workflow connections must be an object")

    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("n8n nodes must be objects")
        if not node.get("name"):
            raise ValueError("n8n node missing name")
        if not node.get("type"):
            raise ValueError(f"n8n node '{node.get('name')}' missing type")
        params = node.get("parameters")
        if params is None or not isinstance(params, dict):
            raise ValueError(f"n8n node '{node.get('name')}' missing parameters")

    return nodes, connections


def _index_nodes(nodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    node_map: dict[str, dict[str, Any]] = {}
    for node in nodes:
        name = node.get("name")
        if not name:
            raise ValueError("n8n node missing name")
        if name in node_map:
            raise ValueError(f"Duplicate node name: {name}")
        node_map[name] = node
    return node_map


def _build_incoming_edges(
    connections: dict[str, Any],
) -> dict[str, list[dict[str, str]]]:
    incoming: dict[str, list[dict[str, str]]] = {}
    for src, edges in connections.items():
        if not isinstance(edges, dict):
            continue
        for edge_type, edge_list in edges.items():
            if not isinstance(edge_list, list):
                continue
            for entry in edge_list:
                if not isinstance(entry, list):
                    continue
                for conn in entry:
                    if not isinstance(conn, dict):
                        continue
                    dest = conn.get("node")
                    if not dest:
                        continue
                    incoming.setdefault(dest, []).append(
                        {"node": src, "type": edge_type}
                    )
    return incoming


def _index_model_specs(
    node_map: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for name, node in node_map.items():
        if node.get("type") != N8N_MODEL_SPEC:
            continue
        params = node.get("parameters", {})
        model = params.get("model")
        if not model:
            raise ValueError(f"Model spec node '{name}' missing model")
        options = params.get("options", {})
        config = _parse_model_params(model, options)

        notes = _parse_notes(node.get("notes", ""))
        vlm_config = None
        if "vlm_parameters" in notes:
            vlm_params = notes["vlm_parameters"]
            if not isinstance(vlm_params, dict):
                raise ValueError(
                    f"Model spec node '{name}' vlm_parameters must be an object"
                )
            vlm_model = vlm_params.get("model")
            if not vlm_model:
                raise ValueError(
                    f"Model spec node '{name}' vlm_parameters missing model"
                )
            vlm_options = vlm_params.get("options", {})
            vlm_config = _parse_model_params(vlm_model, vlm_options)

        specs[name] = {"config": config, "vlm_config": vlm_config}
    return specs


def _is_data_source_node(node_map: dict[str, dict[str, Any]], name: str) -> bool:
    node = node_map.get(name)
    if not node:
        return False
    return node.get("type") in {N8N_POSTGRES_NODE, N8N_S3_NODE}


def _find_main_upstream(
    node_name: str,
    incoming: dict[str, list[dict[str, str]]],
    node_map: dict[str, dict[str, Any]],
) -> str | None:
    for edge in incoming.get(node_name, []):
        if edge.get("type") != "main":
            continue
        upstream = edge.get("node")
        if not upstream:
            continue
        if _is_data_source_node(node_map, upstream):
            continue
        return upstream
    return None


def _main_incoming(
    node_name: str,
    incoming: dict[str, list[dict[str, str]]],
    node_map: dict[str, dict[str, Any]],
) -> list[str]:
    return [
        edge["node"]
        for edge in incoming.get(node_name, [])
        if edge.get("type") == "main"
        and edge.get("node")
        and not _is_data_source_node(node_map, edge["node"])
    ]


def _resolve_dependencies(
    node_name: str,
    node: dict[str, Any] | None,
    incoming: dict[str, list[dict[str, str]]],
    node_map: dict[str, dict[str, Any]],
) -> list[str]:
    if not node:
        return []
    deps: list[str] | None
    if node.get("type") == N8N_CODE_NODE:
        notes = _parse_notes(node.get("notes", ""))
        deps = notes.get("inputs")
        if deps is None or deps == []:
            deps = _main_incoming(node_name, incoming, node_map)
        return [dep for dep in deps if isinstance(dep, str)]
    if node.get("type") == N8N_CHAIN_TYPE:
        deps = []
        upstream_main = _find_main_upstream(node_name, incoming, node_map)
        if upstream_main:
            deps.append(upstream_main)
        params = node.get("parameters", {})
        prompt_text = params.get("text", "")
        refs = _extract_refs(_strip_expr(prompt_text or ""))
        notes = _parse_notes(node.get("notes", ""))
        mode = notes.get("mode")
        if mode in {"rowwise", "aggregate"}:
            deps.extend(ref for _, ref in refs)
        else:
            deps.extend(
                ref for _, ref in refs if not _is_data_source_node(node_map, ref)
            )
        if notes.get("op-type") == "text-generation" and notes.get("mode") == "rowwise":
            try:
                postgres_node, _ = _resolve_data_nodes(node_name, incoming, node_map)
                pg_params = postgres_node.get("parameters", {})
                deps.extend(_extract_ref_names(pg_params.get("limit")))
                where = pg_params.get("where", {})
                if isinstance(where, dict):
                    deps.extend(_extract_ref_names(where))
            except Exception:
                # Keep dep resolution permissive; strict validation happens later.
                pass
        deps.extend(_extract_ref_names(notes.get("filters")))
        deps.extend(_extract_ref_names(notes.get("input")))
        image_ref = notes.get("image-input")
        deps.extend(_extract_ref_names(image_ref))
        if isinstance(image_ref, str) and "$(" not in image_ref:
            deps.append(image_ref)
        return [dep for dep in deps if isinstance(dep, str)]
    if node.get("type") == N8N_POSTGRES_NODE:
        deps = []
        params = node.get("parameters", {})
        deps.extend(_extract_ref_names(params.get("limit")))
        where = params.get("where", {})
        if isinstance(where, dict):
            deps.extend(_extract_ref_names(where))
        upstream_main = _find_main_upstream(node_name, incoming, node_map)
        if upstream_main:
            deps.append(upstream_main)
        return [dep for dep in deps if isinstance(dep, str)]
    if node.get("type") == N8N_S3_NODE:
        deps = []
        params = node.get("parameters", {})
        options = params.get("options")
        if isinstance(options, dict):
            deps.extend(_extract_ref_names(options.get("folderKey")))
        for edge in incoming.get(node_name, []):
            if edge.get("type") == "main" and edge.get("node"):
                deps.append(edge["node"])
        return [dep for dep in deps if isinstance(dep, str)]
    upstream_main = _find_main_upstream(node_name, incoming, node_map)
    return [upstream_main] if upstream_main else []


def _resolve_input_key(node_name: str, inputs: dict[str, list[str]]) -> str:
    """Return the matching key in *inputs* for *node_name* (case-insensitive)."""
    if node_name in inputs:
        return node_name
    lower = node_name.lower()
    for key in inputs:
        if key.lower() == lower:
            return key
    raise ValueError(f"Missing input for '{node_name}'")


def _make_input_op(
    scope: str, node_name: str, inputs: dict[str, list[str]]
) -> dict[str, Any]:
    matched_key = _resolve_input_key(node_name, inputs)
    return {
        "_id": _make_id(scope, "input", node_name),
        "_op": "InputOp",
        "_max_iter": None,
        "_inputs": [],
        "name": matched_key,
    }


def _make_chain_ops(
    node: dict[str, Any],
    node_name: str,
    incoming: dict[str, list[dict[str, str]]],
    node_map: dict[str, dict[str, Any]],
    op_ids: dict[str, str],
    inputs: dict[str, list[str]],
    model_specs: dict[str, dict[str, Any]],
    scope: str,
) -> tuple[list[dict[str, Any]], str]:
    params = node.get("parameters", {})
    prompt_text = params.get("text", "")
    messages = params.get("messages", {}).get("messageValues", [])
    system_messages = [m.get("message") for m in messages if m.get("message")]
    notes = _parse_notes(node.get("notes", ""))
    op_type = notes.get("op-type")
    mode = notes.get("mode")
    if not op_type:
        raise ValueError(f"Chain node '{node_name}' missing op-type in notes")

    upstream_main = _find_main_upstream(node_name, incoming, node_map)
    model_config, vlm_config = _resolve_model_config(node_name, incoming, model_specs)
    return_history = bool(notes.get("return-history"))

    ops: list[dict[str, Any]] = []

    if op_type == "data-retrieval":
        raise ValueError(
            "op-type 'data-retrieval' is unsupported. "
            "Use op-type 'text-generation' with notes.mode='rowwise' instead."
        )

    if op_type == "text-generation" and mode == "rowwise":
        msg_content = _build_rowwise_prompt_content(prompt_text)
        rowwise_columns = _extract_rowwise_columns(
            prompt_text, node_map, op_ids, inputs
        )
        msg_op = _build_message_op(scope, node_name, system_messages, msg_content)
        ops.append(msg_op)
        llm_op = _build_llm_chat_op(
            scope=scope,
            node_name=node_name,
            msg_op_id=msg_op["_id"],
            model_config=model_config,
            return_history=return_history,
            rowwise_template=msg_content,
            rowwise_columns=rowwise_columns,
            system_messages=system_messages,
        )
        ops.append(llm_op)
        return ops, llm_op["_id"]

    if op_type == "image-generation":
        if system_messages:
            prompt_text = "\n\n".join(system_messages) + "\n\n" + (prompt_text or "")
        msg_content, extra_ops = _build_prompt_content(
            prompt_text, node_name, op_ids, upstream_main, scope
        )
        ops.extend(extra_ops)
        content_op_id = msg_content if _is_op_id(msg_content) else None
        if content_op_id is None:
            data_op = _build_data_op(scope, node_name, msg_content)
            ops.append(data_op)
            content_op_id = data_op["_id"]

        img_config = model_config
        if "model" in notes:
            img_config = {"model": notes["model"]}

        llm_op = _build_image_generation_op(scope, node_name, content_op_id, img_config)
        ops.append(llm_op)
        return ops, llm_op["_id"]

    if op_type == "image-understanding":
        image_ref = notes.get("image-input")
        if not image_ref:
            raise ValueError(f"Chain node '{node_name}' missing image-input in notes")
        vision_config = vlm_config if vlm_config is not None else model_config
        image_source, _ = _resolve_image_source(image_ref, op_ids, node_name)
        image_path = notes.get("image-path")
        if not isinstance(image_path, str) or not image_path:
            image_path = "images"
        rowwise_columns = _extract_rowwise_columns(
            prompt_text, node_map, op_ids, inputs
        )
        if rowwise_columns:
            msg_content = _build_rowwise_prompt_content(prompt_text)
            extra_ops = []
            rowwise_template = msg_content
        else:
            msg_content, extra_ops = _build_prompt_content(
                prompt_text, node_name, op_ids, upstream_main, scope
            )
            rowwise_template = None
        ops.extend(extra_ops)
        msg_op = _build_message_op(scope, node_name, system_messages, msg_content)
        ops.append(msg_op)
        llm_op = _build_llm_vision_op(
            scope,
            node_name,
            msg_op["_id"],
            vision_config,
            image_source,
            str(image_path),
            return_history,
            rowwise_template=rowwise_template,
            rowwise_columns=rowwise_columns if rowwise_columns else None,
            system_messages=system_messages or None,
        )
        ops.append(llm_op)
        return ops, llm_op["_id"]

    aggregate_table = None
    if op_type == "text-generation" and mode == "aggregate":
        aggregate_prompt, aggregate_table = _build_aggregate_prompt_content(
            prompt_text, op_ids, upstream_main, node_map
        )
        msg_content, extra_ops = _build_prompt_content(
            aggregate_prompt, node_name, op_ids, upstream_main, scope
        )
    else:
        msg_content, extra_ops = _build_prompt_content(
            prompt_text, node_name, op_ids, upstream_main, scope
        )
    ops.extend(extra_ops)
    msg_op = _build_message_op(scope, node_name, system_messages, msg_content)
    ops.append(msg_op)
    llm_op = _build_llm_chat_op(
        scope,
        node_name,
        msg_op["_id"],
        model_config,
        return_history,
        structural_outputs=notes.get("structural-outputs"),
        aggregate_table=aggregate_table,
    )
    ops.append(llm_op)
    return ops, llm_op["_id"]


def _build_prompt_content(
    prompt_text: str,
    node_name: str,
    op_ids: dict[str, str],
    upstream_main: str | None,
    scope: str,
) -> tuple[str, list[dict[str, Any]]]:
    stripped = _strip_expr(prompt_text or "")
    if not stripped and upstream_main and upstream_main in op_ids:
        return op_ids[upstream_main], []

    refs = _extract_refs(stripped)
    if refs:
        template = stripped
        format_kwargs: dict[str, str] = {}
        for idx, (full, ref_name) in enumerate(refs):
            if ref_name not in op_ids:
                raise ValueError(f"Unknown reference '{ref_name}' in prompt")
            key = f"ref{idx}"
            template = template.replace(full, f"{{{key}}}")
            format_kwargs[key] = op_ids[ref_name]
        fmt_op = _build_format_op(scope, node_name, template, format_kwargs)
        return fmt_op["_id"], [fmt_op]

    return stripped, []


def _build_message_op(
    scope: str, node_name: str, system_messages: list[str], user_content: str
) -> dict[str, Any]:
    messages = []
    for msg in system_messages:
        messages.append({"role": "system", "content": msg})
    messages.append({"role": "user", "content": user_content})

    inputs = [msg["content"] for msg in messages if _is_op_id(msg["content"])]
    return {
        "_id": _make_id(scope, "message", node_name),
        "_op": "MessageOp",
        "_max_iter": None,
        "_inputs": inputs,
        "messages": messages,
    }


def _build_format_op(
    scope: str, node_name: str, template: str, format_kwargs: dict[str, str]
) -> dict[str, Any]:
    return {
        "_id": _make_id(scope, "format", node_name),
        "_op": "FormatOp",
        "_max_iter": None,
        "_inputs": list(format_kwargs.values()),
        "template": template,
        "format_args": [],
        "format_kwargs": format_kwargs,
    }


def _build_data_op(scope: str, node_name: str, content: str) -> dict[str, Any]:
    return {
        "_id": _make_id(scope, "data", node_name),
        "_op": "DataOp",
        "_max_iter": None,
        "_inputs": [],
        "data": [content],
    }


def _build_llm_chat_op(
    scope: str,
    node_name: str,
    msg_op_id: str,
    model_config: dict[str, Any],
    return_history: bool = False,
    structural_outputs: Any = None,
    aggregate_table: Any = None,
    rowwise_template: str | None = None,
    rowwise_columns: Any = None,
    system_messages: list[str] | None = None,
) -> dict[str, Any]:
    aggregate_inputs = [
        item["node"]
        for item in aggregate_table or []
        if isinstance(item, dict) and isinstance(item.get("node"), str)
    ]
    rowwise_inputs = [
        item["node"]
        for item in rowwise_columns or []
        if isinstance(item, dict) and isinstance(item.get("node"), str)
    ]
    llm_inputs = list(dict.fromkeys([msg_op_id, *aggregate_inputs, *rowwise_inputs]))

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
    if aggregate_table is not None:
        if not isinstance(aggregate_table, list) or not all(
            isinstance(item, dict) for item in aggregate_table
        ):
            raise ValueError(
                f"Chain node '{node_name}' aggregate table must be a list of objects"
            )
        op["aggregate_table"] = aggregate_table
    if rowwise_template is not None:
        op["rowwise_template"] = rowwise_template
    if rowwise_columns is not None:
        if not isinstance(rowwise_columns, list) or not all(
            isinstance(item, dict) for item in rowwise_columns
        ):
            raise ValueError(
                f"Chain node '{node_name}' rowwise columns must be a list of objects"
            )
        op["rowwise_columns"] = rowwise_columns
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
    if rowwise_columns is not None:
        op["rowwise_columns"] = rowwise_columns
    if system_messages is not None:
        op["system_messages"] = system_messages
    return op


def _resolve_image_source(
    value: Any, op_ids: dict[str, str], node_name: str
) -> tuple[str, str]:
    if isinstance(value, str):
        refs = _extract_refs(value)
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


def _resolve_data_nodes(
    node_name: str,
    incoming: dict[str, list[dict[str, str]]],
    node_map: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    postgres_nodes: list[dict[str, Any]] = []
    s3_nodes: list[dict[str, Any]] = []
    visited: set[str] = set()
    queue = [node_name]
    while queue:
        target = queue.pop(0)
        for edge in incoming.get(target, []):
            if edge.get("type") != "main":
                continue
            src = edge.get("node")
            if not src or src in visited:
                continue
            visited.add(src)
            src_node = node_map.get(src)
            if not src_node:
                continue
            node_type = src_node.get("type")
            if node_type == N8N_POSTGRES_NODE:
                postgres_nodes.append(src_node)
            elif node_type == N8N_S3_NODE:
                s3_nodes.append(src_node)
                queue.append(src)
            else:
                queue.append(src)
    if len(postgres_nodes) != 1:
        raise ValueError(
            f"n8n data-retrieval node '{node_name}' must have exactly one postgres"
            " input"
        )
    if len(s3_nodes) > 1:
        raise ValueError(
            f"n8n data-retrieval node '{node_name}' has multiple s3 inputs"
        )
    return postgres_nodes[0], s3_nodes[0] if s3_nodes else None


def _get_param_value(value: Any, field: str, node_name: str) -> str:
    if isinstance(value, dict) and "value" in value:
        value = value.get("value")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"n8n postgres node '{node_name}' missing {field}")
    return value.strip()


def _normalize_limit(
    value: Any, node_name: str, op_ids: dict[str, str]
) -> tuple[int | str | None, list[str]]:
    if value is None:
        return None, []
    if isinstance(value, bool):
        raise ValueError(f"n8n postgres node '{node_name}' limit must be an integer")
    if isinstance(value, int):
        return value, []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None, []
        resolved = _resolve_value(stripped, op_ids)
        if isinstance(resolved, str) and resolved.startswith(REF_TOKEN_PREFIX):
            ref_node_id, _ = _decode_ref_token(resolved)
            return resolved, [ref_node_id]
        if isinstance(resolved, str) and resolved.isdigit():
            return int(resolved), []
        if stripped.isdigit():
            return int(stripped), []
    raise ValueError(f"n8n postgres node '{node_name}' limit must be an integer")


def _parse_postgres_filters(
    where: Any, op_ids: dict[str, str], node_name: str
) -> tuple[list[list[list[Any]]], list[str]]:
    if where in (None, {}):
        return _normalize_filters([], op_ids, 1)
    if not isinstance(where, dict):
        raise ValueError(f"n8n postgres node '{node_name}' where must be an object")
    values = where.get("values") or []
    if not isinstance(values, list):
        raise ValueError(f"n8n postgres node '{node_name}' where.values must be a list")
    raw_filters: list[list[Any]] = []
    for entry in values:
        if not isinstance(entry, dict):
            raise ValueError(
                f"n8n postgres node '{node_name}' where.values entries must be objects"
            )
        column = entry.get("column")
        if not isinstance(column, str) or not column.strip():
            raise ValueError(
                f"n8n postgres node '{node_name}' where.values missing column"
            )
        op = (
            entry.get("operation")
            or entry.get("operator")
            or entry.get("condition")
            or "="
        )
        value = entry.get("value")
        if isinstance(value, str):
            value = _strip_expr(value)
        raw_filters.append([column, op, value])
    return _normalize_filters(raw_filters, op_ids, 1)


def _make_postgres_retrieval_op(
    scope: str,
    node_name: str,
    node: dict[str, Any],
    incoming: dict[str, list[dict[str, str]]],
    node_map: dict[str, dict[str, Any]],
    op_ids: dict[str, str],
    inputs: dict[str, list[str]],
) -> dict[str, Any]:
    params = node.get("parameters", {})
    if not params or (len(params) == 0):
        raise ValueError(
            f"n8n postgres node '{node_name}' has empty parameters; "
            "schema, table, and operation are required"
        )
    operation = params.get("operation")
    main_inputs = [
        op_ids[dep]
        for dep in _main_incoming(node_name, incoming, node_map)
        if dep in op_ids
    ]

    if operation == "executeQuery":
        query = _get_param_value(params.get("query"), "query", node_name)
        sql_template, sql_params, extra_inputs = (
            _build_execute_query_template_and_params(
                query=query,
                op_ids=op_ids,
                inputs=inputs,
            )
        )
        if not envs.LUMID_DATA_URL and not envs.DATABASE_URL:
            raise ValueError(
                f"n8n data-retrieval node '{node_name}' missing connection_strings "
                "(set DATABASE_URL for direct mode or LUMID_DATA_URL to forward "
                "through lumid.data)"
            )
        dep_inputs = [
            param["node"]
            for param in sql_params
            if isinstance(param, dict) and isinstance(param.get("node"), str)
        ]
        return {
            "_id": _make_id(scope, "retrieval", node_name),
            "_op": "DataRetrievalOp",
            "_max_iter": None,
            "_inputs": list(dict.fromkeys(extra_inputs + dep_inputs + main_inputs)),
            "data_spec": {
                "type": "sql",
                "connection_string": envs.LUMID_DATA_URL or envs.DATABASE_URL,
                "template": sql_template,
                "params": sql_params,
            },
        }

    schema = _get_param_value(params.get("schema"), "schema", node_name)
    table = _get_param_value(params.get("table"), "table", node_name)
    full_table = f"{schema}.{table}"

    limit, limit_inputs = _normalize_limit(params.get("limit"), node_name, op_ids)
    filters, extra_inputs = _parse_postgres_filters(
        params.get("where"), op_ids, node_name
    )

    options = params.get("options") or {}
    output_columns = options.get("outputColumns")
    if output_columns and not isinstance(output_columns, list):
        raise ValueError(
            f"n8n postgres node '{node_name}' options.outputColumns must be a list"
        )
    if output_columns and not all(isinstance(col, str) for col in output_columns):
        raise ValueError(
            f"n8n postgres node '{node_name}' options.outputColumns must be strings"
        )
    locked_columns = list(output_columns) if isinstance(output_columns, list) else []
    filter_columns = [str(f[0]) for f in (filters[0] if filters else []) if f]
    for col in filter_columns:
        if col not in locked_columns:
            locked_columns.append(col)
    sql_template, sql_params = _build_sql_template_and_params(
        table=full_table,
        filters=filters[0] if filters else [],
        locked_columns=locked_columns,
        limit=limit,
        op_ids=op_ids,
        inputs=inputs,
    )

    if not envs.LUMID_DATA_URL and not envs.DATABASE_URL:
        raise ValueError(
            f"n8n data-retrieval node '{node_name}' missing connection_strings "
            "(set DATABASE_URL for direct mode or LUMID_DATA_URL to forward "
            "through lumid.data)"
        )

    dep_inputs = [
        param["node"]
        for param in sql_params
        if isinstance(param, dict) and isinstance(param.get("node"), str)
    ]
    return {
        "_id": _make_id(scope, "retrieval", node_name),
        "_op": "DataRetrievalOp",
        "_max_iter": None,
        "_inputs": list(
            dict.fromkeys(extra_inputs + limit_inputs + dep_inputs + main_inputs)
        ),
        "data_spec": {
            "type": "sql",
            "connection_string": envs.LUMID_DATA_URL or envs.DATABASE_URL,
            "template": sql_template,
            "params": sql_params,
            "table": full_table,
        },
    }


def _make_s3_retrieval_op(
    scope: str,
    node_name: str,
    node: dict[str, Any],
    incoming: dict[str, list[dict[str, str]]],
    node_map: dict[str, dict[str, Any]],
    op_ids: dict[str, str],
    inputs: dict[str, list[str]],
) -> dict[str, Any]:
    params = node.get("parameters", {})
    if not params:
        raise ValueError(
            f"n8n s3 node '{node_name}' has empty parameters; "
            "bucketName and operation are required"
        )
    upstream_main = None
    for edge in incoming.get(node_name, []):
        if edge.get("type") == "main" and edge.get("node"):
            upstream_main = edge["node"]
            break
    if not upstream_main or upstream_main not in op_ids:
        raise ValueError(
            f"n8n s3 node '{node_name}' missing upstream SQL retrieval dependency"
        )
    s3_template, s3_params, param_deps = _build_s3_template_and_params(
        node=node,
        node_name=node_name,
        node_map=node_map,
        op_ids=op_ids,
        inputs=inputs,
    )
    if not envs.LUMID_DATA_URL and not envs.S3_URL:
        raise ValueError(
            f"n8n s3 node '{node_name}' missing S3 connection string "
            "(set S3_URL for direct mode or LUMID_DATA_URL to forward "
            "through lumid.data)"
        )
    _BINARY_EXTENSIONS = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".tiff",
        ".pdf",
    )
    s3_lower = s3_template.lower()
    encoding = (
        "binary"
        if any(
            s3_lower.endswith(ext) or ext + "}" in s3_lower
            for ext in _BINARY_EXTENSIONS
        )
        else "utf-8"
    )
    data_spec: dict[str, Any] = {
        "type": "s3",
        "connection_string": envs.LUMID_DATA_URL or envs.S3_URL,
        "template": s3_template,
        "params": s3_params,
        "encoding": encoding,
    }
    deps = [op_ids[upstream_main], *param_deps]
    return {
        "_id": _make_id(scope, "retrieval", node_name),
        "_op": "DataRetrievalOp",
        "_max_iter": None,
        "_inputs": list(dict.fromkeys(deps)),
        "data_spec": data_spec,
    }


def _is_lake_retrieval_node(node: dict[str, Any]) -> bool:
    return bool(node.get("parameters", {}).get("lakeRetrieval", False))


def _make_lake_retrieval_op(
    scope: str,
    node_name: str,
    node: dict[str, Any],
    incoming: dict[str, list[dict[str, str]]],
    node_map: dict[str, dict[str, Any]],
    op_ids: dict[str, str],
    inputs: dict[str, list[str]],
) -> dict[str, Any]:
    params = node.get("parameters", {})
    options = params.get("options") or {}
    if not isinstance(options, dict):
        raise ValueError(
            f"n8n lake retrieval node for '{node_name}' options must be an object"
        )
    upstream_main = None
    for edge in incoming.get(node_name, []):
        if edge.get("type") == "main" and edge.get("node"):
            upstream_main = edge["node"]
            break
    if not upstream_main or upstream_main not in op_ids:
        raise ValueError(
            f"n8n lake retrieval node '{node_name}' missing upstream SQL retrieval"
            " dependency"
        )
    s3_template, s3_params, param_deps = _build_s3_template_and_params(
        node=node,
        node_name=node_name,
        node_map=node_map,
        op_ids=op_ids,
        inputs=inputs,
    )
    if not envs.LUMID_DATA_URL and not envs.S3_URL:
        raise ValueError(
            f"n8n lake retrieval node '{node_name}' missing S3 connection string "
            "(set S3_URL for direct mode or LUMID_DATA_URL to forward "
            "through lumid.data)"
        )
    _BINARY_EXTENSIONS = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".tiff",
        ".pdf",
    )
    s3_lower = s3_template.lower()
    encoding = (
        "binary"
        if any(
            s3_lower.endswith(ext) or ext + "}" in s3_lower
            for ext in _BINARY_EXTENSIONS
        )
        else "utf-8"
    )
    data_spec: dict[str, Any] = {
        "type": "s3",
        "connection_string": envs.LUMID_DATA_URL or envs.S3_URL,
        "template": s3_template,
        "params": s3_params,
        "encoding": encoding,
    }
    deps = [op_ids[upstream_main], *param_deps]

    return {
        "_id": _make_id(scope, "retrieval", node_name),
        "_op": "DataRetrievalOp",
        "_max_iter": None,
        "_inputs": list(dict.fromkeys(deps)),
        "data_spec": data_spec,
    }


def _build_rowwise_prompt_content(prompt_text: str) -> str:
    stripped = _strip_expr(prompt_text or "")
    refs = _extract_rowwise_ref_bindings(stripped)
    template = stripped
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
    node_map: dict[str, dict[str, Any]],
    op_ids: dict[str, str],
    inputs: dict[str, list[str]],
) -> list[dict[str, Any]]:
    stripped = _strip_expr(prompt_text or "")
    columns: list[dict[str, Any]] = []
    for node_name, path, label in _extract_rowwise_ref_bindings(stripped):
        if node_name not in op_ids:
            continue
        node = node_map.get(node_name)
        if not node:
            continue
        node_type = node.get("type")
        if node_type == N8N_POSTGRES_NODE:
            column = _path_to_label(path)
            if not column:
                raise ValueError(
                    f"Rowwise reference to SQL node '{node_name}' requires a field path"
                )
            columns.append(
                {
                    "label": label,
                    "node": op_ids[node_name],
                    "path": f"items.table.{column}",
                }
            )
        elif node_type == N8N_S3_NODE:
            columns.append(
                {
                    "label": label,
                    "node": op_ids[node_name],
                    "path": "items.content",
                }
            )
        elif node_type == N8N_CHAT_TRIGGER:
            if node_name not in inputs:
                raise ValueError(
                    f"Rowwise reference to input '{node_name}' has no provided values"
                )
            columns.append(
                {
                    "label": label,
                    "data": {"type": "list", "items": inputs[node_name]},
                }
            )
        else:
            runtime_path = _to_runtime_output_path(path) or "items.output"
            columns.append(
                {
                    "label": label,
                    "node": op_ids[node_name],
                    "path": runtime_path,
                }
            )
    return columns


def _extract_rowwise_ref_bindings(
    prompt_text: str,
) -> list[tuple[str, str | None, str]]:
    bindings: list[tuple[str, str | None, str]] = []
    label_by_ref: dict[tuple[str, str | None], str] = {}
    seen_labels: set[str] = set()

    for node_name, path in _extract_ref_details(prompt_text):
        key = (node_name, path)
        label = label_by_ref.get(key)
        if label is None:
            base_label = _path_to_label(path)
            if not base_label:
                base_label = (
                    re.sub(r"[^A-Za-z0-9_]+", "_", node_name).strip("_") or "value"
                )
            label = _ensure_unique_label(base_label, seen_labels)
            label_by_ref[key] = label
        bindings.append((node_name, path, label))
    return bindings


def _build_aggregate_prompt_content(
    prompt_text: str,
    op_ids: dict[str, str],
    upstream_main: str | None,
    node_map: dict[str, dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    stripped = _strip_expr(prompt_text or "")
    table_spec: list[dict[str, Any]] = []
    table_match = TABLE_BLOCK_PATTERN.search(stripped)
    if not table_match:
        return stripped, table_spec

    block = table_match.group(1)
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        column, value = line.split(":", 1)
        column = column.strip()
        value = value.strip()
        ref_detail = _extract_ref_details(value)
        if not ref_detail:
            continue
        ref_node_name, path = ref_detail[0]
        if ref_node_name in op_ids:
            ref_node = node_map.get(ref_node_name)
            ref_type = ref_node.get("type") if isinstance(ref_node, dict) else None
            if ref_type == N8N_POSTGRES_NODE:
                item_path = f"items.table.{_path_to_label(path) or column}"
            else:
                item_path = "items.output"
            table_spec.append(
                {
                    "label": column,
                    "node": op_ids[ref_node_name],
                    "path": item_path,
                }
            )
            continue

        ref_node = node_map.get(ref_node_name)
        if ref_node and ref_node.get("type") == N8N_POSTGRES_NODE and upstream_main:
            upstream_id = op_ids.get(upstream_main)
            if upstream_id:
                table_spec.append(
                    {
                        "label": column,
                        "node": upstream_id,
                        "path": f"items.table.{_path_to_label(path) or column}",
                    }
                )

    rewritten = TABLE_BLOCK_PATTERN.sub("{df}", stripped)
    return rewritten, table_spec


def _path_to_label(path: str | None) -> str | None:
    if not path:
        return None
    normalized = path.strip()
    if normalized.startswith("item.json."):
        normalized = normalized[len("item.json.") :]
    elif normalized.startswith("item."):
        normalized = normalized[len("item.") :]
    elif normalized == "item":
        return "output"
    if not normalized:
        return None
    return normalized.split(".")[-1]


def _build_image_generation_op(
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


def _make_code_ops(
    node: dict[str, Any],
    node_name: str,
    incoming: dict[str, list[dict[str, str]]],
    node_map: dict[str, dict[str, Any]],
    op_ids: dict[str, str],
    scope: str,
) -> tuple[list[dict[str, Any]], str]:
    notes = _parse_notes(node.get("notes", ""))

    input_names = notes.get("inputs")
    if input_names is None or input_names == []:
        input_names = _main_incoming(node_name, incoming, node_map)
    if not isinstance(input_names, list) or not input_names:
        raise ValueError(f"Code node '{node_name}' missing inputs")

    params = node.get("parameters", {})
    code = params.get("pythonCode")
    if not isinstance(code, str) or not code.strip():
        raise ValueError(f"Code node '{node_name}' missing pythonCode")

    input_ids: list[str] = []
    for name in input_names:
        if name not in op_ids:
            raise ValueError(f"Code node '{node_name}' unknown input '{name}'")
        input_ids.append(op_ids[name])

    fn_name = _safe_fn_name(node_name)
    fn_code = _wrap_code(fn_name, code)
    op: dict[str, Any] = {
        "_id": _make_id(scope, "lambda", node_name),
        "_op": "LambdaOp",
        "_max_iter": None,
        "_inputs": input_ids,
        "fn_name": fn_name,
        "_code": fn_code,
    }
    return [op], op["_id"]


def _make_output_op(scope: str, name: str, upstream_id: str) -> dict[str, Any]:
    return {
        "_id": _make_id(scope, "output", name),
        "_op": "OutputOp",
        "_max_iter": None,
        "_inputs": [upstream_id],
        "name": name,
    }


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


def _parse_model_params(model: str, options: Any) -> dict[str, Any]:
    config: dict[str, Any] = {"model": model}
    if isinstance(options, dict):
        if "maxTokens" in options:
            config["max_tokens"] = options.get("maxTokens")
        if "temperature" in options:
            config["temperature"] = options.get("temperature")
        if "topP" in options:
            config["top_p"] = options.get("topP")
    return config


def _normalize_filters(
    filters_raw: Any, op_ids: dict[str, str], table_count: int
) -> tuple[list[list[list[Any]]], list[str]]:
    if filters_raw is None or filters_raw == []:
        return [[] for _ in range(table_count or 1)], []

    if not isinstance(filters_raw, list):
        raise ValueError("n8n filters must be a list")

    filters = filters_raw
    if filters and isinstance(filters[0], (list, tuple)) and len(filters[0]) == 3:
        filters = [filters]

    normalized: list[list[list[Any]]] = []
    extra_inputs: list[str] = []
    for table_filters in filters:
        if not isinstance(table_filters, list):
            raise ValueError("n8n filters must be list of lists")
        table_list: list[list[Any]] = []
        for item in table_filters:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise ValueError("n8n filter entries must be [column, op, value]")
            col, op, val = item
            resolved_val = _resolve_value(val, op_ids)
            if _is_op_id(resolved_val):
                extra_inputs.append(resolved_val)
            elif isinstance(resolved_val, str) and resolved_val.startswith(
                REF_TOKEN_PREFIX
            ):
                ref_node_id, _ = _decode_ref_token(resolved_val)
                extra_inputs.append(ref_node_id)
            table_list.append([col, op, resolved_val])
        normalized.append(table_list)

    if table_count and len(normalized) != table_count:
        raise ValueError("filters length must match tables length")

    return normalized, extra_inputs


def _build_sql_template_and_params(
    table: str,
    filters: list[list[Any]],
    locked_columns: list[str],
    limit: int | str | None,
    op_ids: dict[str, str],
    inputs: dict[str, list[str]],
) -> tuple[str, list[dict[str, Any]]]:
    select_cols = (
        ", ".join(_quote_column_ref(col) for col in locked_columns)
        if locked_columns
        else "*"
    )
    query = f"SELECT {select_cols} FROM {_quote_table_ref(table)}"
    query_params: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    input_names_by_op_id = {
        op_id: name for name, op_id in op_ids.items() if name in inputs
    }

    def add_param(label: str, payload: dict[str, Any]) -> None:
        if label in seen_labels:
            return
        seen_labels.add(label)
        query_params.append({"label": label, **payload})

    where_clauses: list[str] = []
    for raw_filter in filters:
        if not isinstance(raw_filter, (list, tuple)) or len(raw_filter) != 3:
            continue
        col, op, val = raw_filter
        clause_label = _sql_param_label(str(col), str(op))
        placeholder = f"{{{clause_label}}}"
        resolved_val = val

        if isinstance(resolved_val, str) and resolved_val.startswith(REF_TOKEN_PREFIX):
            node_id, path = _decode_ref_token(resolved_val)
            add_param(clause_label, {"node": node_id, "path": path})
            where_clauses.append(f"{_quote_column_ref(str(col))} {op} '{placeholder}'")
            continue
        if isinstance(resolved_val, str) and _is_op_id(resolved_val):
            input_name = input_names_by_op_id.get(resolved_val)
            if input_name:
                add_param(
                    clause_label,
                    {"data": {"type": "list", "items": inputs[input_name]}},
                )
            else:
                add_param(clause_label, {"node": resolved_val, "path": "items.output"})
            where_clauses.append(f"{_quote_column_ref(str(col))} {op} '{placeholder}'")
            continue
        if isinstance(resolved_val, (list, tuple)):
            values = ", ".join(_sql_literal(item) for item in resolved_val)
            where_clauses.append(f"{_quote_column_ref(str(col))} {op} ({values})")
            continue

        where_clauses.append(
            f"{_quote_column_ref(str(col))} {op} {_sql_literal(resolved_val)}"
        )

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    if isinstance(limit, int):
        query += f" LIMIT {limit}"
    elif isinstance(limit, str) and limit.startswith(REF_TOKEN_PREFIX):
        node_id, path = _decode_ref_token(limit)
        add_param("limit", {"node": node_id, "path": path})
        query += " LIMIT {limit}"
    elif isinstance(limit, str) and _is_op_id(limit):
        input_name = input_names_by_op_id.get(limit)
        if input_name:
            add_param("limit", {"data": {"type": "list", "items": inputs[input_name]}})
        else:
            add_param("limit", {"node": limit, "path": "items.output"})
        query += " LIMIT {limit}"
    elif limit is not None:
        query += f" LIMIT {_sql_literal(limit)}"

    return query, query_params


def _build_execute_query_template_and_params(
    query: str, op_ids: dict[str, str], inputs: dict[str, list[str]]
) -> tuple[str, list[dict[str, Any]], list[str]]:
    template = _strip_expr(query).strip()
    params: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    dependencies: list[str] = []
    input_names_by_op_id = {
        op_id: name for name, op_id in op_ids.items() if name in inputs
    }

    def add_param(label: str, payload: dict[str, Any]) -> None:
        if label in seen_labels:
            return
        seen_labels.add(label)
        params.append({"label": label, **payload})

    def replacement(match: re.Match[str]) -> str:
        expr = match.group(1).strip()
        if not expr:
            raise ValueError("n8n postgres executeQuery contains empty expression")
        resolved = _resolve_value(expr, op_ids)
        if isinstance(resolved, str) and resolved.startswith(REF_TOKEN_PREFIX):
            node_id, path = _decode_ref_token(resolved)
            label = _next_sql_label(f"ref_{node_id}", seen_labels)
            add_param(label, {"node": node_id, "path": path})
            dependencies.append(node_id)
            return f"{{{label}}}"
        if isinstance(resolved, str) and _is_op_id(resolved):
            label = _next_sql_label(f"ref_{resolved}", seen_labels)
            input_name = input_names_by_op_id.get(resolved)
            if input_name:
                add_param(
                    label, {"data": {"type": "list", "items": inputs[input_name]}}
                )
            else:
                add_param(label, {"node": resolved, "path": "items.output"})
                dependencies.append(resolved)
            return f"{{{label}}}"
        return _sql_literal(resolved)

    templated = QUERY_EXPR_PATTERN.sub(replacement, template)
    return templated, params, list(dict.fromkeys(dependencies))


def _next_sql_label(base: str, seen_labels: set[str]) -> str:
    label = re.sub(r"[^A-Za-z0-9_]", "_", base).strip("_") or "param"
    if label[0].isdigit():
        label = f"p_{label}"
    if label not in seen_labels:
        return label
    idx = 2
    while f"{label}_{idx}" in seen_labels:
        idx += 1
    return f"{label}_{idx}"


def _build_s3_template_and_params(
    node: dict[str, Any],
    node_name: str,
    node_map: dict[str, dict[str, Any]],
    op_ids: dict[str, str],
    inputs: dict[str, list[str]],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    params = node.get("parameters", {})
    prefix = params.get("bucketName")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError(f"n8n s3 node for '{node_name}' missing bucketName")
    options = params.get("options") or {}
    if not isinstance(options, dict):
        raise ValueError(f"n8n s3 node for '{node_name}' options must be an object")
    folder_key = options.get("folderKey")
    if not isinstance(folder_key, str) or not folder_key.strip():
        raise ValueError(f"n8n s3 node for '{node_name}' missing options.folderKey")

    key_template, key_params, param_deps = _parse_s3_folder_key(
        folder_key=folder_key,
        node_name=node_name,
        node_map=node_map,
        op_ids=op_ids,
        inputs=inputs,
    )
    normalized_prefix = prefix.strip().strip("/")
    normalized_key = key_template.strip().lstrip("/")
    full_template = (
        f"{normalized_prefix}/{normalized_key}" if normalized_key else normalized_prefix
    )
    return full_template, key_params, param_deps


def _parse_s3_folder_key(
    folder_key: str,
    node_name: str,
    node_map: dict[str, dict[str, Any]],
    op_ids: dict[str, str],
    inputs: dict[str, list[str]],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    stripped = _strip_expr(folder_key).strip()
    stripped = re.sub(r"\$\s*\{\{", "{{", stripped)
    if not stripped:
        raise ValueError(f"n8n s3 node for '{node_name}' folderKey cannot be empty")

    refs = _extract_refs(stripped)
    if not refs:
        return stripped.lstrip("/"), [], []

    input_names_by_op_id = {
        op_id: name for name, op_id in op_ids.items() if name in inputs
    }
    template = stripped
    key_params: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    dependencies: list[str] = []

    for idx, (full_literal, ref_name) in enumerate(refs):
        if ref_name not in op_ids:
            raise ValueError(
                f"n8n s3 node for '{node_name}' folderKey references unknown node"
                f" '{ref_name}'"
            )
        detail_matches = _extract_ref_details(full_literal)
        ref_path = detail_matches[0][1] if detail_matches else None
        base_label = _path_to_label(ref_path) or f"value{idx}"
        label = _ensure_unique_label(base_label, seen_labels)
        template = template.replace(full_literal, f"{{{label}}}", 1)

        node_id = op_ids[ref_name]
        input_name = input_names_by_op_id.get(node_id)
        if input_name:
            key_params.append(
                {
                    "label": label,
                    "data": {"type": "list", "items": inputs[input_name]},
                }
            )
            continue

        node_type = (node_map.get(ref_name) or {}).get("type")
        path = _s3_param_path(node_type, ref_path)
        key_params.append({"label": label, "node": node_id, "path": path})
        dependencies.append(node_id)

    return template.lstrip("/"), key_params, dependencies


def _s3_param_path(node_type: Any, ref_path: str | None) -> str:
    if node_type == N8N_POSTGRES_NODE:
        column = _path_to_label(ref_path)
        if not column:
            raise ValueError("S3 folderKey references a SQL node without a field path")
        return f"items.table.{column}"
    if node_type == N8N_S3_NODE:
        label = _path_to_label(ref_path)
        if label in {None, "content"}:
            return "items.content"
    runtime_path = _to_runtime_output_path(ref_path)
    return runtime_path or "items.output"


def _ensure_unique_label(label: str, seen: set[str]) -> str:
    if label not in seen:
        seen.add(label)
        return label
    idx = 2
    while f"{label}_{idx}" in seen:
        idx += 1
    unique = f"{label}_{idx}"
    seen.add(unique)
    return unique


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _quote_identifier(ident: str) -> str:
    stripped = ident.strip()
    if not stripped:
        return stripped
    if stripped.startswith('"') and stripped.endswith('"'):
        return stripped
    if any(ch.isupper() for ch in stripped):
        return f'"{stripped}"'
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", stripped):
        return stripped
    return f'"{stripped}"'


def _quote_table_ref(table: str) -> str:
    parts = [part.strip() for part in table.split(".")]
    return ".".join(_quote_identifier(part) for part in parts if part)


def _quote_column_ref(column: str) -> str:
    parts = [part.strip() for part in column.split(".")]
    return ".".join(_quote_identifier(part) for part in parts if part)


def _sql_param_label(column: str, operator: str) -> str:
    clean_column = re.sub(r"[^A-Za-z0-9_]", "", column) or "value"
    op = operator.strip()
    if op in {">", ">="}:
        return f"{clean_column}Min"
    if op in {"<", "<="}:
        return f"{clean_column}Max"
    return clean_column


def _parse_notes(notes: Any) -> dict[str, Any]:
    if notes is None or notes == "":
        return {}
    if isinstance(notes, dict):
        return notes
    if not isinstance(notes, str):
        raise ValueError("n8n notes must be a JSON string or object")
    raw = notes.strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in n8n notes: {exc}") from exc


def _extract_refs(text: str) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for match in REF_PATTERN.finditer(text):
        full = match.group(0)
        name = match.group(1) or match.group(2)
        if name:
            refs.append((full, name))
    return refs


def _extract_ref_details(text: str) -> list[tuple[str, str | None]]:
    refs: list[tuple[str, str | None]] = []
    for match in REF_DETAIL_PATTERN.finditer(text):
        name = match.group(1)
        path = match.group(2)
        if name:
            refs.append((name, path))
    return refs


def _extract_ref_names(value: Any) -> list[str]:
    names: list[str] = []
    if isinstance(value, str):
        names.extend(name for _, name in _extract_refs(value))
    elif isinstance(value, list):
        for item in value:
            names.extend(_extract_ref_names(item))
    elif isinstance(value, dict):
        for item in value.values():
            names.extend(_extract_ref_names(item))
    return names


def _resolve_value(value: Any, op_ids: dict[str, str]) -> Any:
    if isinstance(value, str):
        refs = _extract_ref_details(_strip_expr(value))
        if refs:
            name, path = refs[0]
            if name in op_ids:
                op_id = op_ids[name]
                runtime_path = _to_runtime_output_path(path)
                if runtime_path:
                    return _encode_ref_token(op_id, runtime_path)
                return op_id
            raise ValueError(f"Unknown reference '{name}' in filters")
    return value


def _to_runtime_output_path(path: str | None) -> str | None:
    if not path:
        return None
    normalized = path.strip()
    if not normalized:
        return None
    if normalized == "item":
        return "items.output"
    if normalized.startswith("item.json."):
        suffix = normalized[len("item.json.") :]
        return f"items.output.{suffix}" if suffix else "items.output"
    if normalized.startswith("item."):
        suffix = normalized[len("item.") :]
        return f"items.output.{suffix}" if suffix else "items.output"
    return f"items.output.{normalized}"


def _encode_ref_token(node_id: str, path: str) -> str:
    return f"{REF_TOKEN_PREFIX}{node_id}:{path}"


def _decode_ref_token(token: str) -> tuple[str, str]:
    payload = token[len(REF_TOKEN_PREFIX) :]
    node_id, _, path = payload.partition(":")
    if not node_id or not path:
        raise ValueError(f"Invalid reference token: {token}")
    return node_id, path


def _strip_expr(value: str) -> str:
    if isinstance(value, str) and value.startswith("="):
        return value[1:]
    return value


def _is_op_id(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(
        (
            "input_",
            "llm_",
            "llmdata_",
            "imagegen_",
            "llmvision_",
            "format_",
            "data_",
            "lambda_",
            "retrieval_",
        )
    )


def _safe_fn_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "lambda_fn"
    if slug[0].isdigit():
        slug = f"fn_{slug}"
    return slug


def _wrap_code(fn_name: str, code: str) -> str:
    stripped = code.strip()
    if stripped.startswith(("def ", "lambda ")):
        return stripped
    lines = stripped.splitlines() or [""]
    indented = "\n".join("    " + line for line in lines)
    return f"def {fn_name}(args):\n{indented}"
