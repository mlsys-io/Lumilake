"""Parse n8n workflows into Lumilake graph specs.

This module keeps the graph walk: node indexing, edge handling, dependency
ordering and output wiring. What each node type turns into lives in
:mod:`lumilake_server.parser.n8n_nodes`, reached through the tables in
:mod:`lumilake_server.parser.n8n_nodes.tables`.
"""

from dataclasses import dataclass
from typing import Any

from .common import make_id as _make_id
from .n8n_nodes import shared
from .n8n_nodes.constants import N8N_CHAT_TRIGGER, N8N_MODEL_SPEC
from .n8n_nodes.entries import OpContext
from .n8n_nodes.tables import (
    INPUT_TYPES,
    NODE_TRANSLATORS,
    NODE_TYPES,
    OUTPUT_TYPES,
    WORK_TYPES,
)


@dataclass
class N8NGraphSpec:
    graph: dict[str, Any]
    inputs: dict[str, list[str]]


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


def parse_n8n_payload(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Parse an n8n request payload into Lumilake-style graph specs.

    Expected payload (flexible):
    {
      "graphs": [
        {
          "workflow": { ... n8n workflow json ... },
          "inputs": { "stock": ["NVDA"] },
          "name": "g0__slice_1",        # routing key; also default scope
          "scope": "g0"                  # optional; DSL-id scope override
        },
        ...
      ]
    }

    Every graph carries a job seed like a YAML document does; an n8n export
    names none, so every graph records 0.
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
        scope = item.get("scope") or graph_name
        compiled = _parse_n8n_workflow(workflow, inputs, scope)
        graph_specs[graph_name] = {
            "graph": compiled.graph,
            "inputs": compiled.inputs,
            "seed": 0,
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

    unsupported = [
        name for name, node in node_map.items() if node.get("type") not in NODE_TYPES
    ]
    if unsupported:
        raise ValueError(f"Unsupported n8n node types: {unsupported}")

    ordered_ops: list[dict[str, Any]] = []
    graph_ops: dict[str, dict[str, Any]] = {}
    op_ids: dict[str, str] = {}
    ctx = OpContext(
        scope=scope,
        inputs=inputs,
        incoming=incoming,
        node_map=node_map,
        op_ids=op_ids,
        model_specs=model_specs,
    )

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
        if node["type"] not in INPUT_TYPES:
            continue
        ops, op_id = NODE_TRANSLATORS[node["type"]].make(node, ctx, node_name)
        for op in ops:
            add_op(op)
        op_ids[node_name] = op_id

    pending = {name for name, node in node_map.items() if node["type"] in WORK_TYPES}
    while pending:
        progress = False
        for node_name in list(pending):
            node = node_map[node_name]
            deps = NODE_TRANSLATORS[node["type"]].dependencies(node_name, ctx)
            if any(dep in pending for dep in deps):
                continue
            if any(dep not in op_ids for dep in deps):
                continue

            ops, op_id = NODE_TRANSLATORS[node["type"]].make(node, ctx, node_name)
            for op in ops:
                add_op(op)
            op_ids[node_name] = op_id
            pending.remove(node_name)
            progress = True
        if not progress:
            missing = ", ".join(sorted(pending))
            raise ValueError(f"Unresolved n8n dependencies: {missing}")

    # Outputs last, type by type in table order, so chain outputs come
    # before code outputs whatever order the nodes are declared in.
    output_owners: dict[str, str] = {}
    for node_name, node in (
        (name, node)
        for node_type in OUTPUT_TYPES
        for name, node in node_map.items()
        if node["type"] == node_type
    ):
        notes = shared.parse_notes(node.get("notes", ""))
        if not notes.get("is-output", False):
            continue
        output_name = notes.get("output-field") or node_name
        upstream = op_ids.get(node_name)
        if not upstream:
            raise ValueError(f"Output node {node_name} missing op id")
        owner = output_owners.get(output_name)
        if owner is not None:
            raise ValueError(
                f"Duplicate output name '{output_name}' from nodes "
                f"'{owner}' and '{node_name}'"
            )
        output_owners[output_name] = node_name
        output_op = make_output_op(scope, output_name, upstream)
        add_op(output_op)

    graph = {op["_id"]: op for op in ordered_ops}
    return N8NGraphSpec(graph=graph, inputs=inputs)


def make_output_op(scope: str, name: str, upstream_id: str) -> dict[str, Any]:
    return {
        "_id": _make_id(scope, "output", name),
        "_op": "OutputOp",
        "_max_iter": None,
        "_inputs": [upstream_id],
        "name": name,
    }


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

        notes = shared.parse_notes(node.get("notes", ""))
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
