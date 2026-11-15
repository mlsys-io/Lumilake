"""How one n8n node type answers a reference from another node.

References are keyed by the same type strings the walk uses.
"""

from typing import Any

from lumilake_server.common import InputRef, project_input_values, retrieval_items_path

from .constants import N8N_AGENT_NODE, N8N_CHAT_TRIGGER, N8N_POSTGRES_NODE, N8N_S3_NODE
from .entries import ColumnRef, Incoming, NodeDict, NodeMap, NodeReference


def _postgres_items_path(ref_path: str | None) -> str | None:
    column = path_to_label(ref_path)
    if not column:
        raise ValueError("Query parameter references a SQL node without a field path")
    return f"{retrieval_items_path('sql')}.{column}"


def _s3_items_path(ref_path: str | None) -> str | None:
    if path_to_label(ref_path) in {None, "content"}:
        return retrieval_items_path("s3")
    return None


def _agent_items_path(_: str | None) -> str:
    return retrieval_items_path("agent")


def _postgres_column(ref: ColumnRef) -> NodeDict:
    column = path_to_label(ref.path)
    if not column:
        raise ValueError(
            f"Rowwise reference to SQL node '{ref.node_name}' requires a field path"
        )
    return {
        "label": ref.label,
        "node": ref.op_ids[ref.node_name],
        "path": f"{retrieval_items_path('sql')}.{column}",
    }


def _s3_column(ref: ColumnRef) -> NodeDict:
    return {
        "label": ref.label,
        "node": ref.op_ids[ref.node_name],
        "path": retrieval_items_path("s3"),
    }


def _agent_column(ref: ColumnRef) -> NodeDict:
    return {
        "label": ref.label,
        "node": ref.op_ids[ref.node_name],
        "path": retrieval_items_path("agent"),
    }


def _postgres_table_column(ref: ColumnRef) -> NodeDict:
    suffix = path_to_label(ref.path) or ref.label
    return {
        "label": ref.label,
        "node": ref.op_ids[ref.node_name],
        "path": f"{retrieval_items_path('sql')}.{suffix}",
    }


def _agent_table_column(ref: ColumnRef) -> NodeDict:
    return {
        "label": ref.label,
        "node": ref.op_ids[ref.node_name],
        "path": retrieval_items_path("agent"),
    }


def _trigger_column(ref: ColumnRef) -> NodeDict:
    values = ref.inputs.get(ref.node_name)
    if values is None:
        raise ValueError(
            f"Rowwise reference to input '{ref.node_name}' has no provided values"
        )
    projected = project_input_values(values, ref.path, ref.node_name)
    input_ref: InputRef = {"key": ref.node_name, "path": ref.path}
    data: dict[str, object] = {"type": "list", "items": projected}
    # A multi-row input ref groups one projected value per input row.
    if len(values) > 1:
        data["grouped"] = True
    return {"label": ref.label, "data": data, "input_ref": input_ref}


def to_runtime_output_path(path: str | None) -> str | None:
    if not path:
        return None
    normalized = path.strip()
    if not normalized:
        return None
    if normalized in {"item", "item.output"}:
        return "items.output"
    if normalized.startswith("item.output."):
        suffix = normalized[len("item.output.") :]
        return f"items.output.{suffix}" if suffix else "items.output"
    if normalized.startswith("item.json."):
        suffix = normalized[len("item.json.") :]
        return f"items.output.{suffix}" if suffix else "items.output"
    if normalized.startswith("item."):
        suffix = normalized[len("item.") :]
        return f"items.output.{suffix}" if suffix else "items.output"
    return f"items.output.{normalized}"


def path_to_label(path: str | None) -> str | None:
    if not path:
        return None
    normalized = path.strip()
    if normalized in {"item", "item.output", "item.json"}:
        return "output"
    if normalized.startswith("item.json."):
        normalized = normalized[len("item.json.") :]
    elif normalized.startswith("item."):
        normalized = normalized[len("item.") :]
    if not normalized:
        return None
    return normalized.split(".")[-1]


NODE_REFERENCES: dict[str, NodeReference] = {
    N8N_CHAT_TRIGGER: NodeReference(column=_trigger_column),
    N8N_POSTGRES_NODE: NodeReference(
        items_path=_postgres_items_path,
        column=_postgres_column,
        table_column=_postgres_table_column,
        retrieval_mode="sql",
    ),
    N8N_S3_NODE: NodeReference(
        items_path=_s3_items_path, column=_s3_column, retrieval_mode="s3"
    ),
    N8N_AGENT_NODE: NodeReference(
        items_path=_agent_items_path,
        column=_agent_column,
        table_column=_agent_table_column,
        retrieval_mode="agent",
    ),
}


def _node_type(node_map: NodeMap, node_name: str) -> str | None:
    node = node_map.get(node_name)
    return node.get("type") if isinstance(node, dict) else None


def _generic_column(ref: ColumnRef) -> NodeDict:
    return {
        "label": ref.label,
        "node": ref.op_ids[ref.node_name],
        "path": to_runtime_output_path(ref.path) or "items.output",
    }


def items_path(node_type: str | None, ref_path: str | None) -> str:
    """Resolve the items path for a node referenced as a query parameter."""
    reference = NODE_REFERENCES.get(node_type or "")
    if reference is not None and reference.items_path is not None:
        resolved = reference.items_path(ref_path)
        if resolved is not None:
            return resolved
    runtime_path = to_runtime_output_path(ref_path)
    return runtime_path or "items.output"


def column(
    node_map: NodeMap,
    op_ids: dict[str, str],
    inputs: dict[str, list[str]],
    node_name: str,
    label: str,
    path: str | None,
) -> NodeDict:
    """Build one rowwise column for a referenced node."""
    ref = ColumnRef(
        node_name=node_name, label=label, path=path, op_ids=op_ids, inputs=inputs
    )
    reference = NODE_REFERENCES.get(_node_type(node_map, node_name) or "")
    rule = reference.column if reference is not None else None
    return rule(ref) if rule is not None else _generic_column(ref)


def table_column(
    node_map: NodeMap,
    op_ids: dict[str, str],
    node_name: str,
    label: str,
    path: str | None,
) -> NodeDict:
    """Build one aggregate-table column for a referenced node."""
    ref = ColumnRef(
        node_name=node_name, label=label, path=path, op_ids=op_ids, inputs={}
    )
    reference = NODE_REFERENCES.get(_node_type(node_map, node_name) or "")
    rule = reference.table_column if reference is not None else None
    return rule(ref) if rule is not None else _generic_column(ref)


def retrieval_mode(node_map: NodeMap, node_name: str) -> str | None:
    """The retrieval mode a referenced node carries, when it is a source."""
    return retrieval_mode_for_type(_node_type(node_map, node_name))


def retrieval_mode_for_type(node_type: str | None) -> str | None:
    reference = NODE_REFERENCES.get(node_type or "")
    return reference.retrieval_mode if reference is not None else None


def is_data_source_node(node_map: NodeMap, name: str) -> bool:
    """Whether the named node is a data source rather than an upstream op."""
    node = node_map.get(name)
    if not node:
        return False
    reference = NODE_REFERENCES.get(node.get("type") or "")
    return reference is not None and reference.data_source


def find_main_upstream(
    node_name: str,
    incoming: Incoming,
    node_map: NodeMap,
) -> str | None:
    for edge in incoming.get(node_name, []):
        if edge.get("type") != "main":
            continue
        upstream = edge.get("node")
        if not upstream:
            continue
        if is_data_source_node(node_map, upstream):
            continue
        return upstream
    return None


def main_incoming(
    node_name: str,
    incoming: Incoming,
    node_map: NodeMap,
) -> list[str]:
    return [
        edge["node"]
        for edge in incoming.get(node_name, [])
        if edge.get("type") == "main"
        and edge.get("node")
        and not is_data_source_node(node_map, edge["node"])
    ]


def resolve_data_nodes(
    node_name: str,
    incoming: Incoming,
    node_map: NodeMap,
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
            mode = retrieval_mode_for_type(src_node.get("type"))
            if mode == "sql":
                postgres_nodes.append(src_node)
            else:
                if mode == "s3":
                    s3_nodes.append(src_node)
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
