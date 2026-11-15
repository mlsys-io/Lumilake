"""Resolving a reference from one n8n node to another.

A prompt or query that names another node resolves through that node
type's :class:`~.entries.NodeReference`, so the reference rules sit beside
the translator for the type they describe.
"""

from . import shared
from .entries import ColumnRef, NodeDict, NodeMap, NodeReference


def _reference_for(node_type: str | None) -> NodeReference | None:
    # Imported here: the table holds the translators of the modules that
    # call this one.
    from .tables import NODE_TRANSLATORS

    translator = NODE_TRANSLATORS.get(node_type or "")
    return translator.reference if translator is not None else None


def items_path(node_type: str | None, ref_path: str | None) -> str:
    """Resolve the items path for a node referenced as a query parameter."""
    reference = _reference_for(node_type)
    if reference is not None and reference.items_path is not None:
        resolved = reference.items_path(ref_path)
        if resolved is not None:
            return resolved
    runtime_path = shared.to_runtime_output_path(ref_path)
    return runtime_path or "items.output"


def _node_type(node_map: NodeMap, node_name: str) -> str | None:
    node = node_map.get(node_name)
    return node.get("type") if isinstance(node, dict) else None


def _make_ref(
    node_name: str,
    label: str,
    path: str | None,
    op_ids: dict[str, str],
    inputs: dict[str, list[str]],
) -> ColumnRef:
    return ColumnRef(
        node_name=node_name,
        label=label,
        path=path,
        op_ids=op_ids,
        inputs=inputs,
    )


def _generic_column(ref: ColumnRef) -> NodeDict:
    return {
        "label": ref.label,
        "node": ref.op_ids[ref.node_name],
        "path": shared.to_runtime_output_path(ref.path) or "items.output",
    }


def column(
    node_map: NodeMap,
    op_ids: dict[str, str],
    inputs: dict[str, list[str]],
    node_name: str,
    label: str,
    path: str | None,
) -> NodeDict:
    """Build one rowwise column for a referenced node."""
    ref = _make_ref(node_name, label, path, op_ids, inputs)
    reference = _reference_for(_node_type(node_map, node_name))
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
    ref = _make_ref(node_name, label, path, op_ids, {})
    reference = _reference_for(_node_type(node_map, node_name))
    rule = reference.table_column if reference is not None else None
    return rule(ref) if rule is not None else _generic_column(ref)


def retrieval_mode(node_map: NodeMap, node_name: str) -> str | None:
    """The retrieval mode a referenced node carries, when it is a source."""
    return retrieval_mode_for_type(_node_type(node_map, node_name))


def retrieval_mode_for_type(node_type: str | None) -> str | None:
    reference = _reference_for(node_type)
    return reference.retrieval_mode if reference is not None else None
