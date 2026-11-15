"""Resolving a reference from one n8n node to another.

A prompt or query that names another node resolves through that node
type's :class:`~.entries.NodeReference`, so the reference rules sit beside
the translator for the type they describe.
"""

from . import shared
from .entries import ColumnRef, NodeDict, NodeReference, OpContext


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


def _node_type(ctx: OpContext, node_name: str) -> str | None:
    node = ctx.node_map.get(node_name)
    return node.get("type") if isinstance(node, dict) else None


def _generic_column(
    ctx: OpContext, node_name: str, label: str, path: str | None
) -> NodeDict:
    return {
        "label": label,
        "node": ctx.op_ids[node_name],
        "path": shared.to_runtime_output_path(path) or "items.output",
    }


def column(ctx: OpContext, node_name: str, label: str, path: str | None) -> NodeDict:
    """Build one column for a referenced node, by that node type's rule."""
    reference = _reference_for(_node_type(ctx, node_name))
    ref = ColumnRef(node_name=node_name, label=label, path=path, ctx=ctx)
    if reference is not None and reference.column is not None:
        return reference.column(ref)
    return _generic_column(ctx, node_name, label, path)


def table_column(
    ctx: OpContext, node_name: str, label: str, path: str | None
) -> NodeDict:
    """Build one aggregate-table column for a referenced node."""
    reference = _reference_for(_node_type(ctx, node_name))
    ref = ColumnRef(node_name=node_name, label=label, path=path, ctx=ctx)
    if reference is not None and reference.table_column is not None:
        return reference.table_column(ref)
    return _generic_column(ctx, node_name, label, path)


def retrieval_mode(ctx: OpContext, node_name: str) -> str | None:
    """The retrieval mode a referenced node carries, when it is a source."""
    return retrieval_mode_for_type(_node_type(ctx, node_name))


def retrieval_mode_for_type(node_type: str | None) -> str | None:
    reference = _reference_for(node_type)
    return reference.retrieval_mode if reference is not None else None
