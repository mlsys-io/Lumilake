"""The two dispatch tables: n8n node types and chain notes ``(op-type,
mode)`` pairs.

Every key is the value the walk already dispatches on, so adding a node
type or a chain op-type is a table entry, not an edit to the walk.
"""

from lumilake_server.common import InputRef, project_input_values, retrieval_items_path

from . import code, llm, retrieval, shared
from .entries import (
    ChainBuilder,
    ColumnRef,
    NodeDict,
    NodeReference,
    NodeTranslator,
    OpContext,
)


def _postgres_items_path(ref_path: str | None) -> str | None:
    column = shared.path_to_label(ref_path)
    if not column:
        raise ValueError("Query parameter references a SQL node without a field path")
    return f"{retrieval_items_path('sql')}.{column}"


def _agent_items_path(_: str | None) -> str:
    return retrieval_items_path("agent")


def _s3_items_path(ref_path: str | None) -> str | None:
    if shared.path_to_label(ref_path) in {None, "content"}:
        return retrieval_items_path("s3")
    return None


def _postgres_column(ref: ColumnRef) -> NodeDict:
    column = shared.path_to_label(ref.path)
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
    suffix = shared.path_to_label(ref.path) or ref.label
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


def make_chain_node(
    node: NodeDict, ctx: OpContext, node_name: str
) -> tuple[list[NodeDict], str]:
    """Dispatch a chain node to the builder its notes select."""
    plan = llm.chain_plan(node, ctx, node_name)
    op_type = plan.notes["op-type"]
    builder = CHAIN_BUILDERS.get((op_type, plan.notes.get("mode")))
    if builder is None:
        builder = CHAIN_BUILDERS.get((op_type, None), llm.build_text_generation)
    return builder(plan)


NODE_TRANSLATORS: dict[str, NodeTranslator] = {
    shared.N8N_CHAIN_TYPE: NodeTranslator(
        make=make_chain_node,
        dependencies=llm.chain_dependencies,
        can_be_output=True,
    ),
    shared.N8N_CHAT_TRIGGER: NodeTranslator(
        make=code.make_trigger,
        dependencies=code.trigger_dependencies,
        is_input=True,
        reference=NodeReference(column=_trigger_column),
    ),
    shared.N8N_CODE_NODE: NodeTranslator(
        make=code.make_code,
        dependencies=code.code_dependencies,
        can_be_output=True,
    ),
    shared.N8N_POSTGRES_NODE: NodeTranslator(
        make=retrieval.make_postgres,
        dependencies=retrieval.postgres_dependencies,
        is_data_source=True,
        reference=NodeReference(
            items_path=_postgres_items_path,
            column=_postgres_column,
            table_column=_postgres_table_column,
            retrieval_mode="sql",
        ),
    ),
    shared.N8N_S3_NODE: NodeTranslator(
        make=retrieval.make_s3,
        dependencies=retrieval.s3_dependencies,
        is_data_source=True,
        reference=NodeReference(
            items_path=_s3_items_path, column=_s3_column, retrieval_mode="s3"
        ),
    ),
    shared.N8N_AGENT_NODE: NodeTranslator(
        make=llm.make_agent,
        dependencies=llm.agent_dependencies,
        reference=NodeReference(
            items_path=_agent_items_path,
            column=_agent_column,
            table_column=_agent_table_column,
            retrieval_mode="agent",
        ),
    ),
}

CHAIN_BUILDERS: dict[tuple[str, str | None], ChainBuilder] = {
    ("text-generation", None): llm.build_text_generation,
    ("text-generation", "rowwise"): llm.build_rowwise,
    ("image-generation", None): llm.build_image_generation,
    ("image-understanding", None): llm.build_image_understanding,
}


NODE_TYPES: frozenset[str] = frozenset(
    {*NODE_TRANSLATORS, *shared.N8N_MODEL_NODE_TYPES}
)

# The node types the walk turns into ops, in the two roles it needs.
INPUT_TYPES: frozenset[str] = frozenset(
    node_type
    for node_type, translator in NODE_TRANSLATORS.items()
    if translator.is_input
)
WORK_TYPES: frozenset[str] = INPUT_TYPES.symmetric_difference(NODE_TRANSLATORS)
OUTPUT_TYPES: frozenset[str] = frozenset(
    node_type
    for node_type, translator in NODE_TRANSLATORS.items()
    if translator.can_be_output
)
