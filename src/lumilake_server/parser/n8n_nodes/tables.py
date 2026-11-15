"""The two dispatch tables: n8n node types and chain notes ``(op-type,
mode)`` pairs.

Every key is the value the walk already dispatches on, so adding a node
type or a chain op-type is a table entry, not an edit to the walk.
"""

from . import code, llm, retrieval, shared
from .entries import ChainBuilder, NodeDict, NodeTranslator, OpContext


def make_chain_node(
    node: NodeDict, ctx: OpContext, node_name: str
) -> "tuple[list[NodeDict], str]":
    """Dispatch a chain node to the builder its notes select."""
    plan = llm.chain_plan(node, ctx, node_name)
    key = (plan.notes["op-type"], plan.notes.get("mode"))
    return CHAIN_BUILDERS.get(key, CHAIN_BUILDERS[(plan.notes["op-type"], None)])(plan)


NODE_TRANSLATORS: dict[str, NodeTranslator] = {
    shared.N8N_CHAIN_TYPE: NodeTranslator(
        make=make_chain_node, dependencies=llm.chain_dependencies
    ),
    shared.N8N_CHAT_TRIGGER: NodeTranslator(
        make=code.make_trigger, dependencies=code.trigger_dependencies
    ),
    shared.N8N_CODE_NODE: NodeTranslator(
        make=code.make_code, dependencies=code.code_dependencies
    ),
    shared.N8N_POSTGRES_NODE: NodeTranslator(
        make=retrieval.make_postgres,
        dependencies=retrieval.postgres_dependencies,
        is_data_source=True,
    ),
    shared.N8N_S3_NODE: NodeTranslator(
        make=retrieval.make_s3,
        dependencies=retrieval.s3_dependencies,
        is_data_source=True,
    ),
    shared.N8N_AGENT_NODE: NodeTranslator(
        make=llm.make_agent, dependencies=llm.agent_dependencies
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
