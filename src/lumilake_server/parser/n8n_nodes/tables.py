"""The n8n node-type and chain-notes tables, keyed by the value the
walk dispatches on.
"""

from . import code, llm, references, retrieval
from .constants import (
    N8N_AGENT_NODE,
    N8N_CHAIN_TYPE,
    N8N_CHAT_TRIGGER,
    N8N_CODE_NODE,
    N8N_MODEL_NODE_TYPES,
    N8N_POSTGRES_NODE,
    N8N_S3_NODE,
)
from .entries import ChainBuilder, NodeDict, NodeTranslator, OpContext


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
    N8N_CHAIN_TYPE: NodeTranslator(
        make=make_chain_node,
        dependencies=llm.chain_dependencies,
        can_be_output=True,
    ),
    N8N_CHAT_TRIGGER: NodeTranslator(
        make=code.make_trigger,
        dependencies=code.trigger_dependencies,
        is_input=True,
        reference=references.NODE_REFERENCES[N8N_CHAT_TRIGGER],
    ),
    N8N_CODE_NODE: NodeTranslator(
        make=code.make_code,
        dependencies=code.code_dependencies,
        can_be_output=True,
    ),
    N8N_POSTGRES_NODE: NodeTranslator(
        make=retrieval.make_postgres,
        dependencies=retrieval.postgres_dependencies,
        reference=references.NODE_REFERENCES[N8N_POSTGRES_NODE],
    ),
    N8N_S3_NODE: NodeTranslator(
        make=retrieval.make_s3,
        dependencies=retrieval.s3_dependencies,
        reference=references.NODE_REFERENCES[N8N_S3_NODE],
    ),
    N8N_AGENT_NODE: NodeTranslator(
        make=llm.make_agent,
        dependencies=llm.agent_dependencies,
        reference=references.NODE_REFERENCES[N8N_AGENT_NODE],
    ),
}

CHAIN_BUILDERS: dict[tuple[str, str | None], ChainBuilder] = {
    ("text-generation", None): llm.build_text_generation,
    ("text-generation", "rowwise"): llm.build_rowwise,
    ("text-generation", "aggregate"): llm.build_aggregate,
    ("image-generation", None): llm.build_image_generation,
    ("image-understanding", None): llm.build_image_understanding,
}


NODE_TYPES: frozenset[str] = frozenset({*NODE_TRANSLATORS, *N8N_MODEL_NODE_TYPES})

# The node types the walk turns into ops, in the two roles it needs.
INPUT_TYPES: frozenset[str] = frozenset(
    node_type
    for node_type, translator in NODE_TRANSLATORS.items()
    if translator.is_input
)
WORK_TYPES: frozenset[str] = frozenset(NODE_TRANSLATORS) - INPUT_TYPES
# Ordered, because the walk emits outputs in this order.
OUTPUT_TYPES: tuple[str, ...] = tuple(
    node_type
    for node_type, translator in NODE_TRANSLATORS.items()
    if translator.can_be_output
)
