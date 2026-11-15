"""The records the n8n parser dispatches on: what the walk knows at
one node, what a node type maps to, and a chain prepared for its builder.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

NodeDict = dict[str, Any]
Incoming = dict[str, list[dict[str, str]]]
NodeMap = dict[str, NodeDict]


@dataclass(frozen=True)
class OpContext:
    """Everything the walk knows when it reaches one node."""

    scope: str
    inputs: dict[str, list[str]]
    incoming: Incoming
    node_map: NodeMap
    op_ids: dict[str, str]
    model_specs: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ColumnRef:
    """One column a prompt builder wants from a referenced node."""

    node_name: str
    label: str
    path: str | None
    op_ids: dict[str, str]
    inputs: dict[str, list[str]]


@dataclass(frozen=True)
class NodeReference:
    """How one node type answers a reference from another node."""

    # The items path carrying this node type's rows, with the reference's
    # field path applied. ``None`` falls through to the generic output path.
    items_path: Callable[[str | None], str | None] | None = None
    # How this node type contributes one column to a rowwise or aggregate
    # column list. ``None`` means the generic LLM-style output path.
    column: Callable[[ColumnRef], NodeDict] | None = None
    # How this node type contributes a column to an aggregate table block,
    # where ``ref.label`` is the block's own column name and serves as the
    # field fallback.
    table_column: Callable[[ColumnRef], NodeDict] | None = None
    # The retrieval mode whose rows this node type carries, when it is a
    # retrieval source that depends on a database or bucket.
    retrieval_mode: str | None = None
    # Whether other nodes treat this type as a data source rather than as an
    # upstream whose output feeds them, which skips it in their input lists.
    data_source: bool = False


@dataclass(frozen=True)
class NodeTranslator:
    """How one n8n node type is turned into Lumilake ops."""

    make: Callable[[NodeDict, OpContext, str], tuple[list[NodeDict], str]]
    dependencies: Callable[[str, OpContext], list[str]]
    is_input: bool = False
    can_be_output: bool = False
    reference: NodeReference | None = None


@dataclass(frozen=True)
class ChainPlan:
    """A chain node whose shared fields are already read."""

    node: NodeDict
    node_name: str
    ctx: OpContext
    prompt_text: str
    system_messages: list[str]
    notes: dict[str, Any]
    upstream_main: str | None
    model_config: dict[str, Any]
    vlm_config: dict[str, Any] | None
    return_history: bool


ChainBuilder = Callable[[ChainPlan], tuple[list[NodeDict], str]]
