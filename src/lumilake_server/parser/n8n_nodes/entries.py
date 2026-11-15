"""The records the n8n parser dispatches on.

:class:`OpContext` is what the walk knows when it reaches one node;
:class:`NodeTranslator` is the entry one n8n node type maps to;
:class:`ChainPlan` is the chain node prepared for its notes ``(op-type,
mode)`` builder.
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
class NodeTranslator:
    """How one n8n node type is turned into Lumilake ops."""

    make: Callable[..., tuple[list[NodeDict], str]]
    dependencies: Callable[[str, OpContext], list[str]]
    is_data_source: bool = False


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
