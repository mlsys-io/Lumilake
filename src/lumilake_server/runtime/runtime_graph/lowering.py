from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from lumilake_server.ops import InputOp, Op
from lumilake_server.runtime.runtime_graph.outputs import OpOutput
from lumilake_server.runtime.runtime_ops import RuntimeOp


class Phase(IntEnum):
    """One lowering pass; iterating the phases in this order emits runtime
    nodes in the same order as the per-type passes it replaces."""

    RETRIEVAL = 0
    LIST_LAMBDA = 1
    LLM = 2
    ROW_LAMBDA = 3


@dataclass
class LoweringResult:
    """What one op lowers to: its runtime nodes in emission order, its
    DSL-to-runtime ids, the nodes an OutputOp archives from, and the upstream
    ops the lowering absorbed."""

    nodes: list[RuntimeOp] = field(default_factory=list)
    dsl_to_runtime: list[str] = field(default_factory=list)
    output_nodes: list[str] = field(default_factory=list)
    visited: list[str] = field(default_factory=list)


class LoweringContext:
    """The build state a lowering reads besides its own op."""

    def __init__(
        self,
        builder: Any,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        input_ops_by_id: dict[str, InputOp],
        task_type_override: str | None,
        nodes: dict[str, RuntimeOp],
        node_order: list[str],
        dsl_to_runtime: dict[str, list[str]],
        visited: set[str],
        outputs_by_source: dict[str, list[tuple[str, str | None]]],
        output_node_map: dict[str, str],
        output_paths: dict[str, str],
    ) -> None:
        self.builder = builder
        self.graph_dict = graph_dict
        self.inputs_dict = inputs_dict
        self.input_ops_by_id = input_ops_by_id
        self.task_type_override = task_type_override
        self.nodes = nodes
        self.node_order = node_order
        self.dsl_to_runtime = dsl_to_runtime
        self.visited = visited
        self.outputs_by_source = outputs_by_source
        self.output_node_map = output_node_map
        self.output_paths = output_paths

    def op_output(self, op: Op) -> OpOutput:
        return self.builder._op_output(op)

    @property
    def profiling(self) -> bool:
        return self.task_type_override == "data_profile"


LoweringFn = Callable[[Any, str, Any, LoweringContext], LoweringResult | None]

_LOWERINGS: dict[type[Op], tuple[Phase, LoweringFn]] = {}


def register(op_cls: type[Op], phase: Phase) -> Callable[[LoweringFn], LoweringFn]:
    """Register ``op_cls``'s lowering; a subclass inherits it through the MRO
    unless it registers its own."""

    def _register(fn: LoweringFn) -> LoweringFn:
        _LOWERINGS[op_cls] = (phase, fn)
        return fn

    return _register


def lookup(op: Op) -> tuple[Phase, LoweringFn] | None:
    for cls in type(op).__mro__:
        found = _LOWERINGS.get(cls)
        if found is not None:
            return found
    return None
