from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from lumilake_server.ops import InputOp, Op
from lumilake_server.runtime.runtime_ops import RuntimeOp


class Phase(IntEnum):
    """Runtime nodes are emitted phase by phase in this order, which
    ``topological_order`` uses to break ties between ready nodes."""

    RETRIEVAL = 0
    LIST_LAMBDA = 1
    LLM = 2
    ROW_LAMBDA = 3


@dataclass(frozen=True)
class LoweringContext:
    """The build state a lowering reads besides its own op."""

    graph_dict: dict[str, Op]
    inputs_dict: dict[str, list[str]]
    input_ops_by_id: dict[str, InputOp]
    task_type_override: str | None
    outputs_by_source: dict[str, list[tuple[str, str | None]]]
    nodes: dict[str, RuntimeOp]
    dsl_to_runtime: dict[str, list[str]]
    visited: set[str]


@dataclass
class LoweringResult:
    """The runtime nodes one op lowers to, in emission order, and the upstream
    ops the lowering absorbed."""

    nodes: list[RuntimeOp] = field(default_factory=list)
    visited: list[str] = field(default_factory=list)


LoweringFn = Callable[[Any, Any, LoweringContext], LoweringResult | None]
AfterPhaseFn = Callable[[Any, LoweringContext], None]

_LOWERINGS: dict[type[Op], list[tuple[Phase, LoweringFn]]] = {}
_AFTER_PHASE: dict[Phase, Callable[[Any, LoweringContext], None]] = {}


def after_phase(phase: Phase) -> Callable[[AfterPhaseFn], AfterPhaseFn]:
    """Register work that runs once after ``phase``, for a rule over the nodes
    every lowering in that phase has built."""

    def _register(fn: AfterPhaseFn) -> AfterPhaseFn:
        _AFTER_PHASE[phase] = fn
        return fn

    return _register


def phase_hook(phase: Phase) -> Callable[[Any, LoweringContext], None] | None:
    return _AFTER_PHASE.get(phase)


def register(op_cls: type[Op], phase: Phase) -> Callable[[LoweringFn], LoweringFn]:
    """Register ``op_cls``'s lowering in ``phase``.

    The lookup is per phase: a subclass inherits an ancestor's lowering for a
    phase it does not register in itself. A lowering module must be in
    ``RuntimeGraphBuilder``'s bases, because registration happens at import
    time and only those modules are imported.
    """

    def _register(fn: LoweringFn) -> LoweringFn:
        _LOWERINGS.setdefault(op_cls, []).append((phase, fn))
        return fn

    return _register


def lookup(op: Op, phase: Phase) -> LoweringFn | None:
    for cls in type(op).__mro__:
        for entry_phase, fn in _LOWERINGS.get(cls, ()):
            if entry_phase is phase:
                return fn
    return None


def has_lowering(op: Op) -> bool:
    """Whether any lowering is registered for the op's class or an ancestor."""
    return any(cls in _LOWERINGS for cls in type(op).__mro__)
