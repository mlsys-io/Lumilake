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
    DERIVED = 3
    ROW_LAMBDA = 4


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
    # Ops whose lowering has run, so an on-demand read lowers each only once
    # and the phase loop skips it.
    lowered: set[str]
    # Emission order and the output maps, shared by the phase loop and any
    # lowering that runs on demand.
    node_order: list[str]
    output_node_map: dict[str, str]
    output_paths: dict[str, str]


@dataclass
class LoweringResult:
    """The runtime nodes one op lowers to, in emission order, and the mapping
    of any other op whose nodes the lowering also built, such as a Lambda step
    built by recursion."""

    nodes: list[RuntimeOp] = field(default_factory=list)
    built: dict[str, list[str]] = field(default_factory=dict)


LoweringFn = Callable[[Any, Any, LoweringContext], LoweringResult | None]
AfterPhaseFn = Callable[[Any, LoweringContext], None]

_LOWERINGS: dict[type[Op], list[tuple[Phase, LoweringFn]]] = {}
_AFTER_PHASE: dict[Phase, AfterPhaseFn] = {}


def after_phase(phase: Phase) -> Callable[[AfterPhaseFn], AfterPhaseFn]:
    """Register work that runs once after ``phase``, for a rule over the nodes
    every lowering in that phase has built."""

    def _register(fn: AfterPhaseFn) -> AfterPhaseFn:
        if phase in _AFTER_PHASE:
            raise ValueError(f"phase {phase.name} already has an after-phase hook")
        _AFTER_PHASE[phase] = fn
        return fn

    return _register


def phase_hook(phase: Phase) -> AfterPhaseFn | None:
    """The hook registered for ``phase``, if any."""
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


_READ_HOOKS: dict[type[Op], LoweringFn] = {}


def register_read_hook(op_cls: type[Op]) -> Callable[[LoweringFn], LoweringFn]:
    """Register how an op is built when a consumer reads it before its phase.

    A lowering declines an op its own phase has already visited, which is right
    in that phase; a consumer that reads the op early needs it built anyway, so
    it registers the build here and the driver runs it on demand.
    """

    def _register(fn: LoweringFn) -> LoweringFn:
        _READ_HOOKS[op_cls] = fn
        return fn

    return _register


def read_hook(op: Op) -> LoweringFn | None:
    """The on-demand build registered for the op's class or an ancestor."""
    for cls in type(op).__mro__:
        fn = _READ_HOOKS.get(cls)
        if fn is not None:
            return fn
    return None
