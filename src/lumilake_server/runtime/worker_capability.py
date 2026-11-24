"""Worker capability and capacity facts that placement reads.

A worker profile lists the task types its executors advertise under
``supported_task_types`` and the GPU memory fraction it owns. A node names the
fraction of a card its engine reserves. An optimizer compares the two.
"""

import math
from collections.abc import Iterable, Mapping
from typing import Any

from lumilake_server.data_specs import PYTHON_TASK_TYPE, DataSpecModel, PythonData

CAPABILITY_GATED_TASK_TYPES = frozenset({PYTHON_TASK_TYPE})

CPU_ONLY_TASK_TYPES = frozenset({"data_retrieval", "api", "python"})
"""Task types that run on a CPU worker, unless the node asks for GPUs."""


def advertised_task_types(profile: Mapping[str, Any]) -> frozenset[str]:
    """The task types a worker profile says its executors run."""
    return frozenset(
        str(task_type) for task_type in profile.get("supported_task_types", ())
    )


def gpu_memory_fraction(profile: Mapping[str, Any]) -> float | None:
    """The fraction of its card a GPU worker owns, ``None`` for a whole device."""
    fraction = profile.get("gpu_memory_fraction")
    if isinstance(fraction, int | float) and not isinstance(fraction, bool):
        return float(fraction)
    return None


def required_task_types(task_types: Iterable[str]) -> frozenset[str]:
    """The gated task types among ``task_types``."""
    return CAPABILITY_GATED_TASK_TYPES.intersection(task_types)


def python_step_requests_gpu(data_spec: DataSpecModel) -> bool:
    """True when a python step's own data spec asks FlowMesh for GPUs."""
    return isinstance(data_spec, PythonData) and data_spec.gpu is not None


def split_required_task_types(
    nodes: Iterable[tuple[str, DataSpecModel]],
) -> tuple[frozenset[str], frozenset[str]]:
    """The gated task types ``nodes`` need from the CPU group and from the GPU group.

    ``nodes`` yields a ``(task_type, data_spec)`` pair per node. A python step
    that requests GPUs runs on a GPU worker, so its task type is a GPU
    requirement; every other gated step is a CPU requirement.
    """
    cpu_types: set[str] = set()
    gpu_types: set[str] = set()
    for task_type, data_spec in nodes:
        if task_type not in CAPABILITY_GATED_TASK_TYPES:
            continue
        if python_step_requests_gpu(data_spec):
            gpu_types.add(task_type)
        else:
            cpu_types.add(task_type)
    return frozenset(cpu_types), frozenset(gpu_types)


def model_spec_gpu_memory_utilization(model_spec: Mapping[str, Any]) -> float | None:
    """The fraction of a card's memory a model spec reserves for its vLLM engine."""
    vllm = model_spec.get("vllm")
    if not isinstance(vllm, Mapping):
        return None
    utilization = vllm.get("gpu_memory_utilization")
    if isinstance(utilization, bool) or not isinstance(utilization, int | float):
        return None
    value = float(utilization)
    if not math.isfinite(value):
        return None
    return value


def fraction_holds_utilization(fraction: float | None, utilization: float) -> bool:
    """True when a worker owning ``fraction`` of a card can reserve ``utilization``."""
    return fraction is None or utilization <= fraction


def holds_gpu_utilizations(
    profile: Mapping[str, Any], utilizations: Iterable[float]
) -> bool:
    """True when a worker profile can host an engine reserving each utilization."""
    fraction = gpu_memory_fraction(profile)
    return all(
        fraction_holds_utilization(fraction, utilization)
        for utilization in utilizations
    )


def requires_cpu_worker(nodes: Iterable[tuple[str, DataSpecModel]]) -> bool:
    """True when any of ``nodes`` needs a CPU worker.

    ``nodes`` yields a ``(task_type, data_spec)`` pair per node. A batch with
    such a node claims a CPU worker even when the configured CPU group size is
    zero.
    """
    return any(
        (task_type or "").strip().lower() in CPU_ONLY_TASK_TYPES
        and not python_step_requests_gpu(data_spec)
        for task_type, data_spec in nodes
    )
