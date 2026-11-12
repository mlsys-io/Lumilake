"""FlowMesh task types that only some workers' executors run.

A worker profile lists the task types its executors advertise under
``supported_task_types``. A node whose task type is gated here is placed only on
a CPU worker that advertises it; every other node is placed by engine alone.
python runs only on Docker-backed workers, so it is the one gated type.
"""

import math
from collections.abc import Iterable, Mapping
from typing import Any

from lumilake_server.runtime import python_step

CAPABILITY_GATED_TASK_TYPES = frozenset({python_step.TASK_TYPE})

CPU_ONLY_TASK_TYPES = frozenset({"data_retrieval", "api", "python"})
"""Task types that run on a CPU worker only (HALO's CPU-only engines)."""


def advertised_task_types(profile: Mapping[str, Any]) -> frozenset[str]:
    """The task types a worker profile says its executors run."""
    return frozenset(
        str(task_type) for task_type in profile.get("supported_task_types", ())
    )


def gpu_memory_share(profile: Mapping[str, Any]) -> float | None:
    """The fraction of its card's memory a GPU worker owns, or ``None``.

    A normalized profile records the fraction under ``gpu_memory_share`` and a
    raw FlowMesh profile under the first entry of ``gpu.devices[]``; both are
    read here. A worker owns one share across its devices, and ``None`` means it
    holds a whole device, so an engine of any size fits.
    """
    if "gpu_memory_share" in profile:
        recorded = profile["gpu_memory_share"]
        if isinstance(recorded, int | float) and not isinstance(recorded, bool):
            return float(recorded)
        return None
    gpu_section = profile.get("gpu")
    devices = gpu_section.get("devices") if isinstance(gpu_section, Mapping) else None
    if not isinstance(devices, list) or not devices:
        return None
    device = devices[0]
    if not isinstance(device, Mapping):
        return None
    share = device.get("memory_share")
    if isinstance(share, int | float) and not isinstance(share, bool):
        return float(share)
    return None


def required_task_types(task_types: Iterable[str]) -> frozenset[str]:
    """The gated task types among ``task_types``."""
    return CAPABILITY_GATED_TASK_TYPES.intersection(task_types)


def model_spec_gpu_memory_utilization(model_spec: Any) -> float | None:
    """The fraction of a card's memory a model spec reserves for its vLLM engine.

    The value sits under ``vllm``; a spec that names none returns ``None``, and
    a node carrying such a spec can run on any GPU worker.
    """
    if not isinstance(model_spec, Mapping):
        return None
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


def share_holds_utilization(share: float | None, utilization: float) -> bool:
    """True when a worker owning ``share`` of a card can reserve ``utilization``.

    ``None`` is a whole device, which takes any utilization; a shared worker
    takes one no larger than its share.
    """
    return share is None or utilization <= share


def holds_gpu_utilizations(
    profile: Mapping[str, Any], utilizations: Iterable[float]
) -> bool:
    """True when a worker profile can host an engine reserving each utilization.

    A profile that records no share owns a whole device and holds every
    utilization, as does a request naming none.
    """
    share = gpu_memory_share(profile)
    return all(
        share_holds_utilization(share, utilization) for utilization in utilizations
    )


def requires_cpu_worker(task_types: Iterable[str]) -> bool:
    """True when any of ``task_types`` runs on a CPU worker only.

    A batch with such a node claims a CPU worker even when the configured CPU
    group size is zero.
    """
    return any(
        (task_type or "").strip().lower() in CPU_ONLY_TASK_TYPES
        for task_type in task_types
    )
