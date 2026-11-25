"""Worker capability and capacity facts that placement reads.

A worker profile lists the task types its executors advertise under
``supported_task_types`` and the GPU memory fraction it owns. A node names the
fraction of a card its engine reserves. An optimizer compares the two.
"""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from lumilake_server.data_specs import (
    PYTHON_TASK_TYPE,
    DataSpecModel,
    GpuRequest,
    PythonData,
)
from lumilake_server.utils.utils import parse_memory_to_bytes

CAPABILITY_GATED_TASK_TYPES = frozenset({PYTHON_TASK_TYPE})

CPU_ONLY_TASK_TYPES = frozenset({"data_retrieval", "api", "python"})
"""Task types that run on a CPU worker, unless the node asks for GPUs."""

GPU_TYPE_WILDCARDS = frozenset({"any", "auto", "*"})
"""``GpuRequest.type`` values FlowMesh treats as no type constraint."""


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


def python_step_gpu_request(data_spec: DataSpecModel) -> GpuRequest | None:
    """The GPU request a python step's own data spec carries, if any."""
    if isinstance(data_spec, PythonData):
        return data_spec.gpu
    return None


def python_step_requests_gpu(data_spec: DataSpecModel) -> bool:
    """True when a python step's own data spec asks FlowMesh for GPUs."""
    return python_step_gpu_request(data_spec) is not None


@dataclass(frozen=True, slots=True)
class GpuDevice:
    """One card a worker reports: its name and total memory, either may be unknown."""

    name: str | None = None
    memory_total_bytes: int | None = None


def gpu_devices(profile: Mapping[str, Any]) -> tuple[GpuDevice, ...] | None:
    """The cards a raw worker profile lists, ``None`` when it reports no detail."""
    devices = (profile.get("gpu") or {}).get("devices")
    if not isinstance(devices, list):
        return None
    listed: list[GpuDevice] = []
    for device in devices:
        if not isinstance(device, dict):
            continue
        name = device.get("name")
        vram = device.get("memory_total_bytes")
        listed.append(
            GpuDevice(
                name=name if isinstance(name, str) else None,
                memory_total_bytes=vram if isinstance(vram, int) else None,
            )
        )
    return tuple(listed)


def gpu_request_fits(
    devices: tuple[GpuDevice, ...] | None, request: GpuRequest | None
) -> bool:
    """True when ``devices`` hold ``request``: count, type substring and memory.

    ``None`` devices mean the worker reports no per-device detail, so the check
    degrades to "no constraint" rather than a stall.
    """
    if request is None:
        return True
    if devices is None:
        return True
    matches = 0
    for device in devices:
        if not device_matches_request(device, request):
            continue
        matches += 1
    return matches >= request.count


def profile_fits_gpu_request(
    profile: Mapping[str, Any], request: GpuRequest | None
) -> bool:
    """True when a raw GPU worker profile carries ``request``.

    Callers pass GPU workers only; the raw profile carries no ``has_gpu`` flag.
    """
    if request is None:
        return True
    return gpu_request_fits(gpu_devices(profile), request)


def device_matches_request(device: GpuDevice, request: GpuRequest) -> bool:
    """True when one card meets the request's type substring and memory floor."""
    wanted = request.type.strip().lower() if request.type else ""
    if (
        wanted
        and wanted not in GPU_TYPE_WILDCARDS
        and device.name is not None
        and wanted not in device.name.lower()
    ):
        return False
    required_bytes = parse_memory_to_bytes(request.memory) if request.memory else None
    return (
        required_bytes is None
        or device.memory_total_bytes is None
        or device.memory_total_bytes >= required_bytes
    )


def gated_nodes_gpu_requests(
    nodes: Iterable[tuple[str, DataSpecModel]],
) -> tuple[GpuRequest, ...]:
    """The distinct GPU requests among the python steps of ``nodes``, in order."""
    requests: dict[str, GpuRequest] = {}
    for task_type, data_spec in nodes:
        if task_type not in CAPABILITY_GATED_TASK_TYPES:
            continue
        request = python_step_gpu_request(data_spec)
        if request is not None:
            requests.setdefault(request.model_dump_json(), request)
    return tuple(requests.values())


GpuRequirement = tuple[frozenset[str], tuple[GpuRequest, ...]]
"""One GPU-group requirement: task types to advertise and requests to carry."""


def gated_nodes_gpu_requirements(
    task_types: frozenset[str], requests: Iterable[GpuRequest]
) -> tuple[GpuRequirement, ...]:
    """The requirements a GPU group must hold, one per task type and per request."""
    requirements: list[GpuRequirement] = [
        (frozenset({task_type}), ()) for task_type in sorted(task_types)
    ]
    requirements.extend((frozenset(), (request,)) for request in requests)
    return tuple(requirements)


def gpu_worker_can_run(
    profile: Mapping[str, Any],
    task_types: frozenset[str] = frozenset(),
    requests: tuple[GpuRequest, ...] = (),
) -> bool:
    """True when a GPU worker advertises ``task_types`` and carries ``requests``."""
    return task_types <= advertised_task_types(profile) and all(
        profile_fits_gpu_request(profile, request) for request in requests
    )


def gpu_group_holds(
    profiles: Mapping[str, Mapping[str, Any]],
    workers: Iterable[str],
    requirements: Iterable[GpuRequirement],
) -> bool:
    """True when every requirement has some worker of ``workers`` that can run it."""
    return all(
        any(
            gpu_worker_can_run(profiles.get(worker, {}), task_types, requests)
            for worker in workers
        )
        for task_types, requests in requirements
    )


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
