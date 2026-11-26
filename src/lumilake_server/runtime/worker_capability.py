"""Worker capability and capacity facts that placement reads.

A worker profile lists the task types its executors advertise under
``supported_task_types`` and the GPU memory fraction it owns. A node names the
fraction of a card its engine reserves. An optimizer compares the two.
"""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

from lumilake_server.data_specs import (
    PYTHON_TASK_TYPE,
    DataSpecModel,
    GpuRequest,
    PythonData,
)

CAPABILITY_GATED_TASK_TYPES = frozenset({PYTHON_TASK_TYPE})

CPU_ONLY_TASK_TYPES = frozenset({"data_retrieval", "api", "python"})
"""Task types that run on a CPU worker, unless the node asks for GPUs."""


class SessionLane(NamedTuple):
    """What each class of a batch's workers must admit as a session.

    ``cpu_memory_mb`` holds the memory every python step in the batch's CPU group
    asks for, in MiB (``None`` for a step on the job's memory). FlowMesh caps a
    worker's python or SSH session at its ``ssh_limits``, so the job's CPU and
    each of those memories must fit under the limits. ``gpu_memory_mb`` holds the
    same for the python steps that ask for GPUs themselves, which run on a GPU
    worker; a GPU worker running other task types is held to its hardware alone,
    as FlowMesh does not cap those sessions.
    """

    cpu_memory_mb: frozenset[int | None] = frozenset()
    gpu_memory_mb: frozenset[int | None] = frozenset()

    def is_empty(self) -> bool:
        """True when no step of the batch runs in a session."""
        return not self.cpu_memory_mb and not self.gpu_memory_mb


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


@dataclass(frozen=True, slots=True)
class _GpuDevice:
    """One card a worker reports: its name and total memory, either may be unknown."""

    name: str | None = None
    memory_total_bytes: int | None = None


def _gpu_devices(profile: Mapping[str, Any]) -> tuple[_GpuDevice, ...] | None:
    """The cards a raw worker profile lists, ``None`` when it reports no detail."""
    devices = (profile.get("gpu") or {}).get("devices")
    if not isinstance(devices, list):
        return None
    listed: list[_GpuDevice] = []
    for device in devices:
        if not isinstance(device, dict):
            continue
        name = device.get("name")
        vram = device.get("memory_total_bytes")
        listed.append(
            _GpuDevice(
                name=name if isinstance(name, str) else None,
                memory_total_bytes=vram if isinstance(vram, int) else None,
            )
        )
    return tuple(listed)


def profile_fits_gpu_request(
    profile: Mapping[str, Any], request: GpuRequest | None
) -> bool:
    """True when a GPU worker's profile carries ``request``; no device detail passes."""
    if request is None:
        return True
    devices = _gpu_devices(profile)
    if devices is None:
        return True
    matched = sum(
        request.allows_device(device.name, device.memory_total_bytes)
        for device in devices
    )
    return matched >= request.count


def python_steps_gpu_requests(
    data_specs: Iterable[DataSpecModel],
) -> tuple[GpuRequest, ...]:
    """The distinct GPU requests the python steps among ``data_specs`` carry."""
    requests: dict[str, GpuRequest] = {}
    for data_spec in data_specs:
        request = data_spec.gpu_request
        if request is not None:
            requests.setdefault(request.model_dump_json(), request)
    return tuple(requests.values())


def gpu_worker_can_run(
    profile: Mapping[str, Any], requests: Iterable[GpuRequest]
) -> bool:
    """True when a GPU worker runs python steps and carries each of ``requests``."""
    return PYTHON_TASK_TYPE in advertised_task_types(profile) and all(
        profile_fits_gpu_request(profile, request) for request in requests
    )


def cpu_required_task_types(
    nodes: Iterable[tuple[str, DataSpecModel]],
) -> frozenset[str]:
    """The gated task types ``nodes`` need from the CPU group: every gated step but a
    python step that requests GPUs, which runs on a GPU worker."""
    return CAPABILITY_GATED_TASK_TYPES.intersection(
        task_type for task_type, data_spec in nodes if data_spec.gpu_request is None
    )


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
        and data_spec.gpu_request is None
        for task_type, data_spec in nodes
    )


def python_step_session_lane(
    nodes: Iterable[tuple[str, DataSpecModel]],
) -> SessionLane:
    """The session lanes the python steps among ``nodes`` put on the batch's groups.

    A python step without a GPU request runs on a CPU worker and lands in
    ``cpu_memory_mb``; one that asks for GPUs runs on a GPU worker and lands in
    ``gpu_memory_mb``.
    """
    cpu_memory_mb: set[int | None] = set()
    gpu_memory_mb: set[int | None] = set()
    for task_type, data_spec in nodes:
        if (task_type or "").strip().lower() != PYTHON_TASK_TYPE:
            continue
        memory_mb = data_spec.memory_mb if isinstance(data_spec, PythonData) else None
        if data_spec.gpu_request is None:
            cpu_memory_mb.add(memory_mb)
        else:
            gpu_memory_mb.add(memory_mb)
    return SessionLane(frozenset(cpu_memory_mb), frozenset(gpu_memory_mb))


def session_memory_bytes(
    session_memory_mb: frozenset[int | None], job_memory_bytes: int | None
) -> int | None:
    """The largest memory the ``session_memory_mb`` steps ask for, in bytes.

    A step on the job's memory (``None``) is counted at ``job_memory_bytes``, so
    this is the memory the session-limits check is made against.
    """
    step_memory = [
        job_memory_bytes if memory_mb is None else memory_mb * 1024**2
        for memory_mb in session_memory_mb
    ]
    return max((b for b in step_memory if b is not None), default=None)


def format_memory_bytes(value: int | None) -> str:
    """``value`` as a Kubernetes-style memory string, e.g. ``2048Mi`` or ``12Gi``.

    ``None`` formats as ``unknown``, for a memory the profile did not report.
    """
    if value is None:
        return "unknown"
    for suffix, size in (
        ("Ti", 1024**4),
        ("Gi", 1024**3),
        ("Mi", 1024**2),
        ("Ki", 1024),
    ):
        if value >= size and value % size == 0:
            return f"{value // size}{suffix}"
    return f"{value}B"


def session_limits_admit(
    profile: Mapping[str, Any], cpu: int, memory_bytes: int | None
) -> bool:
    """True when a session on this worker can have ``cpu`` cores and ``memory_bytes``.

    A session gets the smaller of the worker's hardware and its ``ssh_limits``,
    the same bound FlowMesh's dispatcher applies to a python or SSH task. A
    dimension the profile does not report passes.
    """
    limits = profile.get("ssh_limits")
    if not isinstance(limits, Mapping):
        return True
    cores = (profile.get("cpu") or {}).get("logical_cores")
    max_cores = limits.get("max_cpu_cores")
    if isinstance(cores, int) and isinstance(max_cores, int | float):
        if min(cores, max_cores) < cpu:
            return False
    total = (profile.get("memory") or {}).get("total_bytes")
    max_memory = limits.get("max_memory_bytes")
    if (
        memory_bytes is not None
        and isinstance(total, int)
        and total > 0
        and isinstance(max_memory, int)
        and min(total, max_memory) < memory_bytes
    ):
        return False
    return True
