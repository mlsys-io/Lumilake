"""Worker capability and capacity facts that placement reads.

A worker profile lists the task types its executors advertise under
``supported_task_types`` and the GPU memory fraction it owns. A node names the
fraction of a card its engine reserves. An optimizer compares the two.
"""

import math
from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple, Self

from flowmesh.models.workers import GpuInfo, Worker
from pydantic import BaseModel, ConfigDict, model_serializer, model_validator

from lumilake_server.data_specs import (
    PYTHON_TASK_TYPE,
    DataSpecModel,
    GpuRequest,
    PythonData,
)

CAPABILITY_GATED_TASK_TYPES = frozenset({PYTHON_TASK_TYPE})

CPU_ONLY_TASK_TYPES = frozenset({"data_retrieval", "api", "python"})
"""Task types that run on a CPU worker, unless the node asks for GPUs."""


class GpuDevice(BaseModel):
    """One card a worker reports: its name and total memory, either may be unknown."""

    model_config = ConfigDict(frozen=True)

    name: str | None = None
    memory_total_bytes: int | None = None


class WorkerProfile(BaseModel):
    """The facts placement reads about a FlowMesh worker.

    ``has_gpu`` is True when the worker reported at least one card.
    ``gpu_memory_fraction`` is the fraction of its first card the worker owns,
    ``None`` for a whole device. The session limits cap the python or SSH session
    FlowMesh gives a step on this worker. On the schedule wire the cards travel
    under ``gpu.devices``, the key a schedule-protocol server reads.
    """

    model_config = ConfigDict(frozen=True)

    has_gpu: bool
    supported_task_types: frozenset[str] = frozenset()
    gpu_memory_fraction: float | None = None
    cpu_cores: int | None = None
    memory_total_bytes: int | None = None
    gpu_devices: tuple[GpuDevice, ...] = ()
    max_session_cpu_cores: float | None = None
    max_session_memory_bytes: int | None = None

    @classmethod
    def from_worker(cls, worker: Worker) -> Self:
        """The placement facts of a FlowMesh worker."""
        hardware = worker.hardware
        devices: list[GpuInfo] = (
            hardware.gpu.devices if hardware is not None and hardware.gpu else []
        )
        limits = worker.ssh_limits
        return cls(
            has_gpu=bool(devices),
            supported_task_types=frozenset(
                str(task_type) for task_type in worker.capabilities.supported_task_types
            ),
            gpu_memory_fraction=devices[0].memory_fraction if devices else None,
            cpu_cores=(
                hardware.cpu.logical_cores
                if hardware is not None and hardware.cpu
                else None
            ),
            memory_total_bytes=(
                hardware.memory.total_bytes
                if hardware is not None and hardware.memory
                else None
            ),
            gpu_devices=tuple(
                GpuDevice(
                    name=device.name, memory_total_bytes=device.memory_total_bytes
                )
                for device in devices
            ),
            max_session_cpu_cores=limits.max_cpu_cores if limits is not None else None,
            max_session_memory_bytes=(
                limits.max_memory_bytes if limits is not None else None
            ),
        )

    @model_validator(mode="before")
    @classmethod
    def _cards_from_gpu_section(cls, data: Any) -> Any:
        """Read the cards a schedule-protocol client sent under ``gpu.devices``."""
        if isinstance(data, cls) or not isinstance(data, Mapping):
            return data
        gpu = data.get("gpu")
        if not isinstance(gpu, Mapping) or "gpu_devices" in data:
            return data
        return {**data, "gpu_devices": gpu.get("devices") or ()}

    @model_serializer(mode="wrap")
    def _with_gpu_section(self, handler: Any) -> dict[str, Any]:
        """Write the cards under ``gpu.devices`` for a schedule-protocol server."""
        dumped: dict[str, Any] = handler(self)
        dumped.pop("gpu_devices")
        dumped["supported_task_types"] = sorted(self.supported_task_types)
        if self.gpu_devices:
            dumped["gpu"] = {"devices": [d.model_dump() for d in self.gpu_devices]}
        return dumped


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

    @classmethod
    def union(cls, *lanes: Self) -> Self:
        """One lane holding every step of ``lanes``, empty for no lanes."""
        cpu: frozenset[int | None] = frozenset()
        gpu: frozenset[int | None] = frozenset()
        for lane in lanes:
            cpu |= lane.cpu_memory_mb
            gpu |= lane.gpu_memory_mb
        return cls(cpu_memory_mb=cpu, gpu_memory_mb=gpu)


def profile_fits_gpu_request(
    profile: WorkerProfile, request: GpuRequest | None
) -> bool:
    """True when a GPU worker's profile carries ``request``; no card detail passes."""
    if request is None or not profile.gpu_devices:
        return True
    matched = sum(
        request.allows_device(device.name, device.memory_total_bytes)
        for device in profile.gpu_devices
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


def gpu_worker_can_run(profile: WorkerProfile, requests: Iterable[GpuRequest]) -> bool:
    """True when a GPU worker runs python steps and carries each of ``requests``."""
    return PYTHON_TASK_TYPE in profile.supported_task_types and all(
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
    profile: WorkerProfile, utilizations: Iterable[float]
) -> bool:
    """True when a worker profile can host an engine reserving each utilization."""
    return all(
        fraction_holds_utilization(profile.gpu_memory_fraction, utilization)
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
    return f"{value}"


def session_limits_admit(
    profile: WorkerProfile, cpu: int, memory_bytes: int | None
) -> bool:
    """True when a session on this worker can have ``cpu`` cores and ``memory_bytes``.

    A session gets the smaller of the worker's hardware and its ``ssh_limits``,
    the same bound FlowMesh's dispatcher applies to a python or SSH task. A
    dimension the profile does not report passes.
    """
    if (
        profile.cpu_cores is not None
        and profile.max_session_cpu_cores is not None
        and min(profile.cpu_cores, profile.max_session_cpu_cores) < cpu
    ):
        return False
    total = profile.memory_total_bytes
    max_memory = profile.max_session_memory_bytes
    if (
        memory_bytes is not None
        and total is not None
        and total > 0
        and max_memory is not None
        and min(total, max_memory) < memory_bytes
    ):
        return False
    return True
