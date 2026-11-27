"""The placement facts an optimizer reads about a worker.

A worker profile lists the task types its executors advertise under
``supported_task_types``, the cards it reported and the fraction of its first
card it owns, and the facts the hardware and session-limit checks read. The
schedule protocol carries one per worker, so both the server and a remote
optimizer share this model.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, field_serializer, model_serializer


class GpuDevice(BaseModel):
    """One card a worker reports: its name and total memory, either may be unknown."""

    model_config = ConfigDict(frozen=True)

    name: str | None = None
    memory_total_bytes: int | None = None


class WorkerGpu(BaseModel):
    """The cards a worker reported."""

    model_config = ConfigDict(frozen=True)

    devices: tuple[GpuDevice, ...] = ()


class WorkerProfile(BaseModel):
    """The facts placement reads about a worker.

    ``has_gpu`` is True when the worker reported at least one card, and is
    derived from ``gpu``. ``gpu_memory_fraction`` is the fraction of its first
    card the worker owns, ``None`` for a whole device. The session limits cap the
    python or SSH session FlowMesh gives a step on this worker.
    """

    model_config = ConfigDict(frozen=True)

    supported_task_types: frozenset[str] = frozenset()
    gpu: WorkerGpu | None = None
    gpu_memory_fraction: float | None = None
    cpu_cores: int | None = None
    memory_total_bytes: int | None = None
    max_session_cpu_cores: float | None = None
    max_session_memory_bytes: int | None = None

    @property
    def has_gpu(self) -> bool:
        """True when the worker reported at least one card."""
        return self.gpu is not None and bool(self.gpu.devices)

    @property
    def gpu_devices(self) -> tuple[GpuDevice, ...]:
        """The cards the worker reported, empty for a CPU worker."""
        return () if self.gpu is None else self.gpu.devices

    @field_serializer("supported_task_types")
    def _sorted_task_types(self, task_types: frozenset[str]) -> list[str]:
        """Write the task types sorted, so the wire form is stable."""
        return sorted(task_types)

    @model_serializer(mode="wrap")
    def _with_has_gpu(self, handler: Any) -> dict[str, Any]:
        """Add the derived ``has_gpu``, which a dict-reading client expects."""
        dumped: dict[str, Any] = handler(self)
        dumped["has_gpu"] = self.has_gpu
        return dumped
