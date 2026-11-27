"""The placement facts an optimizer reads about a worker.

A worker profile lists the task types its executors advertise under
``supported_task_types``, the fraction of a card it owns, and the facts the
hardware and session-limit checks read. The schedule protocol carries one per
worker, so both the server and a remote optimizer share this model.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, model_serializer, model_validator


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

    @model_validator(mode="before")
    @classmethod
    def _cards_from_gpu_section(cls, data: Any) -> Any:
        """Read the cards a schedule-protocol client sent under ``gpu.devices``."""
        if isinstance(data, cls) or not isinstance(data, dict):
            return data
        gpu = data.get("gpu")
        if not isinstance(gpu, dict) or "gpu_devices" in data:
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
