"""Worker / hardware schemas — re-exported from the FlowMesh SDK."""

from collections.abc import Container
from typing import Self

from flowmesh.models.workers import (
    CPUInfo,
    GpuInfo,
    GpuPlatformInfo,
    HostInfo,
    MemoryInfo,
    NetworkInfo,
    StorageInfo,
    Worker,
    WorkerHardware,
    WorkerInfo,
)
from pydantic import Field


class WorkerStatus(WorkerInfo):
    """A FlowMesh worker plus Lumilake's own busy state.

    ``busy`` reflects whether the server currently has this worker claimed by a
    dispatched batch, so an operator can see cluster utilisation.
    """

    busy: bool = Field(
        default=False,
        description=(
            "Whether the server currently has this worker claimed by a "
            "dispatched batch."
        ),
    )

    @classmethod
    def from_worker(cls, worker: WorkerInfo, busy_workers: Container[str]) -> Self:
        """The worker with Lumilake's busy state for it, its fields taken typed."""
        return cls.model_validate(worker, from_attributes=True).model_copy(
            update={"busy": worker.id in busy_workers}
        )


__all__ = [
    "CPUInfo",
    "GpuInfo",
    "GpuPlatformInfo",
    "HostInfo",
    "MemoryInfo",
    "NetworkInfo",
    "StorageInfo",
    "Worker",
    "WorkerHardware",
    "WorkerInfo",
    "WorkerStatus",
]
