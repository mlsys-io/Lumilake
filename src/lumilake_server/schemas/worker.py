"""Worker / hardware schemas — re-exported from the FlowMesh SDK."""

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
