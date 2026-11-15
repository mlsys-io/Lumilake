"""Worker / hardware schemas: the FlowMesh SDK models, with the GPU memory fraction."""

from flowmesh.models import workers as sdk
from flowmesh.models.workers import (
    CPUInfo,
    HostInfo,
    MemoryInfo,
    NetworkInfo,
    StorageInfo,
    Worker,
)
from pydantic import Field


class GpuInfo(sdk.GpuInfo):
    """A GPU device, with the fraction of its card the worker owns."""

    memory_fraction: float | None = Field(
        default=None,
        description=(
            "Fraction of the card's memory the worker is capped at; ``None`` "
            "for a whole device."
        ),
    )


class GpuPlatformInfo(sdk.GpuPlatformInfo):
    """The GPU platform, with each device's owned memory fraction."""

    devices: list[GpuInfo] = Field(
        default_factory=list, description="GPU devices the worker owns."
    )


class WorkerHardware(sdk.WorkerHardware):
    """The hardware report, with the fraction-aware GPU platform."""

    gpu: GpuPlatformInfo | None = Field(default=None, description="GPU platform.")


class WorkerInfo(sdk.WorkerInfo):
    """A FlowMesh worker, with the fraction-aware hardware report."""

    hardware: WorkerHardware | None = Field(default=None, description="Hardware.")


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
