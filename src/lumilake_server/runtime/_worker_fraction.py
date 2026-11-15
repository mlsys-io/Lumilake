"""FlowMesh worker models extended with the GPU memory fraction.

The pinned SDK's device model drops ``memory_fraction``, so the scheduler's
worker read validates into these instead. The pin bump deletes this module.
"""

from flowmesh.models import workers as sdk
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
