"""Worker / hardware schemas inherited from the FlowMesh SDK.

FlowMesh owns the worker wire format; re-exporting the SDK types keeps
the lumilake server automatically aligned with whatever the worker
serializes — no parallel hierarchy to drift.
"""

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
]
