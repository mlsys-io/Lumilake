from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class WorkerStatus(StrEnum):
    UNKNOWN = "UNKNOWN"
    STARTING = "STARTING"
    IDLE = "IDLE"
    BUSY = "BUSY"


class CPUInfo(BaseModel):
    logical_cores: int = Field(description="Number of logical CPU cores.")
    model: str = Field(description="CPU model name.")


class MemoryInfo(BaseModel):
    total_bytes: int | None = Field(description="Total memory in bytes.")


class GpuInfo(BaseModel):
    index: int = Field(description="GPU index.")
    name: str = Field(description="GPU name.")
    uuid: str = Field(description="GPU UUID.")
    memory_total_bytes: int | None = Field(description="Total GPU memory in bytes.")


class GpuPlatformInfo(BaseModel):
    # FlowMesh worker payload field names. ``devices`` is intentional —
    # mirrors what the worker actually serializes; ``gpus`` would be a
    # stale alias that nothing produces.
    driver_version: str | None = Field(default=None, description="GPU driver version.")
    cuda_version: str | None = Field(default=None, description="CUDA version.")
    devices: list[GpuInfo] = Field(default_factory=list, description="List of GPUs.")
    memory_is_unified: bool = Field(
        default=False, description="Whether host and device share memory."
    )
    shared_memory_total_bytes: int | None = Field(
        default=None, description="Shared host-device memory in bytes (unified only)."
    )


class NetworkInfo(BaseModel):
    ip: str | None = Field(description="Network IP address.")
    bandwidth_bytes_per_sec: float | None = Field(
        description="Network bandwidth in bytes per second."
    )


class WorkerHardware(BaseModel):
    cpu: CPUInfo = Field(description="CPU information.")
    memory: MemoryInfo = Field(description="Memory information.")
    gpu: GpuPlatformInfo = Field(description="GPU information.")
    network: NetworkInfo = Field(description="Network information.")


class Worker(BaseModel):
    id: str = Field(description="Worker identifier.")
    alias: str | None = Field(
        default=None, description="Optional human-readable alias."
    )
    namespace: str = Field(description="Worker namespace.")
    cluster: str = Field(description="Worker cluster.")
    node_id: str = Field(description="Owning node identifier.")
    node_alias: str = Field(description="Owning node alias.")
    status: WorkerStatus = Field(
        default=WorkerStatus.UNKNOWN,
        description=(
            "Worker status: `UNKNOWN` (not yet reported), `STARTING` (booting), "
            "`IDLE` (ready), `BUSY` (running workloads)."
        ),
    )
    started_at: str | None = Field(default=None, description="Start timestamp.")
    pid: int | None = Field(default=None, description="Worker process ID.")
    env: dict[str, Any] = Field(default_factory=dict, description="Runtime metadata.")
    hardware: WorkerHardware | None = Field(
        default=None, description="Hardware metadata."
    )
    tags: list[str] = Field(default_factory=list, description="Worker tags.")
    last_seen: str | None = Field(default=None, description="Last heartbeat timestamp.")
    cached_models: list[str] = Field(
        default_factory=list, description="Cached model identifiers."
    )
    cached_datasets: list[str] = Field(
        default_factory=list, description="Cached dataset identifiers."
    )
    cache_updated_ts: str | None = Field(
        default=None, description="Cache metadata update timestamp."
    )
    cost_per_hour: float | None = Field(
        default=None, description="Estimated hourly cost."
    )


class WorkerInfo(Worker):
    stale: bool = Field(description="Whether the worker heartbeat is stale.")
    elastic_disabled: bool = Field(
        default=False,
        description="Whether elastic scaling is disabled for this worker.",
    )
