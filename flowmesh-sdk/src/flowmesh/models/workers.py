"""Worker-related models."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from .common import TaskType


class CPUInfo(BaseModel):
    logical_cores: int | None = None
    model: str | None = None
    arch: str | None = None
    name: str | None = None
    has_avx: bool | None = None


class MemoryInfo(BaseModel):
    total_bytes: int | None = None


class GpuInfo(BaseModel):
    index: int | None = None
    name: str | None = None
    uuid: str | None = None
    memory_total_bytes: int | None = None


class GpuPlatformInfo(BaseModel):
    driver_version: str | None = None
    cuda_version: str | None = None
    devices: list[GpuInfo] = Field(default_factory=list)
    memory_is_unified: bool = False
    shared_memory_total_bytes: int | None = None
    gpu_arch: str | None = None
    compute_cap: int | None = None
    bw_nvlink: float | None = None
    gpu_lanes: int | None = None
    gpu_mem_bw: float | None = None
    pci_gen: float | None = None
    pcie_bw: float | None = None
    cost_per_hour: float | None = None
    total_flops: float | None = None


class NetworkInfo(BaseModel):
    ip: str | None = None
    bandwidth_bytes_per_sec: float | None = None
    public_ipaddr: str | None = None
    local_ipaddrs: list[str] | None = None
    geolocation: str | None = None
    inet_down: float | None = None
    inet_up: float | None = None


class StorageInfo(BaseModel):
    disk_name: str | None = None
    disk_space: float | None = None
    disk_usage: float | None = None
    disk_util: float | None = None


class HostInfo(BaseModel):
    os_version: str | None = None
    reliability: float | None = None


class WorkerHardware(BaseModel):
    cpu: CPUInfo | None = None
    memory: MemoryInfo | None = None
    gpu: GpuPlatformInfo | None = None
    network: NetworkInfo | None = None
    storage: StorageInfo | None = None
    host: HostInfo | None = None
    extra: dict[str, Any] | None = None


class SSHLimits(BaseModel):
    max_cpu_cores: float | None = None
    max_memory_bytes: int | None = None
    max_pids: int | None = None


class WorkerCapabilities(BaseModel):
    supported_task_types: frozenset[TaskType] = Field(default_factory=frozenset)


class Worker(BaseModel):
    id: str
    alias: str | None = None
    namespace: str
    cluster: str
    node_id: str
    node_alias: str
    version: str | None = None
    status: str
    started_at: str | None = None
    pid: int | None = None
    env: dict[str, Any] = Field(default_factory=dict)
    hardware: WorkerHardware | None = None
    capabilities: WorkerCapabilities = Field(default_factory=WorkerCapabilities)
    ssh_limits: SSHLimits | None = None
    tags: list[str] = Field(default_factory=list)
    last_seen: str | None = None
    cached_models: list[str] = Field(default_factory=list)
    cached_datasets: list[str] = Field(default_factory=list)
    cache_updated_ts: str | None = None
    cost_per_hour: float | None = None

    @field_validator("tags", mode="before")
    @classmethod
    def validate_tags(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [tag.strip() for tag in value.split(",") if tag.strip()]
        return list(value)


class WorkerInfo(Worker):
    stale: bool = False
