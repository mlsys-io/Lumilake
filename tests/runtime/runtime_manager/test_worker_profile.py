"""``get_worker_profile`` carries the placement facts of a FlowMesh worker, so the
scheduler can place a node only on a worker that can run it."""

import pytest
from flowmesh.models.common import TaskType
from flowmesh.models.workers import (
    CPUInfo,
    GpuInfo,
    GpuPlatformInfo,
    MemoryInfo,
    SSHLimits,
    WorkerCapabilities,
    WorkerHardware,
    WorkerInfo,
)

from lumilake_hook import GpuDevice, WorkerGpu, WorkerProfile
from lumilake_server.runtime.runtime_manager import FlowmeshRuntimeManager


def _worker(
    worker_id: str,
    task_types: frozenset[TaskType],
    hardware: WorkerHardware | None = None,
    ssh_limits: SSHLimits | None = None,
) -> WorkerInfo:
    return WorkerInfo(
        id=worker_id,
        namespace="ns",
        cluster="cluster-0",
        node_id="node-0",
        node_alias="node-0",
        status="IDLE",
        hardware=hardware,
        capabilities=WorkerCapabilities(supported_task_types=task_types),
        ssh_limits=ssh_limits,
    )


class _FakeWorkersResource:
    def __init__(self, worker: WorkerInfo) -> None:
        self._worker = worker

    async def retrieve(self, worker_id: str) -> WorkerInfo:
        return self._worker


class _FakeFlowMesh:
    def __init__(self, worker: WorkerInfo) -> None:
        self.workers = _FakeWorkersResource(worker)


def _serve(monkeypatch: pytest.MonkeyPatch, worker: WorkerInfo) -> None:
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh_for_server",
        lambda: _FakeFlowMesh(worker),
    )


@pytest.mark.asyncio
async def test_get_worker_profile_returns_the_served_workers_facts(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = GpuInfo(
        index=0, name="H100", memory_fraction=0.3, memory_total_bytes=1 << 30
    )
    hardware = WorkerHardware(
        cpu=CPUInfo(logical_cores=32),
        memory=MemoryInfo(total_bytes=29 * 1024**3),
        gpu=GpuPlatformInfo(devices=[card, card]),
    )
    served = _worker(
        "w",
        frozenset({TaskType.PYTHON}),
        hardware,
        SSHLimits(max_cpu_cores=8.0, max_memory_bytes=12 * 1024**3),
    )
    _serve(monkeypatch, served)

    assert await flowmesh_manager.get_worker_profile("w") == WorkerProfile(
        supported_task_types=frozenset({"python"}),
        gpu=WorkerGpu(
            devices=(GpuDevice(name="H100", memory_total_bytes=1 << 30),) * 2
        ),
        gpu_memory_fraction=0.3,
        cpu_cores=32,
        memory_total_bytes=29 * 1024**3,
        max_session_cpu_cores=8.0,
        max_session_memory_bytes=12 * 1024**3,
    )
