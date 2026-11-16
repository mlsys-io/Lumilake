"""``get_worker_profile`` carries the task types a worker's executors advertise,
so the scheduler can place a node only on a worker that can run it."""

from typing import Any

import pytest
from flowmesh.models.common import TaskType
from flowmesh.models.workers import (
    GpuInfo,
    GpuPlatformInfo,
    MemoryInfo,
    WorkerCapabilities,
    WorkerHardware,
    WorkerInfo,
)

from lumilake_server.runtime.runtime_manager import FlowmeshRuntimeManager


def _worker(
    worker_id: str,
    task_types: frozenset[TaskType],
    hardware: WorkerHardware | None = None,
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
    )


def _serve(monkeypatch: pytest.MonkeyPatch, worker: WorkerInfo) -> None:
    payload = worker.model_dump(mode="json")

    async def fake_get(path: str) -> dict[str, Any]:
        return payload

    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh_get_for_server",
        fake_get,
    )


@pytest.mark.asyncio
async def test_profile_lists_the_advertised_task_types_sorted(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _serve(monkeypatch, _worker("w", frozenset({TaskType.PYTHON})))

    profile = await flowmesh_manager.get_worker_profile("w")

    assert profile["supported_task_types"] == ["python"]


@pytest.mark.asyncio
async def test_profile_keeps_the_hardware_sections(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu = GpuPlatformInfo(
        devices=[GpuInfo(index=0, memory_fraction=0.3), GpuInfo(index=1)]
    )
    hardware = WorkerHardware(memory=MemoryInfo(total_bytes=1024), gpu=gpu)
    _serve(monkeypatch, _worker("w", frozenset(), hardware))

    profile: dict[str, Any] = await flowmesh_manager.get_worker_profile("w")

    assert profile["memory"]["total_bytes"] == 1024
    assert profile["gpu_memory_fraction"] == 0.3
