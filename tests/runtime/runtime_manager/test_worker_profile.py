from typing import Any

import pytest
from flowmesh.models.common import TaskType
from flowmesh.models.workers import MemoryInfo, WorkerCapabilities

from lumilake_server.runtime import _worker_fraction as w
from lumilake_server.runtime.runtime_manager import FlowmeshRuntimeManager


def _worker(
    worker_id: str,
    task_types: frozenset[TaskType],
    hardware: w.WorkerHardware | None = None,
) -> w.WorkerInfo:
    return w.WorkerInfo(
        id=worker_id,
        namespace="ns",
        cluster="cluster-0",
        node_id="node-0",
        node_alias="node-0",
        status="IDLE",
        hardware=hardware,
        capabilities=WorkerCapabilities(supported_task_types=task_types),
    )


def _serve(monkeypatch: pytest.MonkeyPatch, worker: w.WorkerInfo) -> dict[str, Any]:
    payload = worker.model_dump(mode="json")

    async def fake_get(path: str) -> dict[str, Any]:
        return payload

    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh_get_for_server",
        fake_get,
    )
    return payload


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
    gpu = w.GpuPlatformInfo(
        devices=[w.GpuInfo(index=0, memory_fraction=0.3), w.GpuInfo(index=1)]
    )
    hardware = w.WorkerHardware(memory=MemoryInfo(total_bytes=1024), gpu=gpu)
    _serve(monkeypatch, _worker("w", frozenset(), hardware))

    profile: dict[str, Any] = await flowmesh_manager.get_worker_profile("w")

    assert profile["gpu_memory_fraction"] == 0.3
