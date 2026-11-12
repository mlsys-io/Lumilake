"""``get_worker_profile`` carries the task types a worker's executors advertise,
so the scheduler can place a node only on a worker that can run it, and each GPU
device's memory share, so it can tell a whole card from a fraction of one."""

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

from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


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


class _FakeWorkersResource:
    def __init__(self, worker: WorkerInfo) -> None:
        self._worker = worker

    async def retrieve(self, worker_id: str) -> WorkerInfo:
        return self._worker


class _FakeFlowMesh:
    def __init__(self, worker: WorkerInfo) -> None:
        self.workers = _FakeWorkersResource(worker)


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    worker: WorkerInfo,
    raw_worker: dict[str, Any] | None = None,
) -> list[str]:
    requested: list[str] = []

    async def fake_get(path: str) -> dict[str, Any]:
        requested.append(path)
        return raw_worker or {}

    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.flowmesh_for_server",
        lambda: _FakeFlowMesh(worker),
    )
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.flowmesh_get_for_server",
        fake_get,
    )
    return requested


def _shared_gpu_worker(shares: list[float | None]) -> WorkerInfo:
    hardware = WorkerHardware(
        gpu=GpuPlatformInfo(
            devices=[
                GpuInfo(index=index, memory_total_bytes=40 << 30)
                for index in range(len(shares))
            ]
        )
    )
    return _worker("w", frozenset(), hardware)


def _raw_with_shares(shares: list[float | None]) -> dict[str, Any]:
    return {
        "id": "w",
        "hardware": {
            "gpu": {
                "devices": [
                    {"index": index, "memory_share": share}
                    for index, share in enumerate(shares)
                ]
            }
        },
    }


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
    hardware = WorkerHardware(memory=MemoryInfo(total_bytes=1024))
    _serve(monkeypatch, _worker("w", frozenset(), hardware))

    profile: dict[str, Any] = await flowmesh_manager.get_worker_profile("w")

    assert profile["memory"]["total_bytes"] == 1024
    assert profile["supported_task_types"] == []


@pytest.mark.asyncio
async def test_profile_carries_each_devices_memory_share(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _serve(
        monkeypatch,
        _shared_gpu_worker([0.3, 0.1]),
        _raw_with_shares([0.3, 0.1]),
    )

    profile: dict[str, Any] = await flowmesh_manager.get_worker_profile("w")

    assert [device["memory_share"] for device in profile["gpu"]["devices"]] == [
        0.3,
        0.1,
    ]


@pytest.mark.asyncio
async def test_a_whole_device_reads_as_no_share(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _serve(monkeypatch, _shared_gpu_worker([None]), _raw_with_shares([None]))

    profile: dict[str, Any] = await flowmesh_manager.get_worker_profile("w")

    assert profile["gpu"]["devices"][0]["memory_share"] is None


@pytest.mark.asyncio
async def test_a_cpu_only_worker_makes_no_extra_request(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hardware = WorkerHardware(memory=MemoryInfo(total_bytes=1024))
    requested = _serve(monkeypatch, _worker("w", frozenset(), hardware))

    await flowmesh_manager.get_worker_profile("w")

    assert requested == []
