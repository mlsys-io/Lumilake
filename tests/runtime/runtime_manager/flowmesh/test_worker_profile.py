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


def _worker_payload(
    worker_id: str,
    task_types: frozenset[TaskType],
    hardware: WorkerHardware | None = None,
) -> dict[str, Any]:
    return _worker(worker_id, task_types, hardware).model_dump(mode="json")


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    raw_worker: dict[str, Any],
) -> list[tuple[str, str]]:
    requested: list[tuple[str, str]] = []

    async def fake_get(path: str) -> dict[str, Any]:
        requested.append(("GET", path))
        return raw_worker

    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.flowmesh_get_for_server",
        fake_get,
    )
    return requested


def _shared_gpu_hardware(devices: int) -> WorkerHardware:
    return WorkerHardware(
        gpu=GpuPlatformInfo(
            devices=[
                GpuInfo(index=index, memory_total_bytes=40 << 30)
                for index in range(devices)
            ]
        )
    )


def _raw_with_shares(
    worker: dict[str, Any], shares: list[float | None]
) -> dict[str, Any]:
    worker = dict(worker)
    hardware = dict(worker["hardware"])
    gpu = dict(hardware["gpu"])
    gpu["devices"] = [
        {**device, "memory_share": share}
        for device, share in zip(gpu["devices"], shares, strict=True)
    ]
    hardware["gpu"] = gpu
    worker["hardware"] = hardware
    return worker


@pytest.mark.asyncio
async def test_profile_lists_the_advertised_task_types_sorted(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _serve(monkeypatch, _worker_payload("w", frozenset({TaskType.PYTHON})))

    profile = await flowmesh_manager.get_worker_profile("w")

    assert profile["supported_task_types"] == ["python"]


@pytest.mark.asyncio
async def test_profile_keeps_the_hardware_sections(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hardware = WorkerHardware(memory=MemoryInfo(total_bytes=1024))
    _serve(monkeypatch, _worker_payload("w", frozenset(), hardware))

    profile: dict[str, Any] = await flowmesh_manager.get_worker_profile("w")

    assert profile["memory"]["total_bytes"] == 1024
    assert profile["supported_task_types"] == []


@pytest.mark.asyncio
async def test_profile_is_fetched_with_one_request(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = _serve(
        monkeypatch,
        _raw_with_shares(
            _worker_payload("w", frozenset(), _shared_gpu_hardware(1)), [0.1]
        ),
    )

    await flowmesh_manager.get_worker_profile("w")

    assert requested == [("GET", "/workers/w")]


@pytest.mark.asyncio
async def test_profile_carries_each_devices_memory_share(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _serve(
        monkeypatch,
        _raw_with_shares(
            _worker_payload("w", frozenset(), _shared_gpu_hardware(2)), [0.3, 0.1]
        ),
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
    _serve(
        monkeypatch,
        _raw_with_shares(
            _worker_payload("w", frozenset(), _shared_gpu_hardware(1)), [None]
        ),
    )

    profile: dict[str, Any] = await flowmesh_manager.get_worker_profile("w")

    assert profile["gpu"]["devices"][0]["memory_share"] is None


@pytest.mark.asyncio
async def test_a_cpu_only_worker_profile_has_no_devices(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hardware = WorkerHardware(memory=MemoryInfo(total_bytes=1024))
    _serve(monkeypatch, _worker_payload("w", frozenset(), hardware))

    profile: dict[str, Any] = await flowmesh_manager.get_worker_profile("w")

    assert profile["gpu"] is None
