"""``get_workers`` must never hand a stale (dead-heartbeat) worker id to the
scheduler as a selection candidate, even though the upstream FlowMesh
``workers.list(status="IDLE")`` call can still return stale entries."""

from typing import Any

import pytest
from flowmesh.models.workers import WorkerInfo

from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


def _worker_info(
    worker_id: str, *, stale: bool, status: str = "IDLE", cordoned: bool = False
) -> WorkerInfo:
    return WorkerInfo(
        id=worker_id,
        namespace="ns",
        cluster="cluster-0",
        node_id="node-0",
        node_alias="node-0",
        status=status,
        stale=stale,
        cordoned=cordoned,
    )


class _FakeWorkersResource:
    def __init__(self, workers: list[WorkerInfo]) -> None:
        self._workers = workers

    async def list(self, **kwargs: Any) -> list[WorkerInfo]:
        return self._workers


class _FakeFlowMesh:
    def __init__(self, workers: list[WorkerInfo]) -> None:
        self.workers = _FakeWorkersResource(workers)


@pytest.mark.asyncio
async def test_get_workers_excludes_stale_worker_from_candidates(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = _worker_info("wkr-live", stale=False)
    stale = _worker_info("wkr-16", stale=True)
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.flowmesh_for_server",
        lambda: _FakeFlowMesh([stale, live]),
    )

    workers = await flowmesh_manager.get_workers()

    assert workers == ["wkr-live"]


@pytest.mark.asyncio
async def test_get_workers_returns_empty_when_all_candidates_are_stale(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_a = _worker_info("wkr-16", stale=True)
    stale_b = _worker_info("wkr-17", stale=True)
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.flowmesh_for_server",
        lambda: _FakeFlowMesh([stale_a, stale_b]),
    )

    workers = await flowmesh_manager.get_workers()

    assert workers == []


@pytest.mark.asyncio
async def test_get_dispatchable_workers_keeps_only_idle_and_busy_uncordoned(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a live, uncordoned worker FlowMesh lists idle or busy can take a
    queued task, since any other state may lift the pin and run it elsewhere."""
    listed = [
        _worker_info("wkr-idle", stale=False, status="IDLE"),
        _worker_info("wkr-busy", stale=False, status="BUSY"),
        _worker_info("wkr-starting", stale=False, status="STARTING"),
        _worker_info("wkr-stopping", stale=False, status="STOPPING"),
        _worker_info("wkr-unavailable", stale=False, status="UNAVAILABLE"),
        _worker_info("wkr-cordoned", stale=False, cordoned=True),
        _worker_info("wkr-idle-stale", stale=True),
    ]
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.flowmesh_for_server",
        lambda: _FakeFlowMesh(listed),
    )

    workers = await flowmesh_manager.get_dispatchable_workers()

    assert workers == ["wkr-idle", "wkr-busy"]


@pytest.mark.asyncio
async def test_get_all_workers_still_lists_non_dispatchable_workers(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full pool used to detect unsatisfiable partitions keeps every
    registered worker, including the ones left out of the re-offer."""
    listed = [
        _worker_info("wkr-cordoned", stale=False, cordoned=True),
        _worker_info("wkr-stopping", stale=False, status="STOPPING"),
    ]
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.flowmesh_for_server",
        lambda: _FakeFlowMesh(listed),
    )

    workers = await flowmesh_manager.get_all_workers()

    assert workers == ["wkr-cordoned", "wkr-stopping"]
