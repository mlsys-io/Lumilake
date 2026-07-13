"""``get_workers`` must never hand a stale (dead-heartbeat) worker id to the
scheduler as a selection candidate, even though the upstream FlowMesh
``workers.list(status="IDLE")`` call can still return stale entries."""

from typing import Any

import pytest
from flowmesh.models.workers import WorkerInfo

from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


def _worker_info(worker_id: str, *, stale: bool) -> WorkerInfo:
    return WorkerInfo(
        id=worker_id,
        namespace="ns",
        cluster="cluster-0",
        node_id="node-0",
        node_alias="node-0",
        status="IDLE",
        stale=stale,
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
