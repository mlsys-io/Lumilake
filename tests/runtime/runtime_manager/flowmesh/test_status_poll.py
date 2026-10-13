"""The status poll uses FlowMesh's lightweight status view when it can.

GET /tasks/{id} returns the whole task, ~0.5-1 MB for an inference task.
Polling that for every in-flight slice preceded the 2026-10-12 home-cluster
failures (FlowMesh#225). SDKs with ``tasks.status()`` fetch only status,
placement, timing and errors; older SDKs keep using ``retrieve()``.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


class _Tasks:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def retrieve(self, task_id: str) -> Any:
        self.calls.append("retrieve")
        return SimpleNamespace(status="DONE", error=None, last_error=None)


class _TasksWithStatus(_Tasks):
    async def status(self, task_id: str) -> Any:
        self.calls.append("status")
        return SimpleNamespace(status="FAILED", error="boom", last_error=None)


def _use(monkeypatch: pytest.MonkeyPatch, tasks: _Tasks) -> None:
    fm = SimpleNamespace(tasks=tasks)
    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: fm))


@pytest.mark.asyncio
async def test_polls_the_status_view_when_the_sdk_has_it(
    flowmesh_manager: FlowmeshRuntimeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = _TasksWithStatus()
    _use(monkeypatch, tasks)
    info = await flowmesh_manager.fetch_task_status("t1")
    assert tasks.calls == ["status"]
    assert info.status == "FAILED"


@pytest.mark.asyncio
async def test_falls_back_to_retrieve_on_an_older_sdk(
    flowmesh_manager: FlowmeshRuntimeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = _Tasks()
    _use(monkeypatch, tasks)
    info = await flowmesh_manager.fetch_task_status("t1")
    assert tasks.calls == ["retrieve"]
    assert info.status == "DONE"


@pytest.mark.asyncio
async def test_a_failed_status_view_still_records_its_error(
    flowmesh_manager: FlowmeshRuntimeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use(monkeypatch, _TasksWithStatus())
    statuses = {"t1": "DISPATCHED"}
    done = await flowmesh_manager.update_task_status(statuses)
    assert done is True and statuses == {"t1": "FAILED"}
    assert flowmesh_manager._execution_task_errors["t1"] == "boom"
