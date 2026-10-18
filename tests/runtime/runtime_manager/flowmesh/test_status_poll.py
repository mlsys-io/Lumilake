"""The status poll uses FlowMesh's status view, not the full task.

GET /tasks/{id} returns the whole task, spec and source included, about
0.5-1 MB for an inference task. ``tasks.status()`` fetches only status,
placement, timing and errors.
"""

from types import SimpleNamespace

import pytest
from flowmesh.models.tasks import TaskStatusView

from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


class _Tasks:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def status(self, task_id: str) -> TaskStatusView:
        self.calls.append("status")
        return TaskStatusView(
            task_id=task_id,
            workflow_id="wf-1",
            status="FAILED",
            submitted_at="2026-01-01T00:00:00Z",
            submitted_ts=0.0,
            attempts=1,
            max_attempts=1,
            completed=False,
            failed=True,
            error="boom",
        )


def _use(monkeypatch: pytest.MonkeyPatch, tasks: _Tasks) -> None:
    fm = SimpleNamespace(tasks=tasks)
    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: fm))


@pytest.mark.asyncio
async def test_polls_the_status_view(
    flowmesh_manager: FlowmeshRuntimeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = _Tasks()
    _use(monkeypatch, tasks)
    info = await flowmesh_manager.fetch_task_status("t1")
    assert tasks.calls == ["status"]
    assert info.status == "FAILED"


@pytest.mark.asyncio
async def test_a_failed_status_view_records_its_error(
    flowmesh_manager: FlowmeshRuntimeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use(monkeypatch, _Tasks())
    statuses = {"t1": "DISPATCHED"}
    done = await flowmesh_manager.update_task_status(statuses)
    assert done is True and statuses == {"t1": "FAILED"}
    assert flowmesh_manager._execution_task_errors["t1"] == "boom"
