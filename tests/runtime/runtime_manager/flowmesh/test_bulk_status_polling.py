from typing import Any

import pytest
from flowmesh.models.tasks import TaskStatusView

from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


def _status(task_id: str, status: str, error: str | None = None) -> TaskStatusView:
    return TaskStatusView(
        task_id=task_id,
        workflow_id="wf-1",
        status=status,
        submitted_at="2026-01-01T00:00:00Z",
        submitted_ts=0.0,
        attempts=1,
        max_attempts=1,
        completed=status == "DONE",
        failed=status == "FAILED",
        error=error,
    )


@pytest.mark.asyncio
async def test_workflow_status_update_keeps_unmapped_incomplete_task_open(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _list_statuses(*, workflow_id: str) -> list[TaskStatusView]:
        assert workflow_id == "wf-1"
        return [_status("task-1", "DONE")]

    class _FakeTasks:
        list_statuses = staticmethod(_list_statuses)

    class _FakeFm:
        tasks = _FakeTasks()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

    task_status = {"task-1": "PENDING", "task-2": "PENDING"}

    completed = await flowmesh_manager._update_workflow_task_statuses(
        task_status, {"task-1": "wf-1"}
    )

    assert not completed
    assert task_status == {"task-1": "DONE", "task-2": "PENDING"}


@pytest.mark.asyncio
async def test_update_task_status_polls_many_tasks_in_one_bulk_request(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    task_status = {f"task-{i}": "PENDING" for i in range(1_500)}
    task_status["task-0"] = "DONE"

    response = [
        _status("task-0", "PENDING"),
        _status("task-1", "DONE"),
        _status("task-2", "FAILED", error="Authorization: Bearer secret"),
        _status("task-3", "PENDING"),
    ]

    async def _list_statuses(*, workflow_id: str) -> list[TaskStatusView]:
        calls.append(workflow_id)
        return response

    class _FakeTasks:
        list_statuses = staticmethod(_list_statuses)

    class _FakeFm:
        tasks = _FakeTasks()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

    completed = await flowmesh_manager.update_task_status(task_status, "wf-1")

    assert calls == ["wf-1"]
    assert not completed
    assert task_status["task-0"] == "DONE"
    assert task_status["task-1"] == "DONE"
    assert task_status["task-2"] == "FAILED"
    assert task_status["task-3"] == "PENDING"
    assert task_status["task-4"] == "PENDING"
    assert (
        flowmesh_manager._execution_task_errors["task-2"]
        == "Authorization: Bearer ***REDACTED***"
    )


@pytest.mark.asyncio
async def test_update_task_status_skips_bulk_request_when_all_tasks_terminal(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _list_statuses(*, workflow_id: str) -> list[Any]:
        raise AssertionError("terminal tasks must not be polled")

    class _FakeTasks:
        list_statuses = staticmethod(_list_statuses)

    class _FakeFm:
        tasks = _FakeTasks()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

    task_status = {"task-1": "DONE", "task-2": "FAILED"}

    assert await flowmesh_manager.update_task_status(task_status, "wf-1")
    assert task_status == {"task-1": "DONE", "task-2": "FAILED"}
