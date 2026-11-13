import pytest

from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager
from tests.runtime.runtime_manager.flowmesh.conftest import serve


@pytest.mark.asyncio
async def test_update_task_status_polls_one_workflow_listing(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A poll refreshes the batch's pending tasks from one workflow status
    listing, records a failed task's redacted error, and skips a terminal batch."""
    polled = serve(
        monkeypatch,
        {"task-0": "DONE", "task-1": "FAILED", "task-2": "PENDING"},
        errors={"task-1": "Authorization: Bearer secret"},
    )
    statuses = {"task-0": "PENDING", "task-1": "PENDING", "task-2": "PENDING"}

    assert not await flowmesh_manager.update_task_status(statuses, "wf-1")
    assert polled == ["wf-1"]
    assert statuses == {"task-0": "DONE", "task-1": "FAILED", "task-2": "PENDING"}
    assert (
        flowmesh_manager._execution_task_errors["task-1"]
        == "Authorization: Bearer ***REDACTED***"
    )

    assert await flowmesh_manager.update_task_status(
        {"task-0": "DONE", "task-1": "FAILED"}, "wf-1"
    )
    assert polled == ["wf-1"]
