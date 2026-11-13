import pytest

from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager
from tests.runtime.runtime_manager.flowmesh.conftest import serve


@pytest.mark.asyncio
async def test_update_task_status_refreshes_a_workflow_in_one_request(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One status listing per poll refreshes the batch's pending tasks; a
    failed task records its redacted error, and a terminal batch is not
    polled."""
    serve(
        monkeypatch,
        {
            "task-0": ("DONE", None),
            "task-1": ("DONE", None),
            "task-2": ("FAILED", None),
            "task-3": ("PENDING", None),
        },
        {"task-2": "Authorization: Bearer secret"},
    )
    task_statuses = {f"task-{i}": "PENDING" for i in range(1_500)}
    task_statuses["task-0"] = "DONE"

    assert not await flowmesh_manager.update_task_status(task_statuses, "wf-1")
    assert task_statuses["task-1"] == "DONE"
    assert task_statuses["task-2"] == "FAILED"
    assert task_statuses["task-3"] == "PENDING"
    assert task_statuses["task-4"] == "PENDING"
    assert (
        flowmesh_manager._execution_task_errors["task-2"]
        == "Authorization: Bearer ***REDACTED***"
    )

    assert await flowmesh_manager.update_task_status(
        {"task-1": "DONE", "task-2": "FAILED"}, "wf-1"
    )
