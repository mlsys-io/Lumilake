import pytest
from flowmesh.models.tasks import TaskStatusPage

from lumilake_server.runtime.runtime_manager import FlowmeshRuntimeManager
from tests.runtime.runtime_manager.conftest import serve


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


@pytest.mark.asyncio
async def test_a_poll_reads_each_transition_once_and_stops_its_cursor(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first poll takes a snapshot and keeps the cursor the page returns;
    the next sends it, so only rows changed after it are read, and a finished
    workflow stops its cursor."""
    cursors: list[str | None] = []
    served = {"task-1": "DISPATCHED"}
    serve(monkeypatch, served, cursors=cursors)
    task_statuses = {"task-1": "PENDING"}

    assert not await flowmesh_manager.update_task_status(task_statuses, "wf-1")
    assert cursors == [None]
    assert flowmesh_manager._workflow_status_cursors == {"wf-1": "task-1"}

    served["task-1"] = "DONE"
    assert await flowmesh_manager.update_task_status(task_statuses, "wf-1")
    assert cursors == [None, "task-1"]
    assert flowmesh_manager._workflow_status_cursors == {}

    assert await flowmesh_manager.update_task_status({"task-1": "PENDING"}, "wf-1")
    assert cursors == [None, "task-1", None]


@pytest.mark.asyncio
async def test_a_poll_drains_every_page_and_relists_an_expired_cursor(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A poll reads on while ``has_more``; an expired cursor relists once."""
    pages = [
        TaskStatusPage(rows=[{"task_id": "task-1", "status": s}], next_cursor=c)
        for s, c in (("DISPATCHED", "5"), ("DONE", "6"))
    ]
    pages[0].has_more = True
    cursors: list[str | None] = []
    serve(monkeypatch, {}, cursors=cursors, pages=pages)
    assert await flowmesh_manager.update_task_status({"task-1": "PENDING"}, "wf-1")
    assert cursors == [None, "5"]

    flowmesh_manager._workflow_status_cursors["wf-1"] = "3"
    serve(monkeypatch, {"task-1": "DONE"}, cursors=cursors, expired={"3"})
    assert await flowmesh_manager.update_task_status({"task-1": "PENDING"}, "wf-1")
    assert cursors[-2:] == ["3", None]


@pytest.mark.asyncio
async def test_a_progress_read_leaves_a_running_loops_batch_and_cursor_alone(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A progress read neither moves the loop's cursor nor reads its feed."""
    cursors: list[str | None] = []
    serve(monkeypatch, {"task-1": "DISPATCHED"}, cursors=cursors)
    flowmesh_manager._workflow_status_cursors["wf-1"] = "1:2"

    batch_key = ("req-1", "batch-1")
    loop_status = {"task-1": "PENDING", "task-2": "PENDING"}
    statuses_by_batch = {batch_key: loop_status}

    assert not await flowmesh_manager._update_batch_task_statuses(
        statuses_by_batch, [batch_key], {}
    )
    assert not await flowmesh_manager._update_batch_task_statuses(
        statuses_by_batch, [batch_key], {batch_key: "wf-1"}
    )
    assert flowmesh_manager._workflow_status_cursors == {"wf-1": "1:2"}

    flowmesh_manager._active_execution_loops.add(batch_key)
    assert not await flowmesh_manager._update_batch_task_statuses(
        statuses_by_batch, [batch_key], {batch_key: "wf-1"}
    )

    assert cursors == [None]
    assert loop_status == {"task-1": "DISPATCHED", "task-2": "PENDING"}
