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
    the next poll sends it, so only the rows that changed after it are read. A
    finished workflow stops its cursor, so a later poll lists every task
    again."""
    cursors: list[str | None] = []
    serve(monkeypatch, {"task-1": "DISPATCHED"}, cursors=cursors)
    task_statuses = {"task-1": "PENDING"}

    assert not await flowmesh_manager.update_task_status(task_statuses, "wf-1")
    assert cursors == [None]
    assert flowmesh_manager._workflow_status_cursors == {"wf-1": "task-1"}

    serve(monkeypatch, {"task-1": "DONE"}, cursors=cursors)
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
    """A page carries only part of the changes after a cursor, so a poll reads
    until ``has_more`` is false; an unresumable cursor is dropped and the poll
    restarts from a snapshot."""
    cursors: list[str | None] = []
    serve(
        monkeypatch,
        {},
        cursors=cursors,
        pages=[
            TaskStatusPage(
                rows=[{"task_id": "task-1", "status": "DISPATCHED"}],
                next_cursor="5",
                has_more=True,
            ),
            TaskStatusPage(
                rows=[{"task_id": "task-1", "status": "DONE"}], next_cursor="6"
            ),
        ],
    )
    task_statuses = {"task-1": "PENDING"}
    assert await flowmesh_manager.update_task_status(task_statuses, "wf-1")
    assert cursors == [None, "5"]
    assert task_statuses == {"task-1": "DONE"}

    flowmesh_manager._workflow_status_cursors["wf-1"] = "3"
    serve(monkeypatch, {"task-1": "DONE"}, cursors=cursors, expired={"3"})
    task_statuses = {"task-1": "PENDING"}
    assert await flowmesh_manager.update_task_status(task_statuses, "wf-1")
    assert cursors[-2:] == ["3", None]
    assert task_statuses == {"task-1": "DONE"}


@pytest.mark.asyncio
async def test_a_progress_read_leaves_a_running_loops_batch_and_cursor_alone(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the execution loop advances the cursor of a workflow it runs. A
    progress read lists every task and stores nothing, so the narrower view of
    a caller's own tasks never moves a position the loop would resume from, and
    a batch whose loop is running is reported as the loop's map has it."""
    cursors: list[str | None] = []
    serve(monkeypatch, {"task-1": "DISPATCHED"}, cursors=cursors)
    flowmesh_manager._workflow_status_cursors["wf-1"] = "1:2"

    batch_key = ("req-1", "batch-1")
    loop_status = {"task-1": "PENDING", "task-2": "PENDING"}
    statuses_by_batch = {batch_key: loop_status}

    # A batch with no workflow id cannot be read, so it is not finished; a
    # progress read of a batch no loop is reading stores no cursor.
    assert not await flowmesh_manager._update_batch_task_statuses(
        statuses_by_batch, [batch_key], {}
    )
    assert not await flowmesh_manager._update_batch_task_statuses(
        statuses_by_batch, [batch_key], {batch_key: "wf-1"}
    )
    assert flowmesh_manager._workflow_status_cursors == {"wf-1": "1:2"}

    # While the loop runs, its own map is reported and the feed is not read.
    flowmesh_manager._active_execution_loops.add(batch_key)
    assert not await flowmesh_manager._update_batch_task_statuses(
        statuses_by_batch, [batch_key], {batch_key: "wf-1"}
    )

    assert cursors == [None]
    assert loop_status == {"task-1": "DISPATCHED", "task-2": "PENDING"}
