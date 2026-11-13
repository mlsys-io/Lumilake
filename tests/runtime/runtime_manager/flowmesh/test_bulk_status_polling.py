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


@pytest.mark.asyncio
async def test_a_poll_reads_each_transition_once_and_stops_its_cursor(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first poll lists every task and keeps the feed cursor the page
    returns; the next poll sends that cursor, so only the rows that changed
    after it are read. A finished workflow stops its cursor, so a later poll of
    the same id lists every task again instead of resuming a spent position."""
    cursors: list[str | None] = []
    serve(
        monkeypatch,
        {"task-1": ("DISPATCHED", None)},
        cursors=cursors,
        page_cursor="1:2",
    )
    task_statuses = {"task-1": "PENDING"}

    assert not await flowmesh_manager.update_task_status(task_statuses, "wf-1")
    assert cursors == [None]
    assert flowmesh_manager._workflow_status_cursors == {"wf-1": "1:2"}

    serve(monkeypatch, {"task-1": ("DONE", None)}, cursors=cursors, page_cursor="1:3")
    assert await flowmesh_manager.update_task_status(task_statuses, "wf-1")
    assert cursors == [None, "1:2"]
    assert flowmesh_manager._workflow_status_cursors == {}

    assert await flowmesh_manager.update_task_status({"task-1": "PENDING"}, "wf-1")
    assert cursors == [None, "1:2", None]


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
    serve(monkeypatch, {"task-1": ("DISPATCHED", None)}, cursors=cursors)
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
