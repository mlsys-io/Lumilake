import asyncio
from typing import Any, cast

import pytest
from support.runtime_server import (
    RecordingRuntimeManager,
    attach_request_states,
    make_batch,
    make_workflow,
)

from lumilake_server.runtime.capacity import FreeCapacity
from lumilake_server.runtime.job_manager.base import Job
from lumilake_server.runtime.request import WorkflowSliceMeta
from lumilake_server.runtime.server import LumilakeServer


def _patch_capacity(server: Any, cpu_ids: list[str], gpu_ids: list[str]) -> None:
    async def _snapshot() -> FreeCapacity:
        busy = server._busy_workers
        return FreeCapacity(
            cpu_worker_ids=tuple(w for w in cpu_ids if w not in busy),
            gpu_worker_ids=tuple(w for w in gpu_ids if w not in busy),
        )

    async def _total_snapshot() -> FreeCapacity:
        return FreeCapacity(
            cpu_worker_ids=tuple(cpu_ids),
            gpu_worker_ids=tuple(gpu_ids),
        )

    server._snapshot_free_capacity = _snapshot  # type: ignore[method-assign]
    server._snapshot_total_capacity = _total_snapshot  # type: ignore[method-assign]


async def _enqueue_job(server: LumilakeServer, workflow: Any) -> Any:
    job = Job(
        request_id=workflow.request_id,
        runtime_graphs={workflow.graph_name: workflow.runtime_graph},
        data_profile_graphs={workflow.graph_name: workflow.data_profile_graph},
        dsl_graphs={workflow.graph_name: workflow.dsl_graph},
        workflow_slices={
            workflow.graph_name: WorkflowSliceMeta(
                public_graph_name=workflow.public_graph_name,
                slice_index=workflow.slice_index,
                slice_start=workflow.slice_start,
                slice_length=workflow.slice_length,
                total_length=workflow.total_length,
                template_hash=workflow.template_hash,
                varying_input_keys=workflow.varying_input_keys,
            )
        },
        config=workflow.config,
    )
    enqueued = await server.job_manager.enqueue(job)
    return enqueued[0]


@pytest.mark.asyncio
async def test_dispatch_loop_records_one_row_per_workflow(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two jobs queued behind each other each get one raw dispatch row with
    the right batch ids, and no summary or duration field is added."""
    server = server_factory()
    server.config.batch_size = 1
    server.runtime_manager = cast(Any, RecordingRuntimeManager())

    workflows = [
        make_workflow(
            workflow_id="wf-a",
            request_id="req-a",
            graph_name="ga",
            public_graph_name="shared",
        ),
        make_workflow(
            workflow_id="wf-b",
            request_id="req-b",
            graph_name="gb",
            public_graph_name="shared",
        ),
    ]
    attach_request_states(server, workflows)
    enqueued = [await _enqueue_job(server, workflow) for workflow in workflows]

    async def _no_accumulation_wait(free: Any = None) -> None:
        return

    async def _fake_process_batch(
        selected_batch: Any,
        batch_id: str,
        selected_workers: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        *,
        execution_request_id: str,
        member_request_ids: set[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        outputs = {
            item.workflow_id: {"output": [f"value-{item.request_id}"]}
            for item in selected_batch.workflows
        }
        return outputs, {}

    _patch_capacity(server, ["cpu-0", "cpu-1"], [])
    server._maybe_wait_for_batch_accumulation = _no_accumulation_wait  # type: ignore[method-assign]
    monkeypatch.setattr(server, "_process_batch", _fake_process_batch)

    scheduler_task = asyncio.create_task(server._scheduler_loop())
    for _ in range(200):
        if server.workflow_dispatches_for_request(
            "req-a"
        ) and server.workflow_dispatches_for_request("req-b"):
            break
        await asyncio.sleep(0)
    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass

    rows_a = server.workflow_dispatches_for_request("req-a")
    rows_b = server.workflow_dispatches_for_request("req-b")

    assert len(rows_a) == 1
    assert len(rows_b) == 1
    row_a = rows_a[0]
    row_b = rows_b[0]

    # One row per workflow, keyed to the right workflow.
    assert row_a.workflow_id == enqueued[0].workflow_id
    assert row_b.workflow_id == enqueued[1].workflow_id
    assert row_a.graph_name == "ga"
    assert row_b.graph_name == "gb"
    assert row_a.public_graph_name == "shared"

    # Each job was its own batch (batch_size=1), so each row's batch holds
    # only that workflow.
    assert row_a.batch_id != row_b.batch_id
    assert row_a.batch_workflow_ids == [enqueued[0].workflow_id]
    assert row_b.batch_workflow_ids == [enqueued[1].workflow_id]
    assert row_a.workers == ["cpu-0"]
    assert row_b.workers == ["cpu-1"]

    # enqueued_at <= dispatched_at, and dispatched_at is a raw epoch second.
    assert row_a.enqueued_at <= row_a.dispatched_at
    assert row_b.enqueued_at <= row_b.dispatched_at
    assert row_a.dispatched_at > 0

    # Raw facts only: no summary or duration field was added.
    dumped = row_a.model_dump()
    assert "duration_seconds" not in dumped
    assert "wait_seconds" not in dumped
    assert "count" not in dumped
    assert set(dumped) == {
        "workflow_id",
        "graph_name",
        "public_graph_name",
        "slice_index",
        "slice_start",
        "slice_length",
        "total_length",
        "enqueued_at",
        "dispatched_at",
        "miss_count",
        "batch_id",
        "execution_request_id",
        "batch_workflow_ids",
        "workers",
        "flowmesh_workflow_id",
    }


@pytest.mark.asyncio
async def test_failed_batch_keeps_flowmesh_workflow_id_on_rows(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch whose ``_process_batch`` raises after the runtime manager
    recorded a FlowMesh workflow id keeps that id on its dispatch rows."""
    server = server_factory()
    runtime_manager = RecordingRuntimeManager()
    server.runtime_manager = cast(Any, runtime_manager)

    workflows = [
        make_workflow(
            workflow_id="wf-a",
            request_id="req-a",
            graph_name="ga",
            public_graph_name="shared",
        )
    ]
    attach_request_states(server, workflows)
    batch = make_batch(workflows)

    async def _raising_process_batch(
        selected_batch: Any,
        batch_id: str,
        selected_workers: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        *,
        execution_request_id: str,
        member_request_ids: set[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # The runtime manager records the FM workflow id on submit, before the
        # batch later fails.
        runtime_manager._batch_workflow_id[(execution_request_id, batch_id)] = (
            "fm-failed-1"
        )
        raise RuntimeError("boom after submit")

    monkeypatch.setattr(server, "_process_batch", _raising_process_batch)
    await server._run_batch(["worker-1"], batch)

    rows = server.workflow_dispatches_for_request("req-a")
    assert len(rows) == 1
    assert rows[0].flowmesh_workflow_id == "fm-failed-1"
