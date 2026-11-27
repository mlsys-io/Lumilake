from typing import Any, cast

import pytest
from support.runtime_server import FakeHandler, RecordingRuntimeManager

from lumilake_server.runtime.protocol import LumilakeRequestConfig
from lumilake_server.runtime.request import RequestHandler
from lumilake_server.runtime.server import RequestState
from lumilake_server.schemas.progress import (
    BatchNodeStats,
    BatchProgress,
    BatchProgressEntry,
    BatchStatus,
    JobProgress,
    OverallProgress,
    ProgressDetails,
    ProgressStep,
    RuntimeStatus,
)


def _status(
    *,
    succeeded: int = 0,
    pending: int = 0,
    completed: bool = False,
    batches: tuple[BatchProgressEntry, ...] = (),
    batch_completed: int = 0,
    batch_running: int = 0,
    eta_seconds: float | None = None,
) -> RuntimeStatus:
    return RuntimeStatus(
        execution=ProgressStep(
            completed=completed,
            details=ProgressDetails(succeeded=succeeded, pending=pending),
        ),
        batch_progress=BatchProgress(
            total=len(batches),
            completed=batch_completed,
            running=batch_running,
            batches=list(batches),
            overall_progress=OverallProgress(
                total_nodes=sum(batch.nodes.total for batch in batches),
                completed_nodes=sum(batch.nodes.succeeded for batch in batches),
            ),
            eta_seconds=eta_seconds,
        ),
    )


def _batch(
    batch_id: str, status: BatchStatus, succeeded: int, pending: int, total: int
) -> BatchProgressEntry:
    return BatchProgressEntry(
        batch_id=batch_id,
        status=status,
        nodes=BatchNodeStats(succeeded=succeeded, pending=pending, total=total),
    )


@pytest.mark.asyncio
async def test_get_request_status_mirrors_shared_execution_progress(
    server_factory,
) -> None:
    server = server_factory()
    runtime_manager = RecordingRuntimeManager(
        status_by_request={
            "exec-1": _status(
                succeeded=1,
                pending=2,
                batches=(_batch("batch-1", "RUNNING", 1, 2, 3),),
                batch_running=1,
                eta_seconds=10.0,
            )
        }
    )
    server.runtime_manager = cast(Any, runtime_manager)

    server._progress["req-a"] = JobProgress()
    server._progress["req-b"] = JobProgress()
    server._request_execution_ids["req-a"] = {"exec-1"}
    server._request_execution_ids["req-b"] = {"exec-1"}

    status_a = await server.get_request_status("req-a")
    status_b = await server.get_request_status("req-b")

    assert status_a is not None
    assert status_b is not None
    assert status_a.execution is not None
    assert status_b.execution is not None
    assert status_a.execution.details.succeeded == 1
    assert status_b.execution.details.pending == 2
    assert status_a.batch_progress.running == 1
    assert status_b.batch_progress.overall_progress.total_nodes == 3


@pytest.mark.asyncio
async def test_get_request_status_runtime_percentage_is_capped(server_factory) -> None:
    server = server_factory()
    runtime_manager = RecordingRuntimeManager(
        status_by_request={
            "exec-1": _status(
                succeeded=3,
                batches=(_batch("batch-1", "RUNNING", 3, 0, 3),),
                batch_running=1,
            )
        }
    )
    server.runtime_manager = cast(Any, runtime_manager)
    server._progress["req-a"] = JobProgress()
    server._request_execution_ids["req-a"] = {"exec-1"}
    server._requests["req-a"] = RequestState(
        handler=cast(RequestHandler, FakeHandler()),
        config=LumilakeRequestConfig(user_id="req-a", principal_id="req-a"),
        pending_workflows=set(),
        workflow_lengths={},
        pending_runtime_nodes_raw=0,
        processing_runtime_nodes_optimized=2,
        processed_runtime_nodes_optimized=0,
        ready=True,
    )

    status = await server.get_request_status("req-a")

    assert status is not None
    overall = status.batch_progress.overall_progress
    assert overall.total_nodes_runtime == 2
    assert overall.completed_nodes == 3
    assert overall.percentage == 100.0


@pytest.mark.asyncio
async def test_get_request_status_projects_input_completion(server_factory) -> None:
    server = server_factory()
    runtime_manager = RecordingRuntimeManager(
        status_by_request={
            "exec-1": _status(
                batches=(_batch("batch-1", "COMPLETED", 2, 0, 2),),
                batch_completed=1,
                batch_running=1,
                eta_seconds=10.0,
            )
        }
    )
    server.runtime_manager = cast(Any, runtime_manager)
    server._progress["req-a"] = JobProgress()
    server._request_execution_ids["req-a"] = {"exec-1"}
    server._requests["req-a"] = RequestState(
        handler=cast(RequestHandler, FakeHandler()),
        config=LumilakeRequestConfig(user_id="req-a", principal_id="req-a"),
        pending_workflows=set(),
        workflow_lengths={"workflow-a": 4, "workflow-b": 6},
        total_input_items=10,
        completed_input_items_success=4,
        pending_runtime_nodes_raw=0,
        ready=True,
    )

    status = await server.get_request_status("req-a")

    assert status is not None
    overall = status.batch_progress.overall_progress
    assert overall.total_inputs == 10
    assert overall.completed_inputs == 4


@pytest.mark.asyncio
async def test_get_request_status_keeps_completed_batches_with_running_batch(
    server_factory,
) -> None:
    server = server_factory()
    runtime_manager = RecordingRuntimeManager(
        status_by_request={
            "exec-old": _status(
                succeeded=2,
                completed=True,
                batches=(_batch("batch-old", "COMPLETED", 2, 0, 2),),
                batch_completed=1,
            ),
            "exec-new": _status(
                succeeded=1,
                pending=1,
                batches=(_batch("batch-new", "RUNNING", 1, 1, 2),),
                batch_running=1,
                eta_seconds=12.0,
            ),
        }
    )
    server.runtime_manager = cast(Any, runtime_manager)
    server._progress["req-a"] = JobProgress()
    server._request_execution_ids["req-a"] = {"exec-new"}
    server._request_execution_history_ids["req-a"] = {"exec-old", "exec-new"}

    status = await server.get_request_status("req-a")

    assert status is not None
    assert status.batch_progress.completed == 1
    assert status.batch_progress.running == 1
    batch_ids = [batch.batch_id for batch in status.batch_progress.batches]
    assert batch_ids == ["batch-new", "batch-old"]
