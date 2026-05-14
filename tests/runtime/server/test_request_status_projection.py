from typing import Any, cast

import pytest
from support.runtime_server import FakeHandler, RecordingRuntimeManager

from lumilake_server.runtime.protocol import LumilakeRequestConfig
from lumilake_server.runtime.request import RequestHandler
from lumilake_server.runtime.server import RequestState
from lumilake_server.schemas.progress import JobProgress


@pytest.mark.asyncio
async def test_get_request_status_mirrors_shared_execution_progress(
    server_factory,
) -> None:
    server = server_factory()
    runtime_manager = RecordingRuntimeManager(
        status_by_request={
            "exec-1": {
                "execution": {
                    "completed": False,
                    "details": {
                        "succeeded": 1,
                        "failed": 0,
                        "pending": 2,
                        "dispatched": 0,
                    },
                },
                "batch_progress": {
                    "total": 1,
                    "completed": 0,
                    "running": 1,
                    "pending": 0,
                    "failed": 0,
                    "batches": [
                        {
                            "batch_id": "batch-1",
                            "status": "RUNNING",
                            "nodes": {
                                "succeeded": 1,
                                "failed": 0,
                                "pending": 2,
                                "dispatched": 0,
                                "total": 3,
                            },
                        }
                    ],
                    "overall_progress": {
                        "total_nodes": 3,
                        "completed_nodes": 1,
                        "percentage": 33.3,
                    },
                    "eta_seconds": 10.0,
                },
            }
        }
    )
    server.runtime_manager = cast(Any, runtime_manager)

    server._progress["req-a"] = JobProgress()
    server._progress["req-b"] = JobProgress()
    server._request_execution_ids["req-a"] = {"exec-1"}
    server._request_execution_ids["req-b"] = {"exec-1"}

    status_a = await server.get_request_status("req-a")
    status_b = await server.get_request_status("req-b")

    assert "error" not in status_a
    assert "error" not in status_b
    assert status_a["execution"]["details"]["succeeded"] == 1
    assert status_b["execution"]["details"]["pending"] == 2
    assert status_a["batch_progress"]["running"] == 1
    assert status_b["batch_progress"]["overall_progress"]["total_nodes"] == 3


@pytest.mark.asyncio
async def test_get_request_status_runtime_percentage_is_capped(server_factory) -> None:
    server = server_factory()
    runtime_manager = RecordingRuntimeManager(
        status_by_request={
            "exec-1": {
                "execution": {
                    "completed": False,
                    "details": {
                        "succeeded": 3,
                        "failed": 0,
                        "pending": 0,
                        "dispatched": 0,
                    },
                },
                "batch_progress": {
                    "total": 1,
                    "completed": 0,
                    "running": 1,
                    "pending": 0,
                    "failed": 0,
                    "batches": [
                        {
                            "batch_id": "batch-1",
                            "status": "RUNNING",
                            "nodes": {
                                "succeeded": 3,
                                "failed": 0,
                                "pending": 0,
                                "dispatched": 0,
                                "total": 3,
                            },
                        }
                    ],
                    "overall_progress": {
                        "total_nodes": 3,
                        "completed_nodes": 3,
                        "percentage": 100.0,
                    },
                    "eta_seconds": None,
                },
            }
        }
    )
    server.runtime_manager = cast(Any, runtime_manager)
    server._progress["req-a"] = JobProgress()
    server._request_execution_ids["req-a"] = {"exec-1"}
    server._requests["req-a"] = RequestState(
        handler=cast(RequestHandler, FakeHandler()),
        config=LumilakeRequestConfig(user_id="req-a"),
        pending_workflows=set(),
        workflow_lengths={},
        pending_runtime_nodes_raw=0,
        processing_runtime_nodes_optimized=2,
        processed_runtime_nodes_optimized=0,
        ready=True,
    )

    status = await server.get_request_status("req-a")

    assert "error" not in status
    assert status["batch_progress"]["overall_progress"]["total_nodes_runtime"] == 2
    assert status["batch_progress"]["overall_progress"]["completed_nodes"] == 3
    assert status["batch_progress"]["overall_progress"]["percentage"] == 100.0


@pytest.mark.asyncio
async def test_get_request_status_projects_input_completion(server_factory) -> None:
    server = server_factory()
    runtime_manager = RecordingRuntimeManager(
        status_by_request={
            "exec-1": {
                "batch_progress": {
                    "total": 2,
                    "completed": 1,
                    "running": 1,
                    "pending": 0,
                    "failed": 0,
                    "batches": [
                        {
                            "batch_id": "batch-1",
                            "status": "COMPLETED",
                            "nodes": {
                                "succeeded": 2,
                                "failed": 0,
                                "pending": 0,
                                "dispatched": 0,
                                "total": 2,
                            },
                        }
                    ],
                    "overall_progress": {
                        "total_nodes": 2,
                        "completed_nodes": 2,
                        "percentage": 100.0,
                    },
                    "eta_seconds": 10.0,
                }
            }
        }
    )
    server.runtime_manager = cast(Any, runtime_manager)
    server._progress["req-a"] = JobProgress()
    server._request_execution_ids["req-a"] = {"exec-1"}
    server._requests["req-a"] = RequestState(
        handler=cast(RequestHandler, FakeHandler()),
        config=LumilakeRequestConfig(user_id="req-a"),
        pending_workflows=set(),
        workflow_lengths={"workflow-a": 4, "workflow-b": 6},
        total_input_items=10,
        completed_input_items_success=4,
        pending_runtime_nodes_raw=0,
        ready=True,
    )

    status = await server.get_request_status("req-a")

    assert "error" not in status
    overall = status["batch_progress"]["overall_progress"]
    assert overall["total_inputs"] == 10
    assert overall["completed_inputs"] == 4


@pytest.mark.asyncio
async def test_get_request_status_keeps_completed_batches_with_running_batch(
    server_factory,
) -> None:
    server = server_factory()
    runtime_manager = RecordingRuntimeManager(
        status_by_request={
            "exec-old": {
                "execution": {
                    "completed": True,
                    "details": {
                        "succeeded": 2,
                        "failed": 0,
                        "pending": 0,
                        "dispatched": 0,
                    },
                },
                "batch_progress": {
                    "total": 1,
                    "completed": 1,
                    "running": 0,
                    "pending": 0,
                    "failed": 0,
                    "batches": [
                        {
                            "batch_id": "batch-old",
                            "status": "COMPLETED",
                            "nodes": {
                                "succeeded": 2,
                                "failed": 0,
                                "pending": 0,
                                "dispatched": 0,
                                "total": 2,
                            },
                        }
                    ],
                    "overall_progress": {
                        "total_nodes": 2,
                        "completed_nodes": 2,
                        "percentage": 100.0,
                    },
                    "eta_seconds": None,
                },
            },
            "exec-new": {
                "execution": {
                    "completed": False,
                    "details": {
                        "succeeded": 1,
                        "failed": 0,
                        "pending": 1,
                        "dispatched": 0,
                    },
                },
                "batch_progress": {
                    "total": 1,
                    "completed": 0,
                    "running": 1,
                    "pending": 0,
                    "failed": 0,
                    "batches": [
                        {
                            "batch_id": "batch-new",
                            "status": "RUNNING",
                            "nodes": {
                                "succeeded": 1,
                                "failed": 0,
                                "pending": 1,
                                "dispatched": 0,
                                "total": 2,
                            },
                        }
                    ],
                    "overall_progress": {
                        "total_nodes": 2,
                        "completed_nodes": 1,
                        "percentage": 50.0,
                    },
                    "eta_seconds": 12.0,
                },
            },
        }
    )
    server.runtime_manager = cast(Any, runtime_manager)
    server._progress["req-a"] = JobProgress()
    server._request_execution_ids["req-a"] = {"exec-new"}
    server._request_execution_history_ids["req-a"] = {"exec-old", "exec-new"}

    status = await server.get_request_status("req-a")

    assert "error" not in status
    assert status["batch_progress"]["completed"] == 1
    assert status["batch_progress"]["running"] == 1
    batch_ids = [batch["batch_id"] for batch in status["batch_progress"]["batches"]]
    assert batch_ids == ["batch-new", "batch-old"]
