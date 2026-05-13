from unittest.mock import patch

import pytest

from lumilake.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


def _manager() -> FlowmeshRuntimeManager:
    return FlowmeshRuntimeManager()


@pytest.mark.parametrize(
    ("request_id", "batch_id", "total_nodes", "start_t", "end_t", "final_t", "done"),
    [
        ("req-1", "batch-1", 4, 100.0, 130.0, 3000.0, True),
        ("req-2", "batch-2", 2, 200.0, None, 260.0, False),
    ],
)
def test_batch_progress_elapsed_and_node_totals(
    request_id: str,
    batch_id: str,
    total_nodes: int,
    start_t: float,
    end_t: float | None,
    final_t: float,
    done: bool,
) -> None:
    manager = _manager()
    with patch(
        "lumilake.runtime.runtime_manager.flowmesh.time.time", return_value=start_t
    ):
        manager.mark_batch_pending(
            request_id,
            batch_id,
            total_nodes=total_nodes,
            output_nodes=1,
        )
        manager.mark_batch_running(request_id, batch_id)
    if end_t is not None:
        with patch(
            "lumilake.runtime.runtime_manager.flowmesh.time.time", return_value=end_t
        ):
            manager.mark_batch_completed(request_id, batch_id)
    with patch(
        "lumilake.runtime.runtime_manager.flowmesh.time.time", return_value=final_t
    ):
        progress = manager._build_batch_progress([(request_id, batch_id)])

    assert progress["overall_progress"]["raw_nodes"] == total_nodes
    assert progress["overall_progress"]["flowmesh_nodes"] == total_nodes
    batch_info = progress["batches"][0]
    if done:
        assert progress["completed"] == 1
        assert progress["running"] == 0
        assert batch_info["status"] == "COMPLETED"
    else:
        assert progress["completed"] == 0
        assert progress["running"] == 1
        assert batch_info["status"] == "RUNNING"
    expected_elapsed = 30.0 if done else 60.0
    assert batch_info["elapsed_time"] == expected_elapsed


def test_batch_progress_counts_flowmesh_task_status_values() -> None:
    manager = _manager()
    request_id = "req-3"
    batch_id = "batch-3"
    batch_key = (request_id, batch_id)
    manager.mark_batch_pending(request_id, batch_id, total_nodes=4, output_nodes=1)
    manager.mark_batch_running(request_id, batch_id)
    manager._execution_task_status[batch_key] = {
        "t1": "PENDING",
        "t2": "DISPATCHED",
        "t3": "DONE",
        "t4": "FAILED",
    }

    progress = manager._build_batch_progress([batch_key])
    nodes = progress["batches"][0]["nodes"]
    assert nodes == {
        "total": 4,
        "succeeded": 1,
        "failed": 1,
        "dispatched": 1,
        "pending": 1,
    }


def test_batch_progress_counts_multiple_running_batches() -> None:
    manager = _manager()
    request_id = "req-4"
    first_batch = "batch-1"
    second_batch = "batch-2"
    first_key = (request_id, first_batch)
    second_key = (request_id, second_batch)
    manager.mark_batch_pending(request_id, first_batch, total_nodes=2, output_nodes=1)
    manager.mark_batch_pending(request_id, second_batch, total_nodes=3, output_nodes=1)
    manager.mark_batch_running(request_id, first_batch)
    manager.mark_batch_running(request_id, second_batch)
    manager._execution_task_status[first_key] = {
        "a": "DISPATCHED",
        "b": "PENDING",
    }
    manager._execution_task_status[second_key] = {
        "c": "DONE",
        "d": "FAILED",
        "e": "PENDING",
    }

    progress = manager._build_batch_progress([first_key, second_key])

    assert progress["running"] == 2
    statuses = [item["status"] for item in progress["batches"]]
    assert statuses == ["RUNNING", "RUNNING"]
    batch_ids = [item["batch_id"] for item in progress["batches"]]
    assert batch_ids == [first_batch, second_batch]
