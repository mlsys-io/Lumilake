from typing import Any, cast

import pytest
from support.runtime_server import RecordingRuntimeManager

from lumilake_server.runtime.server import ExecutionBatchContext


@pytest.mark.asyncio
async def test_cancel_request_propagates_to_execution_for_single_member(
    server_factory,
) -> None:
    server = server_factory()
    runtime_manager = RecordingRuntimeManager()
    server.runtime_manager = cast(Any, runtime_manager)
    server._execution_contexts["exec-1"] = ExecutionBatchContext(
        execution_request_id="exec-1",
        batch_id="batch-1",
        request_ids=("req-a",),
        workflow_ids=("wf-a",),
    )
    server._request_execution_ids["req-a"] = {"exec-1"}

    await server.cancel_request("req-a")

    assert runtime_manager.cancel_calls == ["req-a", "exec-1"]


@pytest.mark.asyncio
async def test_cancel_request_keeps_execution_running_until_all_members_cancelled(
    server_factory,
) -> None:
    server = server_factory()
    runtime_manager = RecordingRuntimeManager()
    server.runtime_manager = cast(Any, runtime_manager)
    server._execution_contexts["exec-1"] = ExecutionBatchContext(
        execution_request_id="exec-1",
        batch_id="batch-1",
        request_ids=("req-a", "req-b"),
        workflow_ids=("wf-a", "wf-b"),
    )
    server._request_execution_ids["req-a"] = {"exec-1"}
    server._request_execution_ids["req-b"] = {"exec-1"}

    await server.cancel_request("req-a")
    assert runtime_manager.cancel_calls == ["req-a"]

    await server.cancel_request("req-b")
    assert runtime_manager.cancel_calls == ["req-a", "req-b", "exec-1"]
