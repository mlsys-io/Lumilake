import types
from types import SimpleNamespace
from typing import Any

import pytest
from flowmesh.exceptions import APIError
from lumilake import envs

from lumilake_server.common import ApiConfig, GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import LLMChatOp, OpMessage, as_output, input_placeholder
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.request import RequestInfo
from lumilake_server.runtime.runtime_graph import RuntimeGraph, RuntimeGraphBuilder
from lumilake_server.runtime.runtime_manager.flowmesh import (
    FlowmeshRuntimeManager,
    _sanitize_flowmesh_api_error,
)
from lumilake_server.utils.job_storage import InMemoryJobStorage


def test_sanitize_flowmesh_api_error_redacts_authorization_header_in_body() -> None:
    """FlowMesh's rejection body can echo back the task spec we submitted,
    Authorization header included. _sanitize_flowmesh_api_error must strip
    that credential from both the body and the exception message before the
    error is logged or re-raised."""
    original = APIError(
        "task spec invalid",
        status_code=422,
        method="POST",
        url="https://flowmesh.internal/workflows",
        body={
            "detail": "invalid task",
            "spec": {"Authorization": "Bearer sk-live-secret"},
        },
    )

    sanitized = _sanitize_flowmesh_api_error(original)

    assert isinstance(sanitized, APIError)
    assert "sk-live-secret" not in str(sanitized.body)
    assert "sk-live-secret" not in str(sanitized)
    assert "***REDACTED***" in str(sanitized.body)
    assert sanitized.status_code == 422
    assert sanitized.method == "POST"
    assert sanitized.url == "https://flowmesh.internal/workflows"


def test_sanitize_flowmesh_api_error_redacts_bearer_token_in_plain_text_body() -> None:
    """A non-JSON rejection body (plain text) can still embed a bearer
    token; the regex fallback in redact_secrets_in_text must catch it."""
    original = APIError(
        "rejected: Authorization: Bearer sk-live-secret is not permitted",
        status_code=403,
        method="POST",
        url="https://flowmesh.internal/workflows",
        body="rejected: Authorization: Bearer sk-live-secret is not permitted",
    )

    sanitized = _sanitize_flowmesh_api_error(original)

    assert "sk-live-secret" not in str(sanitized.body)
    assert "sk-live-secret" not in str(sanitized)
    assert "***REDACTED***" in str(sanitized.body)


def test_sanitize_flowmesh_api_error_truncates_long_body() -> None:
    """A pathologically large echoed body must be truncated so it cannot
    blow up log lines or the persisted job error record."""
    original = APIError(
        "task spec invalid",
        status_code=422,
        method="POST",
        url="https://flowmesh.internal/workflows",
        body={"detail": "x" * 5000},
    )

    sanitized = _sanitize_flowmesh_api_error(original)

    assert len(sanitized.body) <= 2000 + len("...[truncated]")
    assert sanitized.body.endswith("...[truncated]")


@pytest.mark.asyncio
async def test_fetch_task_status_sanitizes_api_error_before_reraising(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A FlowMesh rejection on status retrieval can echo back the submitted
    task spec, Authorization header included. fetch_task_status must re-raise
    a sanitized APIError so the credential never reaches the caller, the
    log, or the persisted job error record."""
    from flowmesh.exceptions import APIError

    class _FakeTasks:
        async def retrieve(self, task_id: str) -> Any:
            raise APIError(
                "task spec invalid",
                status_code=422,
                method="GET",
                url="https://flowmesh.internal/tasks/t1",
                body={"spec": {"Authorization": "Bearer sk-live-secret"}},
            )

    class _FakeFm:
        def __init__(self) -> None:
            self.tasks = _FakeTasks()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

    with pytest.raises(APIError) as excinfo:
        await flowmesh_manager.fetch_task_status("t1")

    assert "sk-live-secret" not in str(excinfo.value)
    assert "sk-live-secret" not in str(excinfo.value.body)


@pytest.mark.asyncio
async def test_fetch_task_description_sanitizes_api_error_before_reraising(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A FlowMesh rejection on task-description retrieval can echo back the
    submitted task spec, Authorization header included. fetch_task_description
    must re-raise a sanitized APIError so the credential never reaches the
    caller, the log, or the persisted job error record."""

    class _FakeTasks:
        async def retrieve(self, task_id: str) -> Any:
            raise APIError(
                "task spec invalid",
                status_code=422,
                method="GET",
                url="https://flowmesh.internal/tasks/t1",
                body={"spec": {"Authorization": "Bearer sk-live-secret"}},
            )

    class _FakeFm:
        def __init__(self) -> None:
            self.tasks = _FakeTasks()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

    with pytest.raises(APIError) as excinfo:
        await flowmesh_manager.fetch_task_description("t1")

    assert "sk-live-secret" not in str(excinfo.value)
    assert "sk-live-secret" not in str(excinfo.value.body)


@pytest.mark.asyncio
async def test_archive_task_response_sanitizes_api_error_before_reraising(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A FlowMesh rejection while archiving a task response can echo back the
    submitted task spec, Authorization header included. _archive_task_response
    must re-raise a sanitized APIError so the credential never reaches the
    caller, the log, or the persisted job error record."""
    from lumilake_server.runtime.request import RequestInfo

    class _FakeResults:
        async def retrieve(self, task_id: str) -> Any:
            raise APIError(
                "task spec invalid",
                status_code=422,
                method="GET",
                url="https://flowmesh.internal/tasks/t1",
                body={"spec": {"Authorization": "Bearer sk-live-secret"}},
            )

    class _FakeFm:
        def __init__(self) -> None:
            self.results = _FakeResults()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

    request_info = RequestInfo(
        request_id="req-1", runtime_graphs={}, data_profile_graphs={}
    )
    request_info.batch_id = "batch-1"

    with pytest.raises(APIError) as excinfo:
        await flowmesh_manager._archive_task_response(request_info, "task-1", "node-a")

    assert "sk-live-secret" not in str(excinfo.value)
    assert "sk-live-secret" not in str(excinfo.value.body)


def _build_single_row_api_request() -> tuple[RequestInfo, str]:
    stock = input_placeholder("Stock")
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct", api=ApiConfig()
        ),
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    (row_id,) = runtime_graph.dsl_to_runtime[llm.id]

    request_info = RequestInfo(
        request_id="req-history",
        runtime_graphs={"g": runtime_graph},
        data_profile_graphs={},
    )
    request_info.batch_id = "batch-1"
    request_info.runtime_graph = runtime_graph
    request_info.data_profile_graph = RuntimeGraph(
        nodes={}, node_order=[], output_node_map={}
    )
    return request_info, row_id


@pytest.mark.asyncio
async def test_output_result_retrieval_sanitizes_api_error_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A FlowMesh rejection while retrieving an output node's result can echo
    back the submitted task spec, Authorization header included. process_request
    must re-raise a sanitized APIError so the credential never reaches the
    caller, the log, or the persisted job error record."""
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "test-pat")
    manager = FlowmeshRuntimeManager()
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.base.get_job_storage",
        lambda: InMemoryJobStorage(),
    )

    request_info, row_id = _build_single_row_api_request()

    class _FakeWorkflows:
        async def submit(self, task_yaml: str) -> Any:
            return SimpleNamespace(
                tasks=[SimpleNamespace(task_id="task-row0")], workflow_id="wf-1"
            )

    class _FakeResults:
        def __init__(self) -> None:
            self.calls = 0

        async def retrieve(self, task_id: str) -> Any:
            self.calls += 1
            if self.calls == 1:
                return {"text": "assistant reply"}
            raise APIError(
                "task spec invalid",
                status_code=422,
                method="GET",
                url="https://flowmesh.internal/tasks/task-row0",
                body={"spec": {"Authorization": "Bearer sk-live-secret"}},
            )

    class _FakeFm:
        def __init__(self) -> None:
            self.workflows = _FakeWorkflows()
            self.results = _FakeResults()

    fake_fm = _FakeFm()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: fake_fm))

    async def _fetch_task_status(_self: FlowmeshRuntimeManager, task_id: str) -> str:
        return "DONE"

    async def _fetch_task_description(
        _self: FlowmeshRuntimeManager, task_id: str
    ) -> dict[str, Any]:
        return {"graph_node_name": row_id}

    monkeypatch.setattr(
        manager, "fetch_task_status", types.MethodType(_fetch_task_status, manager)
    )
    monkeypatch.setattr(
        manager,
        "fetch_task_description",
        types.MethodType(_fetch_task_description, manager),
    )

    with pytest.raises(APIError) as excinfo:
        await manager.process_request(
            request_info,
            Schedule(worker_assignment={"worker-1": [row_id]}),
            worker_ids=["worker-1"],
        )

    assert "sk-live-secret" not in str(excinfo.value)
    assert "sk-live-secret" not in str(excinfo.value.body)


@pytest.mark.asyncio
async def test_workflow_submit_sanitizes_api_error_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A FlowMesh rejection on workflow submission can echo back the submitted
    task spec, Authorization header included. process_request must re-raise a
    sanitized APIError so the credential never reaches the caller, the log, or
    the persisted job error record."""
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "test-pat")
    manager = FlowmeshRuntimeManager()
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.base.get_job_storage",
        lambda: InMemoryJobStorage(),
    )

    request_info, row_id = _build_single_row_api_request()

    class _FakeWorkflows:
        async def submit(self, task_yaml: str) -> Any:
            raise APIError(
                "task spec invalid",
                status_code=422,
                method="POST",
                url="https://flowmesh.internal/workflows",
                body={"spec": {"Authorization": "Bearer sk-live-secret"}},
            )

    class _FakeFm:
        def __init__(self) -> None:
            self.workflows = _FakeWorkflows()

    fake_fm = _FakeFm()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: fake_fm))

    with pytest.raises(APIError) as excinfo:
        await manager.process_request(
            request_info,
            Schedule(worker_assignment={"worker-1": [row_id]}),
            worker_ids=["worker-1"],
        )

    assert "sk-live-secret" not in str(excinfo.value)
    assert "sk-live-secret" not in str(excinfo.value.body)
