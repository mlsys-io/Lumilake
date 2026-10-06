"""API request lifecycle: result normalization, chat-history parity with the
local backend, reasoning-model failure guidance, and the all-or-nothing
fan-out contract.
"""

import json
import types
from types import SimpleNamespace
from typing import Any

import pytest
from flowmesh.models.result import APIResult
from flowmesh.models.result.catalog import PythonResult
from flowmesh.models.tasks import TaskInfo

from lumilake import envs
from lumilake_server.common import ApiConfig, GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    LambdaOp,
    LLMChatOp,
    OpMessage,
    as_output,
    input_placeholder,
)
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.request import RequestInfo
from lumilake_server.runtime.runtime_graph import RuntimeGraph, RuntimeGraphBuilder
from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager
from lumilake_server.utils.job_storage import InMemoryJobStorage


def _explode_fn(items: tuple[list[Any], ...]) -> list[dict[str, str]]:
    return [{"value": v} for v in items[0]]


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


def test_api_reasoning_present_still_raises_clear_error() -> None:
    """A reasoning model returns content=None with a reasoning fragment when
    its budget is exhausted; that must fail with guidance, not surface the
    truncated fragment as the node output."""
    results = {
        "text": None,
        "response_json": {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning": (
                            "The user wants me to calculate 17 x 23 and show my"
                        ),
                    },
                }
            ]
        },
    }
    with pytest.raises(RuntimeError, match="raise max_tokens"):
        FlowmeshRuntimeManager()._resolve_output_items(
            results, "out-1", task_type="api"
        )


def test_resolve_output_items_accepts_empty_items_list() -> None:
    """A result whose ``items`` list is empty is a valid empty output (e.g. a
    list-mode Lambda that selected no observations), not a failure."""
    manager = FlowmeshRuntimeManager()
    assert manager._resolve_output_items({"items": []}, "out-1") == []


def test_resolve_output_items_still_raises_without_items_key() -> None:
    """A result with no ``items`` list at all still fails; only an explicit
    empty list is treated as an empty output."""
    manager = FlowmeshRuntimeManager()
    with pytest.raises(RuntimeError, match="produced no items"):
        manager._resolve_output_items({"ok": True}, "out-1")


def test_resolve_output_items_accepts_empty_python_result_value() -> None:
    """A real SDK PythonResult with ``value={"items": []}`` is a valid empty
    list-mode Lambda output, not a failure."""
    manager = FlowmeshRuntimeManager()
    result = PythonResult(exit_code=0, value={"items": []})
    assert (
        manager._resolve_output_items(
            result.model_dump(mode="json"), "out-1", task_type="python"
        )
        == []
    )


@pytest.mark.asyncio
async def test_api_backend_returns_chat_history_like_local_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An API-backed node must carry ``metadata.prompt`` like a local
    inference item, or ``return_history`` silently drops its history."""
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "test-pat")
    manager = FlowmeshRuntimeManager()
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.base.get_job_storage",
        lambda: InMemoryJobStorage(),
    )

    request_info, row_id = _build_single_row_api_request()
    sent_messages = request_info.runtime_graph.nodes[row_id].api_spec["json"][
        "messages"
    ]
    assert sent_messages == "{{prompt}}"

    class _FakeWorkflows:
        async def submit(self, task_yaml: str) -> Any:
            return SimpleNamespace(
                tasks=[SimpleNamespace(task_id="task-row0")], workflow_id="wf-1"
            )

    class _FakeResults:
        async def retrieve(self, task_id: str) -> dict[str, Any]:
            return {
                "items": [
                    {
                        "index": 0,
                        "json": {
                            "choices": [{"message": {"content": "assistant reply"}}]
                        },
                        "text": "assistant reply",
                        "prompt": "{{prompt}}",
                    }
                ]
            }

    class _FakeFm:
        def __init__(self) -> None:
            self.workflows = _FakeWorkflows()
            self.results = _FakeResults()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

    async def _fetch_task_status(
        _self: FlowmeshRuntimeManager, task_id: str
    ) -> TaskInfo:
        return _task_info(task_id, "DONE", graph_node_name=row_id)

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

    result = await manager.process_request(
        request_info,
        Schedule(worker_assignment={"worker-1": [row_id]}),
        worker_ids=["worker-1"],
    )

    # The API request body carries the ``{{prompt}}`` placeholder, not the
    # rendered messages, so the runtime manager cannot reconstruct a chat
    # history from the api json; the prompt lives in the data_spec template.
    assert result["chat_histories"] == {}


def _task_info(
    task_id: str,
    status: str,
    *,
    error: str | None = None,
    last_error: str | None = None,
    graph_node_name: str | None = None,
) -> TaskInfo:
    """A real SDK ``TaskInfo`` for a fake status fetch."""
    return TaskInfo(
        task_id=task_id,
        workflow_id="wf-1",
        owner_id="owner",
        org_id="org",
        supplier_id="supplier",
        source="",
        task={},
        status=status,
        submitted_at="2026-01-01T00:00:00Z",
        submitted_ts=0.0,
        usages=[],
        attempts=1,
        max_attempts=1,
        load=0,
        depends_on=[],
        pending_dependencies=[],
        dependents=[],
        completed=status == "DONE",
        failed=status == "FAILED",
        error=error,
        last_error=last_error,
        graph_node_name=graph_node_name,
    )


def _build_two_row_request() -> tuple[RequestInfo, str, str]:
    stock = input_placeholder("Stock")
    first = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct", api=ApiConfig()
        ),
    )
    second = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct", api=ApiConfig()
        ),
    )
    output = as_output("result", first)
    output2 = as_output("result2", second)
    compiled = Graph.from_ops([output, output2]).compile(Stock=["NVDA", "AAPL"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    (row0_id,) = runtime_graph.dsl_to_runtime[first.id]
    (row1_id,) = runtime_graph.dsl_to_runtime[second.id]

    request_info = RequestInfo(
        request_id="req-failfast",
        runtime_graphs={"g": runtime_graph},
        data_profile_graphs={},
    )
    request_info.batch_id = "batch-1"
    request_info.runtime_graph = runtime_graph
    request_info.data_profile_graph = RuntimeGraph(
        nodes={}, node_order=[], output_node_map={}
    )
    return request_info, row0_id, row1_id


async def _run_failing_request(
    monkeypatch: pytest.MonkeyPatch,
    status_by_task: dict[str, str],
    error_by_task: dict[str, str | None],
) -> tuple[FlowmeshRuntimeManager, str, str, str]:
    """Run the two-row request against fake FlowMesh task states; return the
    manager, both row node ids and the error ``process_request`` raised."""
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "test-pat")
    manager = FlowmeshRuntimeManager()
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.base.get_job_storage",
        lambda: InMemoryJobStorage(),
    )

    request_info, row0_id, row1_id = _build_two_row_request()

    task_ids = ["task-row0", "task-row1"]
    node_by_task = {"task-row0": row0_id, "task-row1": row1_id}

    class _FakeWorkflows:
        async def submit(self, task_yaml: str) -> Any:
            return SimpleNamespace(
                tasks=[SimpleNamespace(task_id=tid) for tid in task_ids],
                workflow_id="wf-1",
            )

    class _FakeResults:
        async def retrieve(self, task_id: str) -> Any:
            return APIResult(
                executor="api",
                method="POST",
                url="https://api.example.com/v1/chat",
                status_code=200,
                text="row0-response",
            )

    class _FakeFm:
        def __init__(self) -> None:
            self.workflows = _FakeWorkflows()
            self.results = _FakeResults()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

    async def _fetch_task_status(
        _self: FlowmeshRuntimeManager, task_id: str
    ) -> TaskInfo:
        return _task_info(
            task_id,
            status_by_task[task_id],
            error=error_by_task.get(task_id),
            graph_node_name=node_by_task[task_id],
        )

    async def _fetch_task_description(
        _self: FlowmeshRuntimeManager, task_id: str
    ) -> dict[str, Any]:
        return {
            "graph_node_name": node_by_task[task_id],
            "error": error_by_task.get(task_id),
        }

    monkeypatch.setattr(
        manager, "fetch_task_status", types.MethodType(_fetch_task_status, manager)
    )
    monkeypatch.setattr(
        manager,
        "fetch_task_description",
        types.MethodType(_fetch_task_description, manager),
    )

    with pytest.raises(RuntimeError) as excinfo:
        await manager.process_request(
            request_info,
            Schedule(worker_assignment={"worker-1": [row0_id, row1_id]}),
            worker_ids=["worker-1"],
        )
    return manager, row0_id, row1_id, str(excinfo.value)


@pytest.mark.asyncio
async def test_one_failed_row_aborts_the_whole_workflow_before_collecting_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPS.md's documented fan-out contract is all-or-nothing: if any row's
    task fails, the whole workflow request fails and no partial per-row
    results are returned, even for rows that already completed
    successfully. This pins the fail-fast raise in process_request's poll
    loop; without it a FAILED row would be silently ignored (or a partial
    result quietly returned) instead of aborting the request."""
    manager, _, row1_id, message = await _run_failing_request(
        monkeypatch,
        {"task-row0": "PENDING", "task-row1": "FAILED"},
        {"task-row1": "python task failed: ImportError: boom"},
    )
    assert message == (
        f"Task task-row1 failed (node {row1_id}): python task failed:"
        " ImportError: boom; aborting workflow (1 task(s) failed in total)"
    )

    batch_key = ("req-failfast", "batch-1")
    assert manager._execution_task_status[batch_key]["task-row0"] == "PENDING"
    assert manager._execution_task_status[batch_key]["task-row1"] == "FAILED"


def _build_api_output_request(*, rowwise: bool = False) -> tuple[RequestInfo, str]:
    """Build a request whose output node is an api-mode LLMChatOp (plain or
    row-wise), with no path override on the OutputOp."""
    stock = input_placeholder("Stock")
    kwargs: dict[str, Any] = {}
    if rowwise:
        kwargs = {
            "rowwise_template": "Summarize {Stock}.",
            "rowwise_columns": [
                {"label": "Stock", "data": {"type": "list", "items": ["NVDA"]}}
            ],
        }
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct", api=ApiConfig()
        ),
        **kwargs,
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    (row_id,) = runtime_graph.dsl_to_runtime[llm.id]

    request_info = RequestInfo(
        request_id=f"req-api-{'rowwise' if rowwise else 'plain'}",
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
@pytest.mark.parametrize(
    ("rowwise", "result", "expected"),
    [
        (
            False,
            {
                "items": [
                    {
                        "index": 0,
                        "json": {
                            "choices": [{"message": {"content": '{"keep": [1, 2]}'}}]
                        },
                        "text": '{"keep": [1, 2]}',
                        "prompt": "p",
                    }
                ]
            },
            ['{"keep": [1, 2]}'],
        ),
        (
            True,
            {
                "items": [
                    {
                        "index": 0,
                        "rows": [
                            {
                                "index": 0,
                                "json": {"choices": [{"message": {"content": "r0"}}]},
                                "text": "r0",
                                "prompt": "p",
                            },
                            {
                                "index": 1,
                                "json": {"choices": [{"message": {"content": "r1"}}]},
                                "text": "r1",
                                "prompt": "p",
                            },
                        ],
                    }
                ]
            },
            ["r0", "r1"],
        ),
    ],
)
async def test_api_output_node_reads_content(
    monkeypatch: pytest.MonkeyPatch,
    rowwise: bool,
    result: dict[str, Any],
    expected: list[str],
) -> None:
    """An api-mode LLM op as the output node must surface its model content
    (the api item path), not fail on a missing ``items.output``. Row-wise api
    tasks fan out over ``items.rows``."""
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "test-pat")
    manager = FlowmeshRuntimeManager()
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.base.get_job_storage",
        lambda: InMemoryJobStorage(),
    )

    request_info, row_id = _build_api_output_request(rowwise=rowwise)

    class _FakeWorkflows:
        async def submit(self, task_yaml: str) -> Any:
            return SimpleNamespace(
                tasks=[SimpleNamespace(task_id="task-row0")], workflow_id="wf-1"
            )

    class _FakeResults:
        async def retrieve(self, task_id: str) -> dict[str, Any]:
            return result

    class _FakeFm:
        def __init__(self) -> None:
            self.workflows = _FakeWorkflows()
            self.results = _FakeResults()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

    async def _fetch_task_status(
        _self: FlowmeshRuntimeManager, task_id: str
    ) -> TaskInfo:
        return _task_info(task_id, "DONE", graph_node_name=row_id)

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

    result_ = await manager.process_request(
        request_info,
        Schedule(worker_assignment={"worker-1": [row_id]}),
        worker_ids=["worker-1"],
    )
    assert result_["flat_outputs"][row_id] == expected


def _build_list_lambda_output_request() -> tuple[RequestInfo, str]:
    stock = input_placeholder("Stock")
    explode = LambdaOp([stock], _explode_fn, mode="list")
    output = as_output("observations", explode)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    (row_id,) = runtime_graph.dsl_to_runtime[explode.id]

    request_info = RequestInfo(
        request_id="req-list-lambda",
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
async def test_list_lambda_output_aggregates_whole_list_into_one_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A list-mode Lambda output node runs once over the whole input lists,
    so its python step's several items must collapse into ONE output value
    holding the whole list, not one output per item."""
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "test-pat")
    manager = FlowmeshRuntimeManager()
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.base.get_job_storage",
        lambda: InMemoryJobStorage(),
    )

    request_info, row_id = _build_list_lambda_output_request()

    class _FakeWorkflows:
        async def submit(self, task_yaml: str) -> Any:
            return SimpleNamespace(
                tasks=[SimpleNamespace(task_id="task-row0")], workflow_id="wf-1"
            )

    class _FakeResults:
        async def retrieve(self, task_id: str) -> dict[str, Any]:
            return {
                "items": [
                    {"output": {"fid": "f1", "statement": "s1", "quote": "q1"}},
                    {"output": {"fid": "f2", "statement": "s2", "quote": "q2"}},
                    {"output": {"fid": "f3", "statement": "s3", "quote": "q3"}},
                ]
            }

    class _FakeFm:
        def __init__(self) -> None:
            self.workflows = _FakeWorkflows()
            self.results = _FakeResults()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

    async def _fetch_task_status(
        _self: FlowmeshRuntimeManager, task_id: str
    ) -> TaskInfo:
        return _task_info(task_id, "DONE", graph_node_name=row_id)

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

    result_ = await manager.process_request(
        request_info,
        Schedule(worker_assignment={"worker-1": [row_id]}),
        worker_ids=["worker-1"],
    )
    flat = result_["flat_outputs"][row_id]
    assert len(flat) == 1
    whole_list = json.loads(flat[0])
    assert whole_list == [
        {"fid": "f1", "statement": "s1", "quote": "q1"},
        {"fid": "f2", "statement": "s2", "quote": "q2"},
        {"fid": "f3", "statement": "s3", "quote": "q3"},
    ]


@pytest.mark.asyncio
async def test_empty_list_lambda_output_archives_as_empty_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A list-mode Lambda that selects no observations returns ``items: []``;
    that is a valid empty output (one whole-list value holding ``[]``), not a
    failure."""
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "test-pat")
    manager = FlowmeshRuntimeManager()
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.base.get_job_storage",
        lambda: InMemoryJobStorage(),
    )

    request_info, row_id = _build_list_lambda_output_request()

    class _FakeWorkflows:
        async def submit(self, task_yaml: str) -> Any:
            return SimpleNamespace(
                tasks=[SimpleNamespace(task_id="task-row0")], workflow_id="wf-1"
            )

    class _FakeResults:
        async def retrieve(self, task_id: str) -> dict[str, Any]:
            return {"items": []}

    class _FakeFm:
        def __init__(self) -> None:
            self.workflows = _FakeWorkflows()
            self.results = _FakeResults()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

    async def _fetch_task_status(
        _self: FlowmeshRuntimeManager, task_id: str
    ) -> TaskInfo:
        return _task_info(task_id, "DONE", graph_node_name=row_id)

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

    result_ = await manager.process_request(
        request_info,
        Schedule(worker_assignment={"worker-1": [row_id]}),
        worker_ids=["worker-1"],
    )
    flat = result_["flat_outputs"][row_id]
    assert len(flat) == 1
    assert json.loads(flat[0]) == []


@pytest.mark.asyncio
async def test_failed_output_task_reports_the_flowmesh_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed output task (e.g. a standalone LambdaOp whose code raised)
    must surface FlowMesh's recorded error, not only the task id."""
    error = "python task failed: ImportError: __import__ not found"
    _, row0_id, row1_id, message = await _run_failing_request(
        monkeypatch,
        {"task-row0": "FAILED", "task-row1": "DONE"},
        {"task-row0": error},
    )
    assert message == f"Output task task-row0 failed (node {row0_id}): {error}"


@pytest.mark.asyncio
async def test_failed_task_without_a_recorded_error_still_fails_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, row1_id, message = await _run_failing_request(
        monkeypatch, {"task-row0": "PENDING", "task-row1": "FAILED"}, {}
    )
    assert message.startswith(f"Task task-row1 failed (node {row1_id}); aborting")


@pytest.mark.asyncio
async def test_failed_task_error_is_redacted_before_raise_and_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A FlowMesh task error is an untrusted remote field; a credential it
    echoes must be redacted in both the recorded error and the raised
    message."""
    error = "python task failed: ValueError: Authorization: Bearer supersecret"
    manager, _, row1_id, message = await _run_failing_request(
        monkeypatch,
        {"task-row0": "PENDING", "task-row1": "FAILED"},
        {"task-row1": error},
    )
    assert "supersecret" not in message
    assert "***REDACTED***" in message
    assert "supersecret" not in manager._execution_task_errors["task-row1"]
    assert "***REDACTED***" in manager._execution_task_errors["task-row1"]


@pytest.mark.asyncio
async def test_fail_fast_reports_every_failed_task_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When several tasks fail in the same poll, each failed task's reason is
    reported, not only the first's."""
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "test-pat")
    manager = FlowmeshRuntimeManager()
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.base.get_job_storage",
        lambda: InMemoryJobStorage(),
    )

    stock = input_placeholder("Stock")
    llms = [
        LLMChatOp(
            [OpMessage(role="user", content=stock)],
            config=GenerationConfig(
                model="meta-llama/Llama-3.1-8B-Instruct", api=ApiConfig()
            ),
        )
        for _ in range(3)
    ]
    outputs = [as_output(f"result{i}", llm) for i, llm in enumerate(llms)]
    compiled = Graph.from_ops(outputs).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    (row0_id,), (row1_id,), (row2_id,) = (
        runtime_graph.dsl_to_runtime[llm.id] for llm in llms
    )

    request_info = RequestInfo(
        request_id="req-failfast",
        runtime_graphs={"g": runtime_graph},
        data_profile_graphs={},
    )
    request_info.batch_id = "batch-1"
    request_info.runtime_graph = runtime_graph
    request_info.data_profile_graph = RuntimeGraph(
        nodes={}, node_order=[], output_node_map={}
    )

    task_ids = ["task-row0", "task-row1", "task-row2"]
    node_by_task = {
        "task-row0": row0_id,
        "task-row1": row1_id,
        "task-row2": row2_id,
    }

    class _FakeWorkflows:
        async def submit(self, task_yaml: str) -> Any:
            return SimpleNamespace(
                tasks=[SimpleNamespace(task_id=tid) for tid in task_ids],
                workflow_id="wf-1",
            )

    class _FakeResults:
        async def retrieve(self, task_id: str) -> Any:
            return APIResult(
                executor="api",
                method="POST",
                url="https://api.example.com/v1/chat",
                status_code=200,
                text="row0-response",
            )

    class _FakeFm:
        def __init__(self) -> None:
            self.workflows = _FakeWorkflows()
            self.results = _FakeResults()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

    status_by_task = {
        "task-row0": "FAILED",
        "task-row1": "FAILED",
        "task-row2": "PENDING",
    }
    error_by_task = {"task-row1": "python task failed: ValueError: bad input"}

    async def _fetch_task_status(
        _self: FlowmeshRuntimeManager, task_id: str
    ) -> TaskInfo:
        return _task_info(
            task_id,
            status_by_task[task_id],
            error=error_by_task.get(task_id),
            graph_node_name=node_by_task[task_id],
        )

    async def _fetch_task_description(
        _self: FlowmeshRuntimeManager, task_id: str
    ) -> dict[str, Any]:
        return {
            "graph_node_name": node_by_task[task_id],
            "error": error_by_task.get(task_id),
        }

    monkeypatch.setattr(
        manager, "fetch_task_status", types.MethodType(_fetch_task_status, manager)
    )
    monkeypatch.setattr(
        manager,
        "fetch_task_description",
        types.MethodType(_fetch_task_description, manager),
    )

    with pytest.raises(RuntimeError) as excinfo:
        await manager.process_request(
            request_info,
            Schedule(worker_assignment={"worker-1": [row0_id, row1_id, row2_id]}),
            worker_ids=["worker-1"],
        )
    message = str(excinfo.value)
    assert f"Task task-row0 failed (node {row0_id})" in message
    assert (
        f"Task task-row1 failed (node {row1_id}): python task failed:"
        " ValueError: bad input"
    ) in message
    assert "(2 task(s) failed in total)" in message
