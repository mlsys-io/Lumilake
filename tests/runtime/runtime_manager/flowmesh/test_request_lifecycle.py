"""API request lifecycle: result normalization, chat-history parity with the
local backend, reasoning-model failure guidance, and the all-or-nothing
fan-out contract.
"""

import types
from types import SimpleNamespace
from typing import Any

import pytest
from flowmesh.models.result import APIResult
from lumilake import envs

from lumilake_server.common import ApiConfig, GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import LLMChatOp, OpMessage, as_output, input_placeholder
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.request import RequestInfo
from lumilake_server.runtime.runtime_graph import RuntimeGraph, RuntimeGraphBuilder
from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager
from lumilake_server.utils.job_storage import InMemoryJobStorage


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

    class _FakeWorkflows:
        async def submit(self, task_yaml: str) -> Any:
            return SimpleNamespace(
                tasks=[SimpleNamespace(task_id="task-row0")], workflow_id="wf-1"
            )

    class _FakeResults:
        async def retrieve(self, task_id: str) -> Any:
            return APIResult(
                executor="api",
                method="POST",
                url="https://api.example.com/v1/chat",
                status_code=200,
                text="assistant reply",
            )

    class _FakeFm:
        def __init__(self) -> None:
            self.workflows = _FakeWorkflows()
            self.results = _FakeResults()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

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

    result = await manager.process_request(
        request_info,
        Schedule(worker_assignment={"worker-1": [row_id]}),
        worker_ids=["worker-1"],
    )

    history = result["chat_histories"][row_id]
    assert history == [
        sent_messages + [{"role": "assistant", "content": "assistant reply"}]
    ]


def _build_two_row_request() -> tuple[RequestInfo, str, str]:
    stock = input_placeholder("Stock")
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct", api=ApiConfig()
        ),
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA", "AAPL"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    row0_id, row1_id = runtime_graph.dsl_to_runtime[llm.id]

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
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "test-pat")
    manager = FlowmeshRuntimeManager()
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.base.get_job_storage",
        lambda: InMemoryJobStorage(),
    )

    request_info, row0_id, row1_id = _build_two_row_request()

    task_ids = ["task-row0", "task-row1"]
    node_by_task = {"task-row0": row0_id, "task-row1": row1_id}
    status_by_task = {"task-row0": "PENDING", "task-row1": "FAILED"}

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

    async def _fetch_task_status(_self: FlowmeshRuntimeManager, task_id: str) -> str:
        return status_by_task[task_id]

    async def _fetch_task_description(
        _self: FlowmeshRuntimeManager, task_id: str
    ) -> dict[str, Any]:
        return {"graph_node_name": node_by_task[task_id]}

    monkeypatch.setattr(
        manager, "fetch_task_status", types.MethodType(_fetch_task_status, manager)
    )
    monkeypatch.setattr(
        manager,
        "fetch_task_description",
        types.MethodType(_fetch_task_description, manager),
    )

    with pytest.raises(RuntimeError, match="failed; aborting workflow"):
        await manager.process_request(
            request_info,
            Schedule(worker_assignment={"worker-1": [row0_id, row1_id]}),
            worker_ids=["worker-1"],
        )

    batch_key = ("req-failfast", "batch-1")
    assert manager._execution_task_status[batch_key]["task-row0"] == "PENDING"
    assert manager._execution_task_status[batch_key]["task-row1"] == "FAILED"
