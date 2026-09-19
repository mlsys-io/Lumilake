"""The batch path must fill ``RequestInfo.member_request_ids`` from the
originating ``req-*`` jobs, not the synthetic ``exec-*`` execution id. API
credentials are stored per job id, so dispatch resolves them through these
ids. A ``member_request_ids`` that is never populated (or that carries the
``exec-*`` id) would make credential resolution silently miss."""

import time
from typing import Any, cast

import pytest
from lumilake import envs
from support.runtime_server import (
    RecordingRuntimeManager,
    attach_request_states,
    make_batch,
)

from lumilake_server.common import GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import LLMChatOp, OpMessage, as_output, input_placeholder
from lumilake_server.runtime.job_manager.base import WorkflowItem
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.protocol import LumilakeRequestConfig
from lumilake_server.runtime.request import RequestInfo


def _compiled():
    stock = input_placeholder("Stock")
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    output = as_output("result", llm)
    return Graph.from_ops([output]).compile(Stock=["NVDA"])


def _workflow(workflow_id: str, request_id: str, graph_name: str) -> WorkflowItem:
    compiled = _compiled()
    runtime_graph = server_runtime_builder().build(compiled, node_prefix=graph_name)
    return WorkflowItem(
        workflow_id=workflow_id,
        request_id=request_id,
        graph_name=graph_name,
        public_graph_name="shared",
        slice_index=0,
        slice_start=0,
        slice_length=1,
        total_length=1,
        template_hash=f"template-{graph_name}",
        varying_input_keys=(),
        runtime_graph=runtime_graph,
        data_profile_graph=runtime_graph,
        dsl_graph=compiled,
        config=LumilakeRequestConfig(user_id=request_id, principal_id=request_id),
        enqueued_at=time.time(),
    )


def server_runtime_builder():
    from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder

    return RuntimeGraphBuilder()


@pytest.mark.asyncio
async def test_process_batch_fills_member_request_ids_from_originating_jobs(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = server_factory()
    runtime_manager = RecordingRuntimeManager()
    server.runtime_manager = cast(Any, runtime_manager)

    workflows = [
        _workflow("wf-a", "req-a", "ga"),
        _workflow("wf-b", "req-b", "gb"),
    ]
    attach_request_states(server, workflows)
    for request_id in ("req-a", "req-b"):
        server._requests[request_id].batch_node_counts["batch-1"] = {
            "raw": 1,
            "optimized": 0,
        }
    batch = make_batch(workflows)

    seen: dict[str, Any] = {}

    async def _fake_schedule(**kwargs: Any) -> Schedule:
        runtime_graph = kwargs["runtime_graph"]
        nodes = list(runtime_graph.nodes)
        return Schedule(worker_assignment={"worker-1": nodes})

    async def _fake_process_request(
        request_info: RequestInfo,
        schedule: Schedule,
        worker_ids: list[str],
        data_profile_results: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        seen["member_request_ids"] = set(request_info.member_request_ids)
        seen["request_id"] = request_info.request_id
        return {"flat_outputs": {}, "chat_histories": {}, "task_node_map": {}}

    monkeypatch.setattr(envs, "LUMILAKE_DISABLE_DATA_PROFILE", True)
    monkeypatch.setattr(server, "_generate_schedule_in_subprocess", _fake_schedule)
    monkeypatch.setattr(runtime_manager, "process_request", _fake_process_request)

    await server._process_batch(
        batch,
        "batch-1",
        ["worker-1"],
        {"worker-1": {"gpu": {"count": 0}}},
        execution_request_id="exec-1",
        member_request_ids={"req-a", "req-b"},
    )

    assert seen["member_request_ids"] == {"req-a", "req-b"}
    assert seen["request_id"] == "exec-1"
    assert "exec-1" not in seen["member_request_ids"]
