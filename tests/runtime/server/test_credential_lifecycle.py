"""API credential lifecycle across the server: the digest that splits
partitions, the preview path that carries it, the batch path that resolves
credentials through originating job ids, and the release path that clears the
in-process store.
"""

import time
from typing import Any, cast

import pytest
from lumilake import envs
from support.runtime_graphs import build_dummy_runtime_graph
from support.runtime_server import (
    RecordingRuntimeManager,
    attach_request_states,
    make_batch,
    make_server,
)

from lumilake_server.common import GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import LLMChatOp, OpMessage, as_output, input_placeholder
from lumilake_server.runtime.job_manager.base import WorkflowItem
from lumilake_server.runtime.job_manager.priority_queue import PriorityJobManager
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.protocol import LumilakeRequestConfig
from lumilake_server.runtime.request import (
    RequestHandler,
    RequestInfo,
    WorkflowSliceMeta,
)


def _handler(request_id, principal_id):
    g = build_dummy_runtime_graph("g")
    sm = WorkflowSliceMeta(
        public_graph_name="g",
        slice_index=0,
        slice_start=0,
        slice_length=1,
        total_length=1,
        template_hash="h",
        varying_input_keys=(),
    )
    return RequestHandler(
        query={"g": g},
        data_profile_graphs={"g": g},
        dsl_graphs={"g": object()},
        workflow_slices={"g": sm},
        request_id=request_id,
        config=LumilakeRequestConfig(user_id=request_id, principal_id=principal_id),
    )


@pytest.mark.asyncio
async def test_real_path_different_credentials_split_partitions():
    server = make_server()
    server.runtime_manager.set_api_credential("req-a", "Bearer key-a")
    await server.handle_request(_handler("req-a", "p"), None)
    server.runtime_manager.set_api_credential("req-b", "Bearer key-b")
    await server.handle_request(_handler("req-b", "p"), None)
    parts = server.job_manager._rr_partition_members
    assert len(parts) == 2
    digests = {p[2] for p in parts}
    assert len(digests) == 2
    assert None not in digests


@pytest.mark.asyncio
async def test_real_path_same_credential_one_partition():
    server = make_server()
    server.runtime_manager.set_api_credential("req-a", "Bearer key-same")
    await server.handle_request(_handler("req-a", "p"), None)
    server.runtime_manager.set_api_credential("req-b", "Bearer key-same")
    await server.handle_request(_handler("req-b", "p"), None)
    parts = server.job_manager._rr_partition_members
    assert len(parts) == 1
    digests = {p[2] for p in parts}
    assert len(digests) == 1
    assert None not in digests


def _tiny_preview_graphs():
    stock = input_placeholder("Stock")
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    return {"preview_graph": compiled}


@pytest.mark.asyncio
async def test_preview_path_carries_api_credential_digest(
    monkeypatch: pytest.MonkeyPatch,
):
    server = make_server()
    server.runtime_manager.set_api_credential("preview-1", "Bearer preview-key")
    captured: list[str | None] = []

    async def _capture_enqueue(self, job):
        captured.append(job.api_credential_digest)
        return []

    monkeypatch.setattr(PriorityJobManager, "enqueue", _capture_enqueue)

    async def _stub_select_workers(*a, **k):
        return ["gpu-worker"], {"gpu-worker": {"has_gpu": True}}

    monkeypatch.setattr(
        server, "_select_preview_workers_and_profiles", _stub_select_workers
    )
    await server.preview_schedule(
        graphs=_tiny_preview_graphs(),
        request_id="preview-1",
        data_profile_results={},
    )
    assert len(captured) == 1
    assert captured[0] is not None
    assert captured[0] != "Bearer preview-key"


def test_release_request_workflows_clears_dispatch_token() -> None:
    """The secret-lifetime guarantee: once a request's workflows are released
    (the completion path), the dispatch token and API credential must be gone
    from the in-process store. A credential that outlives its request is the
    bug this mechanism exists to remove."""
    server = make_server()
    server.runtime_manager.set_dispatch_token("req-1", "runtime-token")
    server.runtime_manager.set_api_credential("req-1", "Bearer caller-key")

    assert server.runtime_manager.get_dispatch_token("req-1") == "runtime-token"
    assert server.runtime_manager.get_api_credential("req-1") == "Bearer caller-key"

    server.release_request_workflows("req-1")

    assert server.runtime_manager.get_dispatch_token("req-1") is None
    assert server.runtime_manager.get_api_credential("req-1") is None


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
