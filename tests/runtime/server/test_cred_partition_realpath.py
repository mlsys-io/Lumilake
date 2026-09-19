import pytest
from support.runtime_graphs import build_dummy_runtime_graph
from support.runtime_server import make_server

from lumilake_server.runtime.job_manager.priority_queue import PriorityJobManager
from lumilake_server.runtime.protocol import LumilakeRequestConfig
from lumilake_server.runtime.request import RequestHandler, WorkflowSliceMeta


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
    from lumilake_server.common import GenerationConfig
    from lumilake_server.graphs import Graph
    from lumilake_server.ops import LLMChatOp, OpMessage, as_output, input_placeholder

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
