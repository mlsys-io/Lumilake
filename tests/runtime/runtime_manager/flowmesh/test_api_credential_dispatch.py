import types
from types import SimpleNamespace
from typing import Any

import pytest
from lumilake import envs

from lumilake_server.common import ApiConfig, GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import LLMChatOp, OpMessage, as_output, input_placeholder
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.request import RequestInfo
from lumilake_server.runtime.runtime_graph import (
    _API_CREDENTIAL_PLACEHOLDER,
    RuntimeGraph,
    RuntimeGraphBuilder,
)
from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager
from lumilake_server.utils.job_storage import InMemoryJobStorage

_UNTRUSTED_URL = "https://api.example.com/v1/chat/completions"


def _build_api_request(url: str) -> tuple[RequestInfo, str]:
    stock = input_placeholder("Stock")
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(url=url, authorization="Bearer build-only"),
        ),
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    (row_id,) = runtime_graph.dsl_to_runtime[llm.id]

    request_info = RequestInfo(
        request_id="req-cred",
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
async def test_dispatched_request_carries_resolved_caller_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The placeholder in the graph must be replaced with the real caller
    credential before the task spec is submitted to FlowMesh. If the
    substitution point breaks, the dispatched Authorization header would be
    the literal placeholder."""
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "test-pat")
    manager = FlowmeshRuntimeManager()
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.base.get_job_storage",
        lambda: InMemoryJobStorage(),
    )

    request_info, row_id = _build_api_request(_UNTRUSTED_URL)
    manager.set_api_credential("req-cred", "Bearer caller-key")

    submitted: dict[str, Any] = {}

    class _FakeWorkflows:
        async def submit(self, task_yaml: str) -> Any:
            submitted["yaml"] = task_yaml
            return SimpleNamespace(
                tasks=[SimpleNamespace(task_id="task-1")], workflow_id="wf-1"
            )

    class _FakeResults:
        async def retrieve(self, task_id: str) -> dict[str, Any]:
            return {"text": "assistant reply"}

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

    await manager.process_request(
        request_info,
        Schedule(worker_assignment={"worker-1": [row_id]}),
        worker_ids=["worker-1"],
    )

    import yaml

    task_spec = yaml.safe_load(submitted["yaml"])
    node = task_spec["spec"]["graph"]["nodes"][0]
    assert node["spec"]["api"]["headers"]["Authorization"] == "Bearer caller-key"
    assert _API_CREDENTIAL_PLACEHOLDER not in submitted["yaml"]
