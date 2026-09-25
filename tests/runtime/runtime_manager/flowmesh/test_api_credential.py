"""API credential dispatch and resolution.

The graph carries a constant placeholder for the caller's API credential; it
is replaced with the real value at dispatch time. Trusted origins re-resolve
deterministically from server config; untrusted origins use the caller
credential from the dispatch-token store, keyed by the originating job id
(``member_request_ids``) rather than the synthetic ``exec-*`` request id.
"""

import types
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
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
from lumilake_server.runtime.runtime_manager.flowmesh import (
    FlowmeshRuntimeManager,
)
from lumilake_server.utils.job_storage import InMemoryJobStorage

_TRUSTED_URL = "https://lum.id/llm/v1/chat/completions"
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
        request_id="exec-cred",
        runtime_graphs={"g": runtime_graph},
        data_profile_graphs={},
        member_request_ids={"req-cred"},
    )
    request_info.batch_id = "batch-1"
    request_info.runtime_graph = runtime_graph
    request_info.data_profile_graph = RuntimeGraph(
        nodes={}, node_order=[], output_node_map={}
    )
    return request_info, row_id


def _task_spec(url: str) -> dict[str, Any]:
    return {
        "spec": {
            "graph": {
                "nodes": [
                    {
                        "spec": {
                            "api": {
                                "url": url,
                                "headers": {
                                    "Authorization": _API_CREDENTIAL_PLACEHOLDER
                                },
                            }
                        }
                    }
                ]
            }
        }
    }


@pytest.mark.asyncio
async def test_dispatched_request_carries_resolved_caller_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The placeholder in the graph must be replaced with the real caller
    credential before the task spec is submitted to FlowMesh. If the
    substitution point breaks, the dispatched Authorization header would be
    the literal placeholder. The request reproduces production's id flow: the
    credential is stored under the job id (req-*), while dispatch runs under a
    distinct synthetic exec-* id, so the two must be reconciled via
    member_request_ids."""
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

    task_spec = yaml.safe_load(submitted["yaml"])
    node = task_spec["spec"]["graph"]["nodes"][0]
    assert node["spec"]["api"]["headers"]["Authorization"] == "Bearer caller-key"
    assert _API_CREDENTIAL_PLACEHOLDER not in submitted["yaml"]


def test_resolve_trusted_origin_uses_server_pat(
    flowmesh_manager: FlowmeshRuntimeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trusted-origin PAT is a bare token; the serving endpoint requires an
    auth scheme, so it must be dispatched as ``Bearer <pat>``."""
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "server-pat")
    task_spec = _task_spec(_TRUSTED_URL)

    flowmesh_manager._resolve_api_credentials({"req-1"}, task_spec)

    node = task_spec["spec"]["graph"]["nodes"][0]
    assert node["spec"]["api"]["headers"]["Authorization"] == "Bearer server-pat"


def test_resolve_trusted_origin_bare_pat_gains_single_bearer_prefix(
    flowmesh_manager: FlowmeshRuntimeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trusted path yields a bare PAT (no scheme); it must gain exactly one
    ``Bearer `` prefix and never be double-prefixed."""
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "lm_pat_live_f")
    task_spec = _task_spec(_TRUSTED_URL)

    flowmesh_manager._resolve_api_credentials({"req-1"}, task_spec)

    node = task_spec["spec"]["graph"]["nodes"][0]
    assert node["spec"]["api"]["headers"]["Authorization"] == "Bearer lm_pat_live_f"


def test_resolve_untrusted_origin_schemed_credential_passes_through(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    """A caller-supplied credential that already carries an auth scheme must be
    left unchanged, not double-prefixed into ``Bearer Bearer ...``."""
    flowmesh_manager.set_api_credential("req-1", "Bearer caller-key")
    task_spec = _task_spec(_UNTRUSTED_URL)

    flowmesh_manager._resolve_api_credentials({"req-1"}, task_spec)

    node = task_spec["spec"]["graph"]["nodes"][0]
    assert node["spec"]["api"]["headers"]["Authorization"] == "Bearer caller-key"


def test_resolve_untrusted_origin_basic_scheme_passes_through(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    """A non-Bearer scheme (e.g. Basic) is also left unchanged."""
    flowmesh_manager.set_api_credential("req-1", "Basic dXNlcjpwYXNz")
    task_spec = _task_spec(_UNTRUSTED_URL)

    flowmesh_manager._resolve_api_credentials({"req-1"}, task_spec)

    node = task_spec["spec"]["graph"]["nodes"][0]
    assert node["spec"]["api"]["headers"]["Authorization"] == "Basic dXNlcjpwYXNz"


def test_resolve_untrusted_origin_uses_caller_credential(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    flowmesh_manager.set_api_credential("req-1", "Bearer caller-key")
    task_spec = _task_spec(_UNTRUSTED_URL)

    flowmesh_manager._resolve_api_credentials({"req-1"}, task_spec)

    node = task_spec["spec"]["graph"]["nodes"][0]
    assert node["spec"]["api"]["headers"]["Authorization"] == "Bearer caller-key"


def test_resolve_untrusted_origin_without_credential_leaves_placeholder(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    task_spec = _task_spec(_UNTRUSTED_URL)

    flowmesh_manager._resolve_api_credentials({"req-1"}, task_spec)

    node = task_spec["spec"]["graph"]["nodes"][0]
    assert (
        node["spec"]["api"]["headers"]["Authorization"] == _API_CREDENTIAL_PLACEHOLDER
    )


def test_resolve_skips_non_placeholder_headers(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    task_spec: dict[str, Any] = {
        "spec": {
            "graph": {
                "nodes": [
                    {
                        "spec": {
                            "api": {
                                "url": _UNTRUSTED_URL,
                                "headers": {"Authorization": "Bearer already-set"},
                            }
                        }
                    }
                ]
            }
        }
    }

    flowmesh_manager._resolve_api_credentials({"req-1"}, task_spec)

    node = task_spec["spec"]["graph"]["nodes"][0]
    assert node["spec"]["api"]["headers"]["Authorization"] == "Bearer already-set"
