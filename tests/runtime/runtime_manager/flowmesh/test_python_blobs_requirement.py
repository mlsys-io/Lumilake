"""A FlowMesh server without python-task blobs rejects the whole task spec.

The rejection names only the unexpected field, so the submit error must also
name the FlowMesh release that supports python-task blobs.
"""

from typing import Any

import pytest
from flowmesh.exceptions import ValidationError

from lumilake import envs
from lumilake_server.common import GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    LambdaOp,
    LLMChatOp,
    OpMessage,
    OutputOp,
    as_output,
    input_placeholder,
)
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.request import RequestInfo
from lumilake_server.runtime.runtime_graph import RuntimeGraph, RuntimeGraphBuilder
from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager
from lumilake_server.utils import job_storage as job_storage_module
from lumilake_server.utils.job_storage import InMemoryJobStorage

_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _shout(items: tuple[Any, ...]) -> str:
    return str(items[0]).upper()


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")


def _blobs_rejection() -> ValidationError:
    return ValidationError(
        "Invalid task payload for node 'out': 1 validation error for PythonSpecStrict"
        "\nspec.blobs\n  Extra inputs are not permitted",
        status_code=400,
        method="POST",
        url="https://flowmesh.internal/api/v1/workflows",
        body={"detail": "spec.blobs: Extra inputs are not permitted"},
    )


def _request(
    monkeypatch: pytest.MonkeyPatch, output: OutputOp, rejection: ValidationError
) -> tuple[RequestInfo, list[str]]:
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    request_info = RequestInfo(
        request_id="req-python-blobs",
        runtime_graphs={"g": runtime_graph},
        data_profile_graphs={},
    )
    request_info.batch_id = "batch-1"
    request_info.runtime_graph = runtime_graph
    request_info.data_profile_graph = RuntimeGraph(
        nodes={}, node_order=[], output_node_map={}
    )
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.get_job_storage",
        lambda: InMemoryJobStorage(),
    )
    monkeypatch.setattr(job_storage_module, "_job_storage", InMemoryJobStorage())

    class _FakeWorkflows:
        async def submit(self, task_yaml: str) -> Any:
            raise rejection

    class _FakeFm:
        def __init__(self) -> None:
            self.workflows = _FakeWorkflows()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))
    return request_info, list(runtime_graph.node_order)


@pytest.mark.asyncio
async def test_submit_error_names_the_python_blobs_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = as_output("out", LambdaOp([input_placeholder("Stock")], _shout))
    request_info, node_ids = _request(monkeypatch, output, _blobs_rejection())

    with pytest.raises(ValidationError) as excinfo:
        await FlowmeshRuntimeManager().process_request(
            request_info,
            Schedule(worker_assignment={"worker-1": node_ids}),
            worker_ids=["worker-1"],
        )

    message = str(excinfo.value)
    assert "spec.blobs" in message
    assert "FlowMesh #259" in message
    assert "v0.1.13" in message
    assert message.count("returned 400") == 1


@pytest.mark.asyncio
async def test_submit_error_without_python_steps_keeps_the_server_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = as_output(
        "out",
        LLMChatOp(
            [OpMessage(role="user", content=input_placeholder("Stock"))],
            config=GenerationConfig(model=_MODEL),
        ),
    )
    request_info, node_ids = _request(monkeypatch, output, _blobs_rejection())

    with pytest.raises(ValidationError) as excinfo:
        await FlowmeshRuntimeManager().process_request(
            request_info,
            Schedule(worker_assignment={"worker-1": node_ids}),
            worker_ids=["worker-1"],
        )

    assert "FlowMesh #259" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_submit_error_on_another_field_keeps_the_server_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = as_output("out", LambdaOp([input_placeholder("Stock")], _shout))
    rejection = ValidationError(
        "Invalid task payload for node 'out': 1 validation error for PythonSpecStrict"
        "\nspec.name\n  Field required",
        status_code=422,
        method="POST",
        url="https://flowmesh.internal/api/v1/workflows",
        body={"detail": "spec.name: Field required"},
    )
    request_info, node_ids = _request(monkeypatch, output, rejection)

    with pytest.raises(ValidationError) as excinfo:
        await FlowmeshRuntimeManager().process_request(
            request_info,
            Schedule(worker_assignment={"worker-1": node_ids}),
            worker_ids=["worker-1"],
        )

    assert "FlowMesh #259" not in str(excinfo.value)
