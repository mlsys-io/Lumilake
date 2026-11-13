"""A FlowMesh server without python-task blobs rejects the whole task spec.

The rejection names only the unexpected field, so the submit error must also
name the FlowMesh release that supports python-task blobs — but only when the
spec actually mounts blobs and the rejection is about that field.
"""

from typing import Any

import pytest
from flowmesh.exceptions import ValidationError

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

_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _shout(items: tuple[Any, ...]) -> str:
    return str(items[0]).upper()


def _shout_output() -> OutputOp:
    return as_output("out", LambdaOp([input_placeholder("Stock")], _shout))


def _llm_output() -> OutputOp:
    return as_output(
        "out",
        LLMChatOp(
            [OpMessage(role="user", content=input_placeholder("Stock"))],
            config=GenerationConfig(model=_MODEL),
        ),
    )


def _rejection(status_code: int, field: str, detail: str) -> ValidationError:
    return ValidationError(
        f"Invalid task payload for node 'out': 1 validation error for"
        f" PythonSpecStrict\n{field}\n  {detail}",
        status_code=status_code,
        method="POST",
        url="https://flowmesh.internal/api/v1/workflows",
        body={"detail": f"{field}: {detail}"},
    )


@pytest.mark.parametrize(
    ("output", "rejection", "named"),
    [
        (
            _shout_output,
            _rejection(400, "spec.blobs", "Extra inputs are not permitted"),
            True,
        ),
        (
            _llm_output,
            _rejection(400, "spec.blobs", "Extra inputs are not permitted"),
            False,
        ),
        (_shout_output, _rejection(422, "spec.name", "Field required"), False),
    ],
    ids=["python-step", "no-python-step", "other-field"],
)
@pytest.mark.asyncio
async def test_submit_error_names_the_python_blobs_requirement_only_for_blobs(
    monkeypatch: pytest.MonkeyPatch,
    output: Any,
    rejection: ValidationError,
    named: bool,
) -> None:
    compiled = Graph.from_ops([output()]).compile(Stock=["NVDA"])
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

    class _FakeWorkflows:
        async def submit(self, task_yaml: str) -> Any:
            raise rejection

    class _FakeFm:
        def __init__(self) -> None:
            self.workflows = _FakeWorkflows()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

    with pytest.raises(ValidationError) as excinfo:
        await FlowmeshRuntimeManager().process_request(
            request_info,
            Schedule(worker_assignment={"worker-1": runtime_graph.node_order}),
            worker_ids=["worker-1"],
        )

    assert ("FlowMesh #259" in str(excinfo.value)) is named
    assert str(excinfo.value).count("returned 400") == (rejection.status_code == 400)
