"""A FlowMesh server without python-task blobs rejects the whole task spec.

The rejection names only the unexpected field, so the submit error also names
the FlowMesh release that supports python-task blobs — but only when the spec
mounts blobs and the rejection is about that field.
"""

from typing import Any

import pytest
from flowmesh.exceptions import ValidationError

from lumilake_server.graphs import Graph
from lumilake_server.ops import LambdaOp, as_output, input_placeholder
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.request import RequestInfo
from lumilake_server.runtime.runtime_graph import RuntimeGraph, RuntimeGraphBuilder
from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


def _shout(inputs: tuple[Any, ...]) -> str:
    return str(inputs[0]).upper()


@pytest.mark.parametrize(
    ("status_code", "field", "named"),
    [(400, "spec.blobs", True), (422, "spec.name", False)],
)
@pytest.mark.asyncio
async def test_submit_error_names_the_python_blobs_requirement_only_for_blobs(
    monkeypatch: pytest.MonkeyPatch, status_code: int, field: str, named: bool
) -> None:
    output = as_output("out", LambdaOp([input_placeholder("Stock")], _shout))
    runtime_graph = RuntimeGraphBuilder().build(
        Graph.from_ops([output]).compile(Stock=["NVDA"])
    )
    request_info = RequestInfo(
        request_id="req-blobs", runtime_graphs={}, data_profile_graphs={}
    )
    request_info.batch_id = "batch-1"
    request_info.runtime_graph = runtime_graph
    request_info.data_profile_graph = RuntimeGraph(
        nodes={}, node_order=[], output_node_map={}
    )
    rejection = ValidationError(
        f"node 'out': {field} is not allowed",
        status_code=status_code,
        method="POST",
        url="https://flowmesh.internal/api/v1/workflows",
        body={"detail": f"{field}: rejected"},
    )

    monkeypatch.setattr(
        FlowmeshRuntimeManager, "fm", property(lambda self: _Rejecting(rejection))
    )

    with pytest.raises(ValidationError) as excinfo:
        await FlowmeshRuntimeManager().process_request(
            request_info,
            Schedule(worker_assignment={"worker-1": runtime_graph.node_order}),
            worker_ids=["worker-1"],
        )

    assert ("FlowMesh #259" in str(excinfo.value)) is named


class _Rejecting:
    workflows = None

    def __init__(self, rejection: ValidationError) -> None:
        self._rejection = rejection
        self.workflows = self

    async def submit(self, task_yaml: str) -> Any:
        raise self._rejection
