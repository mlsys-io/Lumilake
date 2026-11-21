"""A row step below a list-mode Lambda keeps the fan-out's cardinality."""

from typing import Any, cast

import pytest
import yaml
from support.runtime_server import (
    RecordingRuntimeManager,
    attach_request_states,
    make_batch,
    make_workflow,
)

from lumilake import envs
from lumilake_server.graphs import CompiledGraph, Graph
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.runtime_graph import RuntimeGraph

_FANOUT = """
name: fanout
inputs: {Note: [one]}
ops:
- id: Seed
  op: DataOp
  data: ['{"u": "a"}', '{"u": "b"}', '{"u": "c"}']
- {id: Rows, op: LambdaOp, mode: row, inputs: [Seed], fn_name: ident,
   code: "def ident(row):\\n    return row[0]\\n"}
- {id: Samples, op: LambdaOp, mode: list, inputs: [Rows], fn_name: expand,
   code: "def expand(x):\\n    return [{'i': n} for n, _ in enumerate(x[0])]\\n"}
- {id: Done, op: LambdaOp, mode: row, inputs: [Samples], fn_name: done,
   code: "def done(row):\\n    return row[0]['i']\\n"}
outputs: [{name: rows, ref: Done}]
"""

_PLAIN_ROW = """
name: plain
inputs: {Note: [a, b]}
ops:
- {id: Tagged, op: LambdaOp, mode: row, inputs: [Note], fn_name: tag,
   code: "def tag(row):\\n    return 'tagged:' + row[0]\\n"}
outputs: [{name: out, ref: Tagged}]
"""


class _OutputValuesManager(RecordingRuntimeManager):
    """Fakes FlowMesh dispatch, returning preset values for every output node."""

    def __init__(self, values: list[str]) -> None:
        super().__init__()
        self._values = values

    async def process_request(
        self,
        request_info: Any,
        schedule: Schedule,
        worker_ids: list[str],
        data_profile_results: dict[str, list[dict[str, Any]]] | None,
    ) -> dict[str, Any]:
        flat_outputs = {
            node_id: list(self._values) for node_id in request_info.output_node_map
        }
        return {"flat_outputs": flat_outputs, "chat_histories": {}, "task_node_map": {}}


@pytest.mark.parametrize(
    "workflow, output_name, slice_length, values",
    [
        pytest.param(
            _FANOUT, "rows", 1, ["0", "1", "2"], id="list-lambda-fanout-row-step"
        ),
        pytest.param(
            _PLAIN_ROW, "out", 2, ["tagged:a", "tagged:b"], id="plain-row-cardinality"
        ),
    ],
)
@pytest.mark.asyncio
async def test_row_cardinality_below_a_list_lambda(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
    workflow: str,
    output_name: str,
    slice_length: int,
    values: list[str],
) -> None:
    """A row step keeps its input's cardinality: below a list Lambda's fan-out
    it emits one value per element, and only a step whose output count is the
    job's input row count is merged against the slice."""
    monkeypatch.setattr(envs, "LUMILAKE_DISABLE_DATA_PROFILE", True)
    spec = next(iter(parse_yaml_payload(yaml.safe_load(workflow)).values()))
    compiled: CompiledGraph = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    server = server_factory()
    server.runtime_manager = cast(Any, _OutputValuesManager(values))

    workflows = [
        make_workflow(
            workflow_id=f"wf-{output_name}",
            request_id=f"req-{output_name}",
            graph_name="ga",
            public_graph_name="shared",
            slice_length=slice_length,
            total_length=slice_length,
        )
    ]
    workflows[0].dsl_graph = compiled
    handlers = attach_request_states(server, workflows)
    batch = make_batch(workflows)

    monkeypatch.setattr(server, "_merge_group_compiled_graph", lambda items: compiled)

    async def _fake_schedule(**kwargs: Any) -> Schedule:
        runtime_graph: RuntimeGraph = kwargs["runtime_graph"]
        return Schedule(
            worker_assignment={
                kwargs["selected_workers"][0]: list(runtime_graph.node_order)
            }
        )

    monkeypatch.setattr(server, "_generate_schedule_in_subprocess", _fake_schedule)

    await server._run_batch(["worker-1"], batch)

    resp = handlers[f"req-{output_name}"].results[0]
    assert resp.error_info is None
    assert resp.outputs["shared"][output_name] == values
