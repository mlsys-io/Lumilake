"""A rowwise chat op binds a per-row seed column into its data spec."""

from collections.abc import Callable
from typing import Any

import pytest
import yaml

from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder

_WORKFLOW = yaml.safe_load("""
name: seeded-chat
inputs: {Q: [Q1, Q2]}
ops:
- {id: Seeds, op: LambdaOp, inputs: [Q], fn_name: seeds, mode: row,
   code: "def seeds(c):\\n    return [7, 9]\\n"}
- {id: Answer, op: LLMChatOp, inputs: [Seeds], config: {model: m},
   messages: [{role: user, content: placeholder}], rowwise_template: "{prompt}",
   rowwise_columns: [{label: prompt, node: Seeds, path: items.output}],
   rowwise_seed: {node: Seeds, path: items.output}}
outputs: [{name: answer, ref: Answer}]
""")


def _build(mutate: Callable[[dict[str, Any]], None]) -> Any:
    payload = yaml.safe_load(yaml.safe_dump(_WORKFLOW))
    mutate(payload)
    spec = next(iter(parse_yaml_payload(payload).values()))
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    nodes = RuntimeGraphBuilder().build(compiled).nodes
    (node_id,) = [node_id for node_id in nodes if "Answer" in node_id]
    return nodes[node_id]


def test_rowwise_seed_lowers_to_task_seeds() -> None:
    node = _build(lambda payload: None)
    assert node.data_spec["seeds"] == {
        "node": next(dep for dep in node.dependencies if "Seeds" in dep),
        "path": "value.items.output",
    }


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda p: p["ops"][1].pop("rowwise_template"), "rowwise_template"),
        (lambda p: p["ops"][1]["config"].update(num_beams=3), "num_beams"),
    ],
)
def test_rowwise_seed_refused(
    mutate: Callable[[dict[str, Any]], None], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _build(mutate)
