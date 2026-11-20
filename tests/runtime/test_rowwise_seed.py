"""A rowwise chat op binds a per-row seed column into its data spec."""

from collections.abc import Callable
from typing import Any

import pytest
import yaml

from lumilake import envs
from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder

_WORKFLOW = yaml.safe_load("""
name: seeded-chat
inputs: {Q: [Q1, Q2]}
ops:
- {id: Prompts, op: LambdaOp, inputs: [Q], fn_name: prompts, mode: row,
   code: "def prompts(c):\\n    return 'ask ' + c[0]\\n"}
- {id: Seeds, op: LambdaOp, inputs: [Q], fn_name: seeds, mode: row,
   code: "def seeds(c):\\n    return len(c[0])\\n"}
- {id: Answer, op: LLMChatOp, inputs: [Prompts, Seeds], config: {model: m},
   messages: [{role: user, content: placeholder}], rowwise_template: "{prompt}",
   rowwise_columns: [{label: prompt, node: Prompts, path: items.output}],
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


def test_rowwise_seed_from_a_literal_input_is_inlined() -> None:
    def seed_from_input(payload: dict[str, Any]) -> None:
        payload["inputs"] = {"Q": ["Q1", "Q2"], "S": ["7", "", "null"]}
        payload["ops"][2]["inputs"] = ["Prompts", "S"]
        payload["ops"][2]["rowwise_seed"] = {"node": "S"}

    node = _build(seed_from_input)
    assert node.data_spec["seeds"] == [7, None, None]


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda p: p["ops"][2].pop("rowwise_template"), "rowwise_template"),
        (lambda p: p["ops"][2]["config"].update(num_beams=3), "num_beams"),
        (
            lambda p: p["ops"][2]["config"].update(api={"url": "http://x"}),
            "API mode",
        ),
        (
            lambda p: p["ops"].insert(
                0,
                {
                    "id": "Rows",
                    "op": "DataRetrievalOp",
                    "inputs": [],
                    "data_spec": {"type": "lumid", "mode": "sql", "template": "s"},
                },
            )
            or p["ops"][3]["rowwise_seed"].update(node="Rows")
            or p["ops"][3]["inputs"].append("Rows"),
            "one value per row",
        ),
    ],
)
def test_rowwise_seed_refused(
    mutate: Callable[[dict[str, Any]], None], match: str, monkeypatch: Any
) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "token")
    with pytest.raises(ValueError, match=match):
        _build(mutate)
