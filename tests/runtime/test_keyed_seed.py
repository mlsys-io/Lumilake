"""A rowwise chat op derives each row's seed from that row's key."""

import json
from typing import Any

import pytest
import yaml

from lumilake_server.dynamic.driver import build_round
from lumilake_server.dynamic.template import DriverTemplate, plan_for_round
from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.parser.keyed_seed import rewrite_native_keys, row_seed
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.utils import lambda_runtime

_ROWS = [
    {"case_id": "c1", "question": "q1"},
    {"case_id": "c2", "question": "q2"},
    {"case_id": "c1", "question": "q3"},
]

_COLUMN = [json.dumps(row) for row in _ROWS]

_WORKFLOW = {
    "seed": 7,
    "inputs": {"Rows": _COLUMN},
    "ops": [
        {
            "id": "Cases",
            "op": "LambdaOp",
            "inputs": ["Rows"],
            "fn_name": "rows",
            "mode": "row",
            "code": "def rows(row):\n    return row[0]\n",
        },
        {
            "id": "Answer",
            "op": "LLMChatOp",
            "inputs": ["Cases"],
            "config": {"model": "m"},
            "messages": [{"role": "user", "content": "placeholder"}],
            "rowwise_template": "{question}",
            "rowwise_columns": [
                {"label": "question", "node": "Cases", "path": "items.output.question"}
            ],
            "rowwise_seed": {"key": {"node": "Cases", "path": "items.output.case_id"}},
        },
    ],
    "outputs": [{"name": "answer", "ref": "Answer"}],
}


def _payload(**changes: Any) -> dict[str, Any]:
    payload = yaml.safe_load(yaml.safe_dump(_WORKFLOW))
    payload.update(changes)
    return payload


def _nodes(payload: dict[str, Any]) -> dict[str, Any]:
    spec = next(iter(parse_yaml_payload(payload).values()))
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    return RuntimeGraphBuilder().build(compiled).nodes


def _seed_spec(payload: dict[str, Any]) -> dict[str, Any]:
    node = next(
        node for node_id, node in _nodes(payload).items() if "__seed" in node_id
    )
    return node.data_spec


def _seeds(spec: dict[str, Any], rows: list[Any]) -> list[Any]:
    """Run the generated row function over each row, as its step does."""
    fn = lambda_runtime.materialize(spec["code"], spec["fn_name"])
    return [fn((row,)) for row in rows]


def test_a_key_gives_the_same_seed_wherever_the_row_sits() -> None:
    spec = _seed_spec(_payload())
    rows = list(_COLUMN)
    seeds = _seeds(spec, rows)
    assert seeds[0] == seeds[2] != seeds[1]
    assert _seeds(spec, [rows[1], rows[2], rows[0]]) == [seeds[1], seeds[2], seeds[0]]


def test_a_keyed_seed_does_not_pin_the_whole_input_list() -> None:
    """A row-mode seed op is not a list-cardinality node, so the input set can
    still be split into slices, and a row's seed is the same in any slice."""
    payload = _payload()
    workflow = next(iter(parse_yaml_payload(payload).values()))
    compiled = Graph.from_json(workflow["graph"]).compile(**workflow["inputs"])
    graph = RuntimeGraphBuilder().build(compiled)
    assert graph.list_lambda_cardinality_nodes() == set()
    seed_spec = _seed_spec(payload)
    rows = list(_COLUMN)
    first, second = _seeds(seed_spec, rows[:2]), _seeds(seed_spec, rows[2:])
    assert first + second == _seeds(seed_spec, rows)


def test_the_chat_op_reads_the_generated_seed_column() -> None:
    nodes = _nodes(_payload())
    chat = next(node for node in nodes.values() if node.task_type == "inference")
    (seed_id,) = [node_id for node_id in nodes if "__seed" in node_id]
    assert "mode" not in nodes[seed_id].data_spec
    assert chat.data_spec["seeds"] == {
        "node": seed_id,
        "path": "value.items.value",
    }
    assert nodes[seed_id].data_spec["plan"] == [
        {
            "kind": "stage",
            "node": next(
                node_id
                for node_id in nodes
                if "Cases" in node_id and "__seed" not in node_id
            ),
        }
    ]


def test_a_different_job_seed_op_id_or_round_gives_a_different_seed() -> None:
    column = list(_COLUMN)
    base = _seeds(_seed_spec(_payload()), column)
    assert _seeds(_seed_spec(_payload(seed=8)), column) != base
    key = _ROWS[0]["case_id"]
    direct = row_seed(7, "Answer", None, key)
    assert row_seed(7, "Other", None, key) != direct
    assert row_seed(7, "Answer", 1, key) != direct


def test_a_template_op_keeps_its_unrendered_id_across_rounds() -> None:
    answer = _payload()["ops"][1]
    body = {
        "ops": [
            dict(answer, id="Answer{{round}}", inputs=["Cases"]),
            {"id": "Cases", "op": "DataOp", "data": _COLUMN},
        ]
    }
    template = DriverTemplate.model_validate(
        {"first": body, "next": body, "stop_when_empty": "x"}
    )
    plan = plan_for_round(template, 2)
    build = build_round(
        plan.ops,
        node_registry={},
        round_index=2,
        goal="g",
        observations=[],
        topology=[],
        preview_width=50,
        model="m",
        max_tokens=10,
        temperature=0.1,
        job_seed=7,
        template_ids=plan.template_ids,
    )
    generated = next(v for k, v in build.graph.items() if k.endswith("__seed"))
    assert "[7, 'Answer{{round}}', 2, value]" in generated["_code"]


def test_a_later_round_keys_on_an_earlier_rounds_node() -> None:
    """A round-2 chat op keys on a column of a round-1 node: the column
    forwards as a flat input, and the row function reads that value itself."""
    earlier = "Cases"
    results = {earlier: [{"output": row} for row in _ROWS]}
    body = {
        "ops": [
            {
                "id": "Answer{{round}}",
                "op": "LLMChatOp",
                "inputs": [earlier],
                "config": {"model": "m"},
                "messages": [{"role": "user", "content": "placeholder"}],
                "rowwise_template": "{question}",
                "rowwise_seed": {
                    "key": {"node": earlier, "path": "items.output.case_id"}
                },
            }
        ]
    }
    template = DriverTemplate.model_validate(
        {"first": body, "next": body, "stop_when_empty": "x"}
    )
    plan = plan_for_round(template, 2)
    build = build_round(
        plan.ops,
        node_registry={earlier: {"op": "DataOp"}},
        round_index=2,
        goal="g",
        observations=[],
        topology=[],
        preview_width=50,
        model="m",
        max_tokens=10,
        temperature=0.1,
        results=results,
        job_seed=7,
        template_ids=plan.template_ids,
    )
    generated = next(v for k, v in build.graph.items() if k.endswith("__seed"))
    forwarded = build.forwarded_inputs[f"{earlier} | items.output.case_id"]
    (input_id,) = generated["_inputs"]
    assert build.graph[input_id]["_op"] == "InputOp"
    assert build.graph[input_id]["name"] == f"{earlier} | items.output.case_id"
    assert forwarded == ["c1", "c2", "c1"]
    spec = {"code": generated["_code"], "fn_name": generated["fn_name"]}
    assert _seeds(spec, forwarded) == [
        row_seed(7, "Answer{{round}}", 2, key) for key in forwarded
    ]


def test_a_missing_or_null_key_fails_naming_the_op() -> None:
    spec = _seed_spec(_payload())
    with pytest.raises(ValueError, match="Answer' has no row key in .*question"):
        _seeds(spec, [json.dumps({"question": "q"})])
    with pytest.raises(ValueError, match="Answer' has a null row key in .*case_id"):
        _seeds(spec, [json.dumps({"case_id": None})])
    with pytest.raises(ValueError, match="'seed' must be an integer"):
        parse_yaml_payload(_payload(seed="7"))


def test_the_key_form_is_refused_without_a_rowwise_template_or_in_api_mode() -> None:
    without_template = _payload()
    without_template["ops"][1].pop("rowwise_template")
    with pytest.raises(ValueError, match="rowwise_template"):
        _nodes(without_template)
    api_mode = _payload()
    api_mode["ops"][1]["config"]["api"] = {"url": "https://example.test"}
    with pytest.raises(ValueError, match="API mode"):
        _nodes(api_mode)


def test_a_native_graph_takes_the_same_key_form() -> None:
    native = next(iter(parse_yaml_payload(_payload()).values()))["graph"]
    chat_id = next(k for k, v in native.items() if v["_op"] == "LLMChatOp")
    cases_id = next(k for k, v in native.items() if v["_op"] == "LambdaOp")
    rewritten = native.copy()
    rewritten[chat_id] = {
        **native[chat_id],
        "rowwise_seed": {"key": {"node": cases_id, "path": "items.output.case_id"}},
    }
    out = rewrite_native_keys(rewritten, job_seed=7)
    generated = out[f"{chat_id}__seed"]
    assert generated["mode"] == "row"
    assert out[chat_id]["rowwise_seed"] == {
        "node": f"{chat_id}__seed",
        "path": "items.output",
    }
    taken = native.copy()
    taken[chat_id] = rewritten[chat_id]
    taken[f"{chat_id}__seed"] = {"_id": f"{chat_id}__seed", "_op": "DataOp"}
    with pytest.raises(ValueError, match="already holds"):
        rewrite_native_keys(taken, job_seed=7)
