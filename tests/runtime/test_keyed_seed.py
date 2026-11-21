"""A rowwise chat op derives each row's seed from that row's key."""

import copy
import hashlib
import json
from typing import Any

import pytest

from lumilake_server.dynamic.driver import build_round
from lumilake_server.dynamic.template import DriverTemplate, plan_for_round
from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.parser.common import make_id
from lumilake_server.parser.keyed_seed import SEED_FN_NAME, rewrite_native_keys, seed_op
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.utils import lambda_runtime

_ROWS = [
    {"case_id": "c1", "question": "q1"},
    {"case_id": "c2", "question": "q2"},
    {"case_id": "c1", "question": "q3"},
]
_COLUMN = [json.dumps(row) for row in _ROWS]
_ANSWER = {
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
}


def _payload() -> dict[str, Any]:
    cases = {"id": "Cases", "op": "DataOp", "inputs": [], "data": _COLUMN}
    return {"seed": 7, "ops": [cases, copy.deepcopy(_ANSWER)]}


def _graph(payload: dict[str, Any]) -> Any:
    spec = next(iter(parse_yaml_payload(payload).values()))
    return RuntimeGraphBuilder().build(
        Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    )


def _seed_node(payload: dict[str, Any]) -> Any:
    nodes = _graph(payload).nodes.values()
    return next(n for n in nodes if n.data_spec.get("fn_name") == SEED_FN_NAME)


def _generated(payload: dict[str, Any]) -> Any:
    node = _seed_node(payload)
    return lambda_runtime.materialize(node.data_spec["code"], SEED_FN_NAME)


def _built(template_id: str = "Answer", round_number: int | None = None) -> Any:
    op = seed_op(
        "Answer",
        "Cases",
        "items.output",
        job_seed=7,
        round_number=round_number,
        template_id=template_id,
    )
    return lambda_runtime.materialize(op["_code"], op["fn_name"])


def _documented_seed(
    keys: list[Any], op_id: str = "Answer", round_number: int | None = None
) -> list[int]:
    """The docs/OPS.md formula: SHA-256 over [seed, op id, round, key] as
    canonical JSON, first 8 bytes big-endian, mod 2^31."""

    def one(key: Any) -> int:
        text = json.dumps([7, op_id, round_number, key], separators=(",", ":"))
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % 2**31

    return [one(key) for key in keys]


def _round_seed(body: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    """The generated seed op of round 2 of a one-op template body."""
    template = DriverTemplate.model_validate(
        {"first": body, "next": body, "stop_when_empty": "x"}
    )
    plan = plan_for_round(template, 2)
    build = build_round(
        plan.ops,
        node_registry={ref: {"op": "DataOp"} for ref in results},
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
    return next(op for op in build.graph.values() if op.get("fn_name") == SEED_FN_NAME)


def test_the_generated_function_matches_the_documented_seed() -> None:
    """The parsed step draws the documented seed by key, and every input counts."""
    fn = _generated(_payload())
    keys = [row["case_id"] for row in _ROWS]
    assert [fn((row,)) for row in _COLUMN] == _documented_seed(keys)
    assert [fn((row,)) for row in reversed(_COLUMN)] == _documented_seed(keys[::-1])
    direct = _built()((keys[0],))
    assert _built("Other")((keys[0],)) != direct
    assert _built(round_number=1)((keys[0],)) != direct


def test_a_keyed_seed_keeps_slices_and_reads_the_chat_op() -> None:
    """A row-mode seed op lets slices split, and the chat op reads its column."""
    graph = _graph(_payload())
    assert graph.list_lambda_cardinality_nodes() == set()
    fn = _generated(_payload())
    split = [fn((row,)) for row in _COLUMN[:2]] + [fn((row,)) for row in _COLUMN[2:]]
    assert split == [fn((row,)) for row in _COLUMN]
    chat = next(n for n in graph.nodes.values() if n.task_type == "inference")
    seed_id = _seed_node(_payload()).node_id
    assert chat.data_spec["seeds"] == {"node": seed_id, "path": "value.items.value"}


def test_a_template_round_keeps_its_id_and_reads_an_earlier_node() -> None:
    """A template op keeps its unrendered id, and a later round's chat op keys
    on an earlier round's node, forwarded flat for the row function to read."""
    key = {"key": {"node": "Cases", "path": "items.output.case_id"}}
    body = {"ops": [{**_ANSWER, "id": "Answer{{round}}", "rowwise_seed": key}]}
    generated = _round_seed(body, {"Cases": [{"output": row} for row in _ROWS]})
    fn = lambda_runtime.materialize(generated["_code"], generated["fn_name"])
    assert generated["_inputs"] == ["input_round_2_Cases_items_output_case_id_4420"]
    assert [fn(("c1",)), fn(("c2",))] == _documented_seed(
        ["c1", "c2"], "Answer{{round}}", 2
    )


def test_a_user_op_named_after_the_seed_keeps_its_own_id() -> None:
    """The generated row function takes an id under the ``seed`` prefix, which
    no op type emits, so a user op named like it builds beside the key form."""
    payload = _payload()
    payload["ops"].append(
        {
            "id": "Answer__seed",
            "op": "LambdaOp",
            "inputs": [],
            "fn_name": "passthrough",
            "code": "def passthrough(row):\n    return row\n",
        }
    )
    spec = next(iter(parse_yaml_payload(payload).values()))
    seed_id = next(
        op_id
        for op_id, op in spec["graph"].items()
        if op.get("fn_name") == SEED_FN_NAME
    )
    user_id = make_id("yaml_workflow", "lambda", "Answer__seed")
    assert user_id in spec["graph"]
    assert spec["graph"][user_id]["fn_name"] == "passthrough"
    assert seed_id != user_id


def test_the_key_form_is_refused() -> None:
    fn = _generated(_payload())
    with pytest.raises(ValueError, match="Answer' has no row key in .*case_id"):
        fn((json.dumps({"question": "q"}),))
    with pytest.raises(ValueError, match="Answer' has a null row key in .*case_id"):
        fn((json.dumps({"case_id": None}),))
    payloads = [_payload() for _ in range(3)]
    payloads[0]["ops"][1].pop("rowwise_template")
    payloads[1]["ops"][1]["config"]["api"] = {"url": "x"}
    payloads[2]["ops"][1]["rowwise_seed"]["node"] = "Cases"
    for payload, match in zip(payloads, ["rowwise_template", "API", "combines"]):
        with pytest.raises(ValueError, match=match):
            _graph(payload)


def test_a_native_graph_takes_the_same_key_form() -> None:
    native = next(iter(parse_yaml_payload(_payload()).values()))["graph"]
    chat_id = next(k for k, v in native.items() if v["_op"] == "LLMChatOp")
    cases_id = next(k for k, v in native.items() if v["_op"] == "DataOp")
    seed_key = {"key": {"node": cases_id, "path": "items.output.case_id"}}
    native[chat_id] = {**native[chat_id], "rowwise_seed": seed_key}
    out = rewrite_native_keys(native, job_seed=7)
    assert out[f"{chat_id}__seed"]["mode"] == "row"
    # The graph is built in insertion order, so the seed op comes first.
    assert list(out).index(f"{chat_id}__seed") < list(out).index(chat_id)
    native[f"{chat_id}__seed"] = {"_op": "DataOp"}
    with pytest.raises(ValueError, match="already holds"):
        rewrite_native_keys(native, job_seed=7)
