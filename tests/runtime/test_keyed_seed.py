"""A rowwise chat op derives each row's seed from that row's key."""

import copy
import hashlib
import json
from typing import Any

import pytest

from lumilake_server.dynamic.driver import build_round
from lumilake_server.dynamic.template import DriverTemplate, plan_for_round
from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload, yaml_parser
from lumilake_server.parser.common import make_id
from lumilake_server.parser.keyed_seed import SEED_FN_NAME, rewrite_native_keys, seed_op
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.utils import lambda_runtime

_ROWS = [
    {"case_id": c, "question": q} for c, q in [("c1", "q1"), ("c2", "q2"), ("c1", "q3")]
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
    "rowwise_seed": {
        "key": {"node": "Cases", "path": "items.output.case_id"},
        "pairing": "shared",
    },
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


def _documented_seed(
    keys: list[Any], op_id: str = "Answer", round_number: int | None = None
) -> list[int]:
    """The docs/OPS.md formula: SHA-256 over [seed, op id, round, key] as
    canonical JSON (sorted keys, no whitespace, non-ASCII unescaped, UTF-8),
    first 8 bytes big-endian, mod 2^31."""

    def one(key: Any) -> int:
        text = json.dumps(
            [7, op_id, round_number, key],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % 2**31

    return [one(key) for key in keys]


def _round_build(body: dict[str, Any], results: dict[str, Any]) -> Any:
    template = DriverTemplate.model_validate(
        {"first": body, "next": body, "stop_when_empty": "x"}
    )
    plan = plan_for_round(template, 2)
    return build_round(
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


def test_the_generated_function_matches_the_documented_seed() -> None:
    """The parsed step draws the documented seed by key."""
    fn = _generated(_payload())
    keys = [row["case_id"] for row in _ROWS]
    assert [fn((row,)) for row in _COLUMN] == _documented_seed(keys)
    assert [fn((row,)) for row in reversed(_COLUMN)] == _documented_seed(keys[::-1])


def test_a_keyed_seed_reads_the_chat_op() -> None:
    """A row-mode seed op lets slices split, and the chat op reads its column."""
    graph = _graph(_payload())
    assert graph.list_lambda_cardinality_nodes() == set()
    chat = next(n for n in graph.nodes.values() if n.task_type == "inference")
    seed_id = _seed_node(_payload()).node_id
    assert chat.data_spec["seeds"] == {"node": seed_id, "path": "value.items.value"}


def test_a_template_round_keeps_its_id_and_reads_an_earlier_node() -> None:
    """A template op keeps its unrendered id, and a later round's chat op keys
    on an earlier round's node, forwarded flat for the row function to read."""
    answer = {**_ANSWER, "id": "Answer{{round}}"}
    results = {"Cases": [{"output": row} for row in _ROWS]}
    build = _round_build({"ops": [answer]}, results)
    generated = next(
        op for op in build.graph.values() if op.get("fn_name") == SEED_FN_NAME
    )
    (producer,) = (build.graph[ref] for ref in generated["_inputs"])
    assert (producer["_op"], producer["name"].startswith("Cases")) == ("InputOp", True)
    fn = lambda_runtime.materialize(generated["_code"], generated["fn_name"])
    assert [fn(("c1",)), fn(("c2",))] == _documented_seed(
        ["c1", "c2"], "Answer{{round}}", 2
    )


def test_a_user_op_named_after_the_seed_keeps_its_own_id() -> None:
    """The generated row function takes an id under the ``seed`` prefix, which
    no op type emits, so a user op named like it builds beside the key form."""
    user = {"id": "Answer__seed", "op": "LambdaOp", "inputs": [], "fn_name": "p"}
    user["code"] = "def p(row):\n    return row\n"
    payload = _payload()
    payload["ops"].append(user)
    spec = next(iter(parse_yaml_payload(payload).values()))["graph"]
    assert spec[make_id("yaml_workflow", "lambda", "Answer__seed")]["fn_name"] == "p"
    assert spec[make_id("yaml_workflow", "seed", "Answer")]["fn_name"] == SEED_FN_NAME


def test_a_text_cell_is_the_key_as_it_stands() -> None:
    """A cell that is not JSON object or array text is kept as written, so the
    text ``"7"`` hashes as the string, not as the number it spells."""
    op = seed_op(
        "Answer",
        "Cases",
        "items.output",
        7,
        None,
        "LLMChatOp",
        pairing="shared",
    )
    fn = lambda_runtime.materialize(op["_code"], op["fn_name"])
    assert fn(("7",)) == _documented_seed(["7"])[0]
    assert fn(("7",)) != _documented_seed([7])[0]


def test_a_duplicate_key_fails_unless_paired() -> None:
    """Two rows of one key draw one seed, so the key form refuses them and
    names the op and both rows, and ``pairing: shared`` declares it."""
    payload = _payload()
    payload["ops"][1]["rowwise_seed"].pop("pairing")
    with pytest.raises(ValueError, match="Answer' .*rows 0 and 2 share the key"):
        yaml_parser.check_keyed_seed_duplicates(payload, {})
    payload["ops"][1]["rowwise_seed"]["pairing"] = "shared"
    yaml_parser.check_keyed_seed_duplicates(payload, {})
    payload["ops"][1]["rowwise_seed"]["pairing"] = "paired"
    with pytest.raises(ValueError, match="pairing must be one of"):
        _graph(payload)


def test_the_key_form_is_refused() -> None:
    fn = _generated(_payload())
    with pytest.raises(ValueError, match="Answer' has no row key in .*case_id"):
        fn((json.dumps({"question": "q"}),))
    with pytest.raises(ValueError, match="Answer' has a null row key in .*case_id"):
        fn((json.dumps({"case_id": None}),))
    # A literal ``%s`` in the op id is escaped before the value placeholder is
    # added, so the error path still raises the documented ValueError.
    payload = _payload()
    payload["ops"][1]["id"] = "50%sale"
    with pytest.raises(ValueError, match="50%sale' has no row key"):
        _generated(payload)((json.dumps({"question": "q"}),))
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
    native[f"{chat_id}__seed"] = {}
    with pytest.raises(ValueError, match="already holds"):
        rewrite_native_keys(native, job_seed=7)
    native[chat_id]["config"] = ""
    with pytest.raises(ValueError, match="config must be a mapping"):
        rewrite_native_keys(native, job_seed=7)


def test_a_row_lambda_takes_a_seeded_key_form() -> None:
    """A row Lambda's ``seed`` form binds a generated seed node ahead of it,
    whose step passes each row's derived seed to the function as a second
    argument; the other modes and a one-parameter function are refused."""
    rows = [{"case_id": "k1"}, {"case_id": "k2"}]
    draw = {
        "id": "Draw",
        "op": "LambdaOp",
        "inputs": ["Cases"],
        "fn_name": "draw",
        "code": "def draw(row, seed):\n    return '%d' % seed\n",
        "seed": {"key": {"node": "Cases", "path": "items.output.case_id"}},
    }
    cases = {
        "id": "Cases",
        "op": "DataOp",
        "inputs": [],
        "data": [json.dumps(row) for row in rows],
    }
    payload: dict[str, Any] = {"seed": 7, "ops": [cases, draw]}
    spec = next(iter(parse_yaml_payload(payload).values()))["graph"]
    (seed_id,) = [
        op_id for op_id, op in spec.items() if op.get("fn_name") == SEED_FN_NAME
    ]
    assert spec[make_id("yaml_workflow", "lambda", "Draw")]["seed"] == {
        "node": seed_id,
        "path": "items.output",
    }
    with pytest.raises(ValueError, match="seed is available in modes"):
        parse_yaml_payload({"ops": [cases, {**draw, "mode": "list"}]})
    with pytest.raises(ValueError, match="exactly 2 parameters"):
        _graph(
            {"ops": [cases, {**draw, "code": "def draw(row):\n    return row[0]\n"}]}
        )
    node = next(
        n
        for n in _graph(payload).nodes.values()
        if n.data_spec.get("fn_name") == "draw"
    )
    assert [e.get("role") for e in node.data_spec["plan"]] == [None, "seed"]
