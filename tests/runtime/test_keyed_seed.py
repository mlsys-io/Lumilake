"""A rowwise chat op derives each row's seed from that row's key."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from lumilake_server.data_specs import ColumnPlanEntry, PythonMode
from lumilake_server.dynamic.driver import build_round
from lumilake_server.dynamic.template import DriverTemplate, plan_for_round
from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload, yaml_parser
from lumilake_server.parser.common import make_id
from lumilake_server.parser.keyed_seed import (
    SAMPLE_FN_NAME,
    SAMPLE_SEED_LABEL,
    SAMPLE_SUFFIX,
    SEED_FN_NAME,
    rewrite_native_keys,
    seed_op,
)
from lumilake_server.runtime import python_step
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.runtime.runtime_ops import RuntimeOp
from lumilake_server.utils import lambda_runtime
from tests.support.seeds import documented_seed

_ROWS = [
    {"case_id": c, "question": q} for c, q in [("c1", "q1"), ("c2", "q2"), ("c1", "q3")]
]
_COLUMN = [json.dumps(row) for row in _ROWS]
_KEY = {"node": "Cases", "path": "items.output.case_id"}
_ANSWER: dict[str, Any] = {
    "id": "Answer",
    "op": "LLMChatOp",
    "inputs": ["Cases"],
    "config": {"model": "m"},
    "messages": [{"role": "user", "content": "placeholder"}],
    "rowwise_template": "{question}",
    "rowwise_columns": [{"label": "question", "node": "Cases"}],
    "rowwise_seed": {"key": _KEY, "pairing": "shared"},
}


def _wire(node: RuntimeOp) -> dict[str, Any]:
    """The data spec as the dict a worker receives."""
    return node.data_spec.model_dump(exclude_none=True)


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
    return next(n for n in nodes if _wire(n).get("fn_name") == SEED_FN_NAME)


def _generated(payload: dict[str, Any]) -> Any:
    return lambda_runtime.materialize(_wire(_seed_node(payload))["code"], SEED_FN_NAME)


def _round_build(
    body: dict[str, Any], results: dict[str, Any], symbols: list[str] | None = None
) -> Any:
    """Build one round over ``body``, with ``results`` as the run's node values."""
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
        symbols=symbols or [],
    )


def test_the_parsed_step_draws_the_documented_seed_by_key() -> None:
    """The generated row function draws the documented seed per key."""
    graph = _graph(_payload())
    seed_node = _seed_node(_payload())
    fn = lambda_runtime.materialize(_wire(seed_node)["code"], SEED_FN_NAME)
    keys = [row["case_id"] for row in _ROWS]
    assert [fn((row,)) for row in _COLUMN] == documented_seed(keys)
    assert [fn((row,)) for row in reversed(_COLUMN)] == documented_seed(keys[::-1])
    assert graph.list_lambda_cardinality_nodes() == set()
    chat = next(n for n in graph.nodes.values() if n.task_type == "inference")
    assert _wire(chat)["seeds"] == {
        "node": seed_node.node_id,
        "path": "value.items.value",
    }
    # Text that is not JSON object or array text is the key as it stands.
    op = seed_op("Answer", "Cases", "items.output", 7, None, "LLMChatOp")
    fn = lambda_runtime.materialize(op["_code"], op["fn_name"])
    assert fn(("7",)) == documented_seed(["7"])[0] != documented_seed([7])[0]


def test_a_repeated_key_is_refused_unless_paired() -> None:
    """A repeated key over a round, the run input or a literal column is refused."""
    results = {"Cases": [{"output": row} for row in _ROWS]}
    build = _round_build({"ops": [{**_ANSWER, "id": "Answer{{round}}"}]}, results)
    generated = next(
        op for op in build.graph.values() if op.get("fn_name") == SEED_FN_NAME
    )
    (producer,) = (build.graph[ref] for ref in generated["_inputs"])
    assert (producer["_op"], producer["name"].startswith("Cases")) == ("InputOp", True)
    fn = lambda_runtime.materialize(generated["_code"], generated["fn_name"])
    assert [fn(("c1",)), fn(("c2",))] == documented_seed(
        ["c1", "c2"], op_id="Answer{{round}}", round_number=2
    )
    repeated = {**_ANSWER, "rowwise_seed": {"key": _KEY}}
    with pytest.raises(ValueError, match="rows 0 and 2 share the key"):
        _round_build({"ops": [repeated]}, results)
    repeated["rowwise_seed"] = {"key": {"node": "Symbols", "path": "items.output"}}
    with pytest.raises(ValueError, match="rows 0 and 2 share the key"):
        _round_build({"ops": [repeated]}, {}, symbols=["AAA", "BBB", "AAA"])
    literal = _payload()
    literal["ops"][1]["rowwise_seed"].pop("pairing")
    with pytest.raises(ValueError, match="Answer' .*rows 0 and 2 share the key"):
        yaml_parser.check_keyed_seed_duplicates(literal, {})
    literal["ops"][1]["rowwise_seed"] = {"key": _KEY, "pairing": "paired"}
    with pytest.raises(
        ValueError, match="Answer' rowwise_seed: pairing: Input should be 'unique'"
    ):
        _graph(literal)


def test_the_key_form_is_refused() -> None:
    """A row with no key or a null key fails, as does a seed that fits no field."""
    bad, null = _payload(), _payload()
    bad["ops"][1]["id"] = "50%sale"
    for payload, cell, message in (
        (_payload(), {"question": "q"}, "has no row key in .*case_id"),
        (null, {"case_id": None}, "has a null row key in .*case_id"),
        (bad, {"question": "q"}, "50%sale' has no row key"),
    ):
        with pytest.raises(ValueError, match=message):
            _generated(payload)((json.dumps(cell),))
    missing, api = _payload(), _payload()
    missing["ops"][1].pop("rowwise_template")
    api["ops"][1]["config"]["api"] = {"url": "x"}
    for payload, match in ((missing, "rowwise"), (api, "API")):
        with pytest.raises(ValueError, match=match):
            _graph(payload)


def test_a_native_graph_builds_each_seed_form_through_its_model() -> None:
    """A native key form adds its row function ahead of the op reading it."""
    native = next(iter(parse_yaml_payload(_payload()).values()))["graph"]
    chat_id = next(k for k, v in native.items() if v["_op"] == "LLMChatOp")
    cases_id = next(k for k, v in native.items() if v["_op"] == "DataOp")
    keyed = {**native[chat_id], "rowwise_seed": {"key": {**_KEY, "node": cases_id}}}
    out = rewrite_native_keys({**native, chat_id: keyed}, job_seed=7)
    assert out[f"{chat_id}__seed"]["mode"] == "row"
    # The graph is built in insertion order, so the seed op comes first.
    assert list(out).index(f"{chat_id}__seed") < list(out).index(chat_id)
    with pytest.raises(ValueError, match="rowwise_seed: node: Field required"):
        Graph.from_json({**native, chat_id: {**keyed, "rowwise_seed": {"path": "p"}}})
    for entry, match in [
        ({**keyed, "config": {"model": "m", "nonsense": 1}}, "config is not valid"),
        ({**keyed, "config": ""}, "config must be a mapping"),
        ({**keyed, "_id": f"{chat_id}__seed"}, "already holds"),
    ]:
        native[f"{chat_id}__seed"] = {}
        with pytest.raises(ValueError, match=match):
            rewrite_native_keys({**native, chat_id: entry}, job_seed=7)
        del native[f"{chat_id}__seed"]
    # An op that names no kind is the graph builder's to refuse, not this pass's.
    untyped = {"_inputs": [], "_max_iter": None, "data": ["{}"]}
    assert rewrite_native_keys({"op": untyped}, job_seed=7) == {"op": untyped}


def test_a_row_lambda_takes_a_seeded_key_form() -> None:
    """A row Lambda's ``seed`` form binds a generated seed node ahead of it."""
    draw: dict[str, Any] = {"id": "Draw", "op": "LambdaOp", "inputs": ["Cases"]}
    draw["fn_name"] = "draw"
    draw["code"] = "def draw(row, seed):\n    return '%d' % seed\n"
    draw["seed"] = {"key": _KEY}
    cases = {"id": "Cases", "op": "DataOp", "inputs": [], "data": _COLUMN}
    payload: dict[str, Any] = {"seed": 7, "ops": [cases, draw]}
    spec = next(iter(parse_yaml_payload(payload).values()))["graph"]
    seed_id = next(k for k, v in spec.items() if v.get("fn_name") == SEED_FN_NAME)
    draw_id = make_id("yaml_workflow", "lambda", "Draw")
    assert spec[draw_id]["seed"] == {"node": seed_id, "path": "items.output"}
    with pytest.raises(ValueError, match="seed is available in modes"):
        parse_yaml_payload({"ops": [cases, {**draw, "mode": "list"}]})
    for code in ("def draw(row):\n    return row[0]\n", "lambda row: row[0]"):
        with pytest.raises(ValueError, match="exactly 2 parameters"):
            _graph({"ops": [cases, {**draw, "code": code}]})
    node = _graph(payload).nodes[draw_id]
    assert [e.get("role") for e in _wire(node)["plan"]] == [None, "seed"]
    # No op type emits the ``seed`` prefix, so a user op named like it builds on.
    named = {"id": "Answer__seed", "op": "LambdaOp", "inputs": []}
    named.update({"fn_name": "p", "code": "def p(row):\n    return row\n"})
    payload = _payload()
    payload["ops"].append(named)
    built = next(iter(parse_yaml_payload(payload).values()))["graph"]
    assert built[make_id("yaml_workflow", "seed", "Answer")]["fn_name"] == SEED_FN_NAME
    assert built[make_id("yaml_workflow", "lambda", "Answer__seed")]["fn_name"] == "p"


def _sampled(samples: int, **config: Any) -> dict[str, Any]:
    answer = copy.deepcopy(_ANSWER)
    answer["config"] = {"model": "m", "n": samples, **config}
    answer["rowwise_columns"].append(
        {"label": "case_id", "node": "Cases", "path": "items.output.case_id"}
    )
    cases = {"id": "Cases", "op": "DataOp", "inputs": [], "data": _COLUMN}
    return {"seed": 7, "ops": [cases, answer]}


def _whole_cell(samples: int) -> dict[str, Any]:
    """The sampled chat, with one column that reads its row's whole cell."""
    payload = _sampled(samples)
    payload["ops"][1]["rowwise_columns"][0] = {"label": "whole", "node": "Cases"}
    return payload


def _fan_out(payload: dict[str, Any]) -> Any:
    spec = next(iter(parse_yaml_payload(payload).values()))["graph"]
    op = next(v for v in spec.values() if v.get("fn_name") == SAMPLE_FN_NAME)
    return lambda_runtime.materialize(op["_code"], op["fn_name"]), op


def _run_fan_out(
    spec: dict[str, Any], inputs: dict[str, str], mount: Path
) -> list[Any]:
    """Run the fan-out through the wrapper a worker runs, and return its items."""
    namespace: dict[str, Any] = {}
    code = python_step.wrapper_source(
        "fan-out",
        spec["fn_name"],
        spec["code"],
        [ColumnPlanEntry.model_validate(entry) for entry in spec["plan"]],
        PythonMode.EXPAND,
    ).replace(python_step.TASK_RUNTIME_MOUNT, str(mount))
    exec(code, namespace)
    return namespace["main"](inputs)["items"]


def _sample_cells(payload: dict[str, Any]) -> list[dict[str, Any]]:
    fn, _ = _fan_out(payload)
    return [item for cell in _COLUMN for item in fn((cell,))]


def test_a_sample_per_row_draws_the_documented_pair() -> None:
    """Two parses of one payload give sample j of every key the same seed."""
    cells = _sample_cells(_sampled(2))
    keys = [row["case_id"] for row in _ROWS]
    assert [item["__seed"] for item in cells] == documented_seed(keys, samples=2)
    assert [item["__seed"] for item in cells] == [
        item["__seed"] for item in _sample_cells(_sampled(2))
    ]


def test_raising_the_sample_count_keeps_the_earlier_seeds() -> None:
    """Samples 0 and 1 of every key keep their seeds when n goes from 2 to 3."""
    keys = [row["case_id"] for row in _ROWS]
    two = [item["__seed"] for item in _sample_cells(_sampled(2))]
    three = [item["__seed"] for item in _sample_cells(_sampled(3))]
    assert three[::3] + three[1::3] == two[::2] + two[1::2]
    assert three == documented_seed(keys, samples=3)


def test_two_keys_never_share_a_sample_seed() -> None:
    """Two distinct keys draw disjoint seeds across all their samples."""
    cells = _sample_cells(_sampled(3))
    first_key = [item["__seed"] for item in cells[0:3]]
    second_key = [item["__seed"] for item in cells[3:6]]
    assert len(set(first_key + second_key)) == 6


def test_a_sample_chat_reads_every_column_from_the_fan_out() -> None:
    """n > 1 rewires the chat op's columns and seed to the fan-out node."""
    spec = next(iter(parse_yaml_payload(_sampled(3)).values()))["graph"]
    fan_out = next(
        op_id for op_id, v in spec.items() if v.get("fn_name") == SAMPLE_FN_NAME
    )
    answer = next(
        v for k, v in spec.items() if k.startswith("llm_yaml_workflow_Answer")
    )
    assert answer["rowwise_seed"] == {
        "node": fan_out,
        "path": f"items.output.{SAMPLE_SEED_LABEL}",
    }
    assert {column["node"] for column in answer["rowwise_columns"]} == {fan_out}
    assert "n" not in answer["config"]


def test_a_native_sample_count_keys_each_sample() -> None:
    """A native key form with n > 1 builds the fan-out and drops the request's n."""
    native = next(iter(parse_yaml_payload(_payload()).values()))["graph"]
    chat_id = next(k for k, v in native.items() if v["_op"] == "LLMChatOp")
    cases_id = next(k for k, v in native.items() if v["_op"] == "DataOp")
    sampled = {
        **native[chat_id],
        "config": {"model": "m", "n": 3},
        "rowwise_columns": [
            {"label": "question", "node": cases_id, "path": "items.output.question"},
            {"label": "case_id", "node": cases_id, "path": "items.output.case_id"},
        ],
        "rowwise_seed": {"key": {**_KEY, "node": cases_id}, "pairing": "shared"},
    }
    out = rewrite_native_keys({**native, chat_id: sampled}, job_seed=7)
    fan_out = out[f"{chat_id}{SAMPLE_SUFFIX}"]
    assert fan_out["mode"] == "expand"
    assert out[chat_id]["config"] == {"model": "m"}
    assert out[chat_id]["rowwise_seed"] == {
        "node": f"{chat_id}{SAMPLE_SUFFIX}",
        "path": f"items.output.{SAMPLE_SEED_LABEL}",
    }
    assert {column["node"] for column in out[chat_id]["rowwise_columns"]} == {
        f"{chat_id}{SAMPLE_SUFFIX}"
    }
    fn = lambda_runtime.materialize(fan_out["_code"], fan_out["fn_name"])
    keys = [row["case_id"] for row in _ROWS]
    cells = [item for cell in _COLUMN for item in fn((cell,))]
    assert [item[SAMPLE_SEED_LABEL] for item in cells] == documented_seed(
        keys, op_id=chat_id, samples=3
    )


def test_a_whole_cell_column_keeps_the_raw_cell(tmp_path: Path) -> None:
    """The task's own read path hands the fan-out a whole-cell column as it stands,
    so the chat renders the cell's text, as n = 1 does."""
    loaded = copy.deepcopy(_ANSWER)
    loaded["inputs"] = ["Load"]
    loaded["rowwise_template"] = "{whole}"
    loaded["rowwise_columns"] = [{"label": "whole", "node": "Load", "path": ""}]
    loaded["rowwise_seed"] = {
        "key": {"node": "Load", "path": "items.output.case_id"},
        "pairing": "shared",
    }
    loaded["config"] = {"model": "m", "n": 3}
    payload = {
        "seed": 7,
        "ops": [
            {"id": "Cases", "op": "DataOp", "inputs": [], "data": _COLUMN},
            {
                "id": "Load",
                "op": "LambdaOp",
                "inputs": ["Cases"],
                "fn_name": "load",
                "code": "def load(row):\n    return row[0]\n",
            },
            loaded,
        ],
    }
    mount = tmp_path / "lumilake-runtime.py"
    mount.write_bytes(python_step.runtime_blob().body)
    for source, upstream in ((_whole_cell(3), False), (payload, True)):
        spec = next(
            _wire(node)
            for node in _graph(source).nodes.values()
            if _wire(node).get("fn_name") == SAMPLE_FN_NAME
        )
        inputs: dict[str, str] = {}
        if upstream:
            (node_id,) = (
                entry["node"] for entry in spec["plan"] if entry["kind"] == "stage"
            )
            stage = tmp_path / node_id
            stage.mkdir()
            (stage / "results.json").write_text(
                json.dumps({"result": {"items": [{"output": c} for c in _COLUMN]}})
            )
            inputs[node_id] = str(stage)
        items = _run_fan_out(spec, inputs, mount)
        # Each whole-cell item carries the cell's own JSON text, not the decoded value.
        rendered = [item["output"]["whole"] for item in items]
        assert len(rendered) == len(_COLUMN) * 3
        assert rendered[::3] == _COLUMN


def test_a_sample_count_needs_local_columns() -> None:
    """n > 1 with API mode or a literal column is refused, naming the op."""
    api = _sampled(2)
    api["ops"][1]["config"]["api"] = {"url": "https://x/v1"}
    literal = _sampled(2)
    literal["ops"][1]["rowwise_columns"].append(
        {"label": "tone", "data": {"type": "list", "items": ["a"]}}
    )
    for payload, match in (
        (api, "Answer' rowwise_seed is available only for local vLLM"),
        (literal, "Answer' config.n > 1 needs node columns"),
    ):
        with pytest.raises(ValueError, match=match):
            parse_yaml_payload(payload)


def test_a_column_labelled_like_the_sample_seed_is_refused() -> None:
    """A user column under the fan-out's seed label is refused, naming the op."""
    sampled = _sampled(2)
    sampled["ops"][1]["rowwise_columns"].append(
        {"label": SAMPLE_SEED_LABEL, "node": "Cases", "path": "items.output.case_id"}
    )
    with pytest.raises(
        ValueError, match="Answer' config.n > 1 labels a rowwise column"
    ):
        parse_yaml_payload(sampled)
