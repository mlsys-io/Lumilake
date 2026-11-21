"""Failed records and ParseOp, through YAML -> graph -> runtime graph -> step."""

import json
from pathlib import Path
from typing import Any

import pytest

from lumilake import envs
from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime import python_step
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder

_RUNTIME_MOUNT = "/nonexistent-runtime"


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")
    path = tmp_path / "lumilake-runtime.py"
    path.write_bytes(python_step.runtime_blob().body)
    global _RUNTIME_MOUNT
    _RUNTIME_MOUNT = str(path)


def _run(
    ops: list[dict[str, Any]], values: dict[str, list[str]], tmp_path: Path
) -> list[Any]:
    """Build the workflow's python tasks from YAML, run them in order, writing
    each result where its consumer reads it."""
    payload = {
        "name": "p",
        "inputs": {name: vals for name, vals in values.items()},
        "ops": ops,
        "outputs": [{"name": "out", "ref": ops[-1]["id"]}],
    }
    spec = parse_yaml_payload(json.dumps(payload))["p"]
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    graph = RuntimeGraphBuilder().build(compiled)
    stages: dict[str, str] = {}
    items: list[Any] = []
    for node_id in graph.node_order:
        node = graph.nodes[node_id]
        if node.task_type != "python":
            continue
        plan = node.data_spec["plan"]
        literals = python_step.literal_blobs(plan).values()
        blobs: dict[str, python_step.PublishedBlob] = {
            blob.key: blob for blob in (python_step.runtime_blob(), *literals)
        }
        task = python_step.flowmesh_spec(node.node_id, node.data_spec, blobs)
        namespace: dict[str, Any] = {}
        code = task["code"].replace(python_step.TASK_RUNTIME_MOUNT, _RUNTIME_MOUNT)
        exec(code, namespace)
        items = namespace["main"](stages)["items"]
        stage = tmp_path / node_id
        stage.mkdir()
        (stage / "results.json").write_text(json.dumps({"value": {"items": items}}))
        stages[node_id] = str(stage)
    return items


def _parse(**fields: Any) -> dict[str, Any]:
    return {"id": "P", "op": "ParseOp", "inputs": ["Answer"], **fields}


def _values(items: list[Any]) -> list[Any]:
    return [item["value"] for item in items]


@pytest.mark.parametrize(
    "answer, expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Sure! Here it is: {"a": 1} hope that helps', {"a": 1}),
        ("[1, 2]", [1, 2]),
    ],
)
def test_parse_json_reads_a_plain_fenced_or_prose_wrapped_record(
    answer: str, expected: Any, tmp_path: Path
) -> None:
    ops = [_parse(format="json")]
    assert _values(_run(ops, {"Answer": [answer]}, tmp_path)) == [expected]


@pytest.mark.parametrize(
    "answer",
    [
        "no idea",
        '{"answer": "yes", "evidence": {"pmid": 1}, "confidence": 0.',
        'I compared [1] and [2]. {"answer": "yes"}',
    ],
)
def test_parse_json_marks_an_unparsable_row(answer: str, tmp_path: Path) -> None:
    ops = [_parse(format="json")]
    items = _run(ops, {"Answer": [answer, '{"a": 1}']}, tmp_path)
    assert items[0]["value"]["_fail"]["reason"] == "no JSON object"
    assert items[1]["value"] == {"a": 1}


def test_parse_json_marks_a_reply_carrying_the_reserved_key(tmp_path: Path) -> None:
    ops = [_parse(format="json")]
    items = _run(ops, {"Answer": ['{"_fail": 1, "a": 2}']}, tmp_path)
    marker = items[0]["value"]["_fail"]
    assert marker["op"].split("_")[2] == "P"
    assert marker["reason"] == "reserved key '_fail'"


def test_parse_json_marks_a_row_missing_a_required_key(tmp_path: Path) -> None:
    ops = [_parse(format="json", require=["eligibility"])]
    items = _run(ops, {"Answer": ['{"a": 1}']}, tmp_path)
    assert items[0]["value"]["_fail"]["reason"] == "missing key 'eligibility'"


def _build(ops: list[dict[str, Any]], values: dict[str, list[str]]) -> Any:
    payload = {
        "name": "p",
        "inputs": values,
        "ops": ops,
        "outputs": [{"name": "out", "ref": ops[-1]["id"]}],
    }
    spec = parse_yaml_payload(json.dumps(payload))["p"]
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    return RuntimeGraphBuilder().build(compiled)


def _model_reads(node: str, path: str) -> dict[str, Any]:
    return {
        "id": "M",
        "op": "LLMChatOp",
        "inputs": [node],
        "messages": [{"role": "user", "content": "hello"}],
        "rowwise_template": "Summarize {value}",
        "rowwise_columns": [{"label": "value", "node": node, "path": path}],
        "config": {"model": "meta-llama/Llama-3.1-8B-Instruct"},
    }


def test_a_model_op_reading_a_marked_source_is_refused_at_build() -> None:
    ops = [_parse(format="json"), _lower(), _model_reads("L", "items.output")]
    with pytest.raises(ValueError, match="non-error branch"):
        _build(ops, {"Answer": ['{"a": 1}']})


def test_a_model_op_reading_a_router_branch_still_builds() -> None:
    router = {
        "id": "R",
        "op": "RouterOp",
        "inputs": ["L"],
        "default_branch": "ok",
        "rules": [{"branch": "no", "match": "equals", "value": "no"}],
    }
    ops = [
        _parse(format="json"),
        _lower(),
        router,
        _model_reads("R.ok", "items.output"),
    ]
    assert _build(ops, {"Answer": ['{"a": 1}']}) is not None


def test_parse_json_fails_require_on_a_list_reply(tmp_path: Path) -> None:
    ops = [_parse(format="json", require=["answer"])]
    items = _run(ops, {"Answer": ['[{"answer": 1}]']}, tmp_path)
    assert items[0]["value"]["_fail"]["reason"] == (
        "require needs a JSON object, not a list"
    )


def test_parse_regex_returns_the_named_groups_of_a_match(tmp_path: Path) -> None:
    ops = [_parse(format="regex", pattern=r"Score: (?P<score>\d+)/\d+")]
    assert _values(_run(ops, {"Answer": ["Score: 7/10"]}, tmp_path)) == [{"score": "7"}]


def test_parse_regex_marks_a_row_the_pattern_does_not_match(tmp_path: Path) -> None:
    ops = [_parse(format="regex", pattern=r"(?P<score>\d+)")]
    items = _run(ops, {"Answer": ["none"]}, tmp_path)
    assert items[0]["value"]["_fail"]["reason"] == "pattern did not match"


def test_parse_rejects_a_bad_spec_naming_the_op() -> None:
    with pytest.raises(ValueError, match="ParseOp 'P' pattern must name"):
        parse_yaml_payload(
            json.dumps(
                {
                    "name": "p",
                    "inputs": {"Answer": ["x"]},
                    "ops": [_parse(format="regex", pattern=r"\d+")],
                }
            )
        )


def _lower(**fields: Any) -> dict[str, Any]:
    return {
        "id": "L",
        "op": "LambdaOp",
        "inputs": ["P"],
        "fn_name": "f",
        "code": "def f(row):\n    return 'ran ' + row[0]\n",
        **fields,
    }


def test_a_row_lambda_skips_the_function_for_a_marked_row(tmp_path: Path) -> None:
    ops = [_parse(format="json"), _lower()]
    items = _run(ops, {"Answer": ["no idea", '{"a": 1}']}, tmp_path)
    assert items[0]["value"]["_fail"]["reason"] == "no JSON object"
    assert items[1]["value"] == 'ran {"a": 1}'


@pytest.mark.parametrize(
    "answers",
    [["no idea", '{"a": 1}', "bad"], ["no idea"], ["bad", "no idea"]],
)
def test_an_aligned_lambda_runs_on_the_live_rows_and_keeps_positions(
    answers: list[str], tmp_path: Path
) -> None:
    aligned = _lower(
        mode="aligned",
        code="def f(columns):\n    return ['ran ' + c for c in columns[0]]\n",
    )
    ops = [_parse(format="json"), aligned]
    items = _run(ops, {"Answer": answers}, tmp_path)
    expected = ["ran " + a if a == '{"a": 1}' else {"_fail": None} for a in answers]
    for item, want in zip(items, expected):
        if want == {"_fail": None}:
            assert item["output"]["_fail"]["reason"] == "no JSON object"
        else:
            assert item["output"] == want


def test_a_router_sends_marked_rows_to_a_declared_error_branch(tmp_path: Path) -> None:
    router = {
        "id": "R",
        "op": "RouterOp",
        "inputs": ["P"],
        "path": "verdict",
        "default_branch": "ok",
        "error_branch": True,
        "rules": [{"branch": "no", "match": "equals", "value": "no"}],
    }
    tail = {
        "id": "Out",
        "op": "LambdaOp",
        "inputs": ["R.error"],
        "fn_name": "first",
        "code": (
            "def first(row):\n"
            "    return json.dumps(row[0]) if row[0] is not None else 'none'\n"
        ),
    }
    values = {"Answer": ['{"verdict": "yes"}', "no idea"]}
    items = _run([_parse(format="json"), router, tail], values, tmp_path)
    assert "_fail" in items[0]["output"]


def test_a_router_without_an_error_branch_sends_a_marked_row_nowhere() -> None:
    from lumilake_server.ops.router_ops import RouterOp, expand_routers
    from lumilake_server.utils.lambda_runtime import materialize

    entry = {
        "id": "R",
        "op": "RouterOp",
        "inputs": ["P"],
        "path": "verdict",
        "default_branch": "ok",
        "rules": [{"branch": "no", "match": "equals", "value": "no"}],
    }
    marked = json.dumps({"_fail": {"op": "P", "reason": "no JSON object"}})
    for branch in RouterOp.from_entry(entry).branches:
        op = next(o for o in expand_routers([entry]) if o["id"] == f"R.{branch}")
        assert materialize(op["code"], op["fn_name"])(([marked],)) == []


def test_a_user_function_returning_the_reserved_key_fails_the_task(
    tmp_path: Path,
) -> None:
    writer = _lower(code="def f(row):\n    return {'_fail': 'mine'}\n")
    with pytest.raises(ValueError, match="reserved '_fail' key"):
        _run([_parse(format="json"), writer], {"Answer": ['{"a": 1}']}, tmp_path)
