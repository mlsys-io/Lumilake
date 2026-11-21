"""Failed records and ParseOp, through YAML -> graph -> runtime graph -> step."""

import json
from pathlib import Path
from typing import Any

import pytest

from lumilake import envs
from lumilake_server.graphs import Graph
from lumilake_server.ops.router_ops import expand_routers
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime import python_step
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.utils.lambda_runtime import materialize

_RUNTIME_NAME = "lumilake-runtime.py"


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")
    (tmp_path / _RUNTIME_NAME).write_bytes(python_step.runtime_blob().body)


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


def _run(
    ops: list[dict[str, Any]], values: dict[str, list[str]], tmp_path: Path
) -> list[Any]:
    """Build the workflow's python tasks from YAML, run them in order, writing
    each result where its consumer reads it."""
    graph = _build(ops, values)
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
        mount = str(tmp_path / _RUNTIME_NAME)
        exec(task["code"].replace(python_step.TASK_RUNTIME_MOUNT, mount), namespace)
        items = namespace["main"](stages)["items"]
        stage = tmp_path / node_id
        stage.mkdir()
        (stage / "results.json").write_text(json.dumps({"value": {"items": items}}))
        stages[node_id] = str(stage)
    return items


def _parse(**fields: Any) -> dict[str, Any]:
    return {"id": "P", "op": "ParseOp", "inputs": ["Answer"], **fields}


_WHOLE = "def f(columns):\n    return list(columns[0])\n"
_MARKED = json.dumps({"_fail": {"op": "P", "reason": "no JSON object"}})


def _lower(**fields: Any) -> dict[str, Any]:
    return {
        "id": "L",
        "op": "LambdaOp",
        "inputs": ["P"],
        "fn_name": "f",
        "code": "def f(row):\n    return 'ran ' + row[0]\n",
        **fields,
    }


def _router(**fields: Any) -> dict[str, Any]:
    return {
        "id": "R",
        "op": "RouterOp",
        "inputs": ["P"],
        "path": "verdict",
        "default_branch": "ok",
        "rules": [{"branch": "no", "match": "equals", "value": "no"}],
        **fields,
    }


_MARKER = {"_fail": {"op": "Q", "reason": "x"}}


@pytest.mark.parametrize(
    "answer, expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Sure! Here it is: {"a": 1} hope that helps', {"a": 1}),
        ("[1, 2]", [1, 2]),
        (json.dumps(_MARKER), _MARKER),
        ("no idea", "no JSON object"),
        ('{"answer": "yes", "confidence": 0.', "no JSON object"),
        ('I compared [1] and [2]. {"answer": "yes"}', "no JSON object"),
        ('{"_fail": 1, "a": 2}', "reserved key"),
    ],
    ids=[
        "record",
        "fenced",
        "prose-wrapped",
        "list",
        "literal-marker",
        "prose",
        "truncated",
        "citation",
        "reserved-key",
    ],
)
def test_parse_json_reads_a_record_or_marks_the_row(
    answer: str, expected: Any, tmp_path: Path
) -> None:
    """A record decodes as it is; anything else becomes a marker whose reason
    names the problem, and a reply that is literally a marker stays one."""
    (item,) = _run([_parse(format="json")], {"Answer": [answer]}, tmp_path)
    if isinstance(expected, str):
        assert item["value"]["_fail"]["reason"].startswith(expected)
    else:
        assert item["value"] == expected


@pytest.mark.parametrize(
    "ops, answer, expected",
    [
        ([_parse(format="json", require=["eligibility"])], '{"a": 1}', "missing key"),
        ([_parse(format="json", require=["answer"])], '[{"answer": 1}]', "not a list"),
        ([_parse(format="regex", pattern=r"(?P<score>\d+)")], "none", "did not match"),
        (
            [_parse(format="regex", pattern=r"Score: (?P<score>\d+)/\d+")],
            "Score: 7/10",
            {"score": "7"},
        ),
    ],
    ids=["missing-key", "require-list", "pattern", "match"],
)
def test_parse_reads_a_row_by_its_spec_or_marks_it(
    ops: list[dict[str, Any]], answer: str, expected: Any, tmp_path: Path
) -> None:
    items = _run(ops, {"Answer": [answer]}, tmp_path)
    if isinstance(expected, str):
        assert expected in items[0]["value"]["_fail"]["reason"]
    else:
        assert items[0]["value"] == expected


def test_parse_rejects_a_bad_spec_naming_the_op() -> None:
    payload = {"name": "p", "inputs": {"Answer": ["x"]}}
    with pytest.raises(ValueError, match="ParseOp 'P' pattern must name"):
        parse_yaml_payload({**payload, "ops": [_parse(format="regex", pattern="1")]})


def _model_reading(node: str, kind: str = "chat") -> dict[str, Any]:
    """A model op of ``kind`` reading ``node`` through its rowwise column or
    its embedding content."""
    if kind == "embedding":
        return {
            "id": "E",
            "op": "EmbeddingOp",
            "inputs": [node],
            "input": node,
            "config": {"model": "BAAI/bge-small-en-v1.5"},
        }
    spec: dict[str, Any] = {
        "id": "M",
        "op": "LLMChatOp" if kind == "chat" else "LLMVisionOp",
        "inputs": [node],
        "messages": [{"role": "user", "content": "hello"}],
        "rowwise_template": "Summarize {value}",
        "rowwise_columns": [{"label": "value", "node": node, "path": "items.output"}],
        "config": {"model": "meta-llama/Llama-3.1-8B-Instruct"},
    }
    return {**spec, "image_source": "Images"} if kind == "vision" else spec


@pytest.mark.parametrize(
    "carrying, value",
    [
        (_lower(inputs=["Answer"]), _MARKED),
        (_lower(inputs=["Answer"], mode="aligned", code=_WHOLE), _MARKED),
        (_lower(inputs=["Answer"], mode="list", code=_WHOLE), _MARKED),
        (_router(inputs=["Answer"], error_branch=True), _MARKED),
        (_lower(inputs=["Answer"]), json.dumps([json.loads(_MARKED)])),
    ],
    ids=["row", "aligned", "list", "error", "grouped"],
)
def test_a_model_op_reading_marked_values_is_refused(
    carrying: dict[str, Any], value: str
) -> None:
    node = "R.error" if carrying["op"] == "RouterOp" else carrying["id"]
    values = {"Answer": [value], "Images": ["img.png"]}
    for kind in ("chat", "embedding", "vision"):
        model = _model_reading(node, kind)
        with pytest.raises(ValueError, match=f"model op '\\w+_p_{model['id']}"):
            _build([carrying, model], values)


def test_a_user_entry_cannot_declare_clears_markers() -> None:
    # The flag is set by a router's expansion only, on the branches it
    # generates, so it is not a workflow field anywhere it could be declared.
    with pytest.raises(ValueError, match="clears_markers"):
        parse_yaml_payload(
            json.dumps(
                {
                    "name": "p",
                    "inputs": {"Answer": ["x"]},
                    "ops": [_parse(format="json"), _lower(clears_markers=True)],
                }
            )
        )
    # The branch a router expands to keeps its clearing flag through the build,
    # so a model op reading it is allowed and one reading the error branch is
    # not.
    values = {"Answer": ['{"a": 1}']}
    router = _router(inputs=["L"], error_branch=True)
    ops = [_parse(format="json"), _lower(), router, _model_reading("R.ok")]
    assert _build(ops, values) is not None
    with pytest.raises(ValueError, match="rule or default"):
        _build([*ops[:-1], _model_reading("R.error")], values)


@pytest.mark.parametrize(
    "carrying, key",
    [
        (_lower(), "value"),
        (
            _lower(
                mode="aligned",
                code="def f(columns):\n    return ['ran ' + c for c in columns[0]]\n",
            ),
            "output",
        ),
    ],
    ids=["row", "aligned"],
)
def test_a_row_or_aligned_lambda_keeps_a_marked_row_in_place(
    carrying: dict[str, Any], key: str, tmp_path: Path
) -> None:
    items = _run(
        [_parse(format="json"), carrying],
        {"Answer": ["no idea", '{"a": 1}', "bad"]},
        tmp_path,
    )
    assert items[0][key]["_fail"]["reason"] == "no JSON object"
    assert items[1][key] == 'ran {"a": 1}'
    assert items[2][key]["_fail"]["reason"] == "no JSON object"


def test_a_list_lambda_emits_every_value_of_a_returned_generator(
    tmp_path: Path,
) -> None:
    lister = _lower(
        mode="list", code="def f(items):\n    return (v.upper() for v in items[0])\n"
    )
    items = _run(
        [_parse(format="json"), lister], {"Answer": ['{"a": 1}', '{"b": 2}']}, tmp_path
    )
    assert [item["output"] for item in items] == ['{"A": 1}', '{"B": 2}']


@pytest.mark.parametrize(
    "tail_code, refused",
    [
        ("def f(items):\n    return items[0]\n", False),
        (
            "def f(items):\n    return items[0][:1] +"
            " [{'_fail': {'op': 'X', 'reason': 'new'}}]\n",
            True,
        ),
    ],
    ids=["pass_through", "new_marker"],
)
def test_a_router_error_branch_passes_its_markers_and_refuses_a_new_one(
    tail_code: str, refused: bool, tmp_path: Path
) -> None:
    # The router's list lambda re-encodes each row, so the error branch gets
    # JSON text and passes on its own decoded copy; a marker it builds itself
    # is refused.
    tail = _lower(inputs=["R.error"], mode="list", code=tail_code)
    ops = [_parse(format="json"), _router(error_branch=True), tail]
    if refused:
        with pytest.raises(ValueError, match="reserved '_fail' key"):
            _run(ops, {"Answer": ["no idea"]}, tmp_path)
        return
    items = _run(ops, {"Answer": ['{"verdict": "yes"}', "no idea"]}, tmp_path)
    (marked,) = [item["output"] for item in items]
    assert marked["_fail"]["reason"] == "no JSON object"


def test_a_router_without_an_error_branch_sends_a_marked_row_nowhere() -> None:
    ops, clearing = expand_routers([_router()])
    assert clearing == {op["id"] for op in ops}
    for op in ops:
        assert materialize(op["code"], op["fn_name"])(([_MARKED],)) == []


def test_a_user_function_returning_the_reserved_key_fails_the_task(
    tmp_path: Path,
) -> None:
    writer = _lower(code="def f(row):\n    return {'_fail': 'mine'}\n")
    with pytest.raises(ValueError, match="reserved '_fail' key"):
        _run([_parse(format="json"), writer], {"Answer": ['{"a": 1}']}, tmp_path)
