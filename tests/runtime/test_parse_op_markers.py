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
    """Build the workflow's python tasks from YAML and run them in order."""
    graph = _build(ops, values)
    stages: dict[str, str] = {}
    items: list[Any] = []
    for node_id in graph.node_order:
        node = graph.nodes[node_id]
        if node.task_type != "python":
            continue
        plan = node.data_spec.plan
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
        "code": "def f(row):\n    return 'ran ' + str(row[0])\n",
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
)
def test_parse_json_reads_a_record_or_marks_the_row(
    answer: str, expected: Any, tmp_path: Path
) -> None:
    """A record decodes as it is; anything else becomes a marker."""
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
    with pytest.raises(ValueError, match="ParseOp 'P': pattern must name"):
        parse_yaml_payload({**payload, "ops": [_parse(format="regex", pattern="1")]})


def _model_reading(node: str, kind: str = "chat") -> dict[str, Any]:
    """A model op of ``kind`` reading ``node``."""
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


def test_a_router_branch_clears_markers_only_where_its_own_expansion_set_it() -> None:
    # The flag is set on the branches a router's expansion produced, so a user graph
    # cannot declare it: its own ids never appear in the parser's set.
    values = {"Answer": ['{"a": 1}']}
    router = _router(inputs=["L"], error_branch=True)
    ops = [_parse(format="json"), _lower(), router, _model_reading("R.ok")]
    assert _build(ops, values) is not None
    user = _lower(_clears_markers=True)
    with pytest.raises(ValueError, match="rule or default"):
        _build([_parse(format="json"), user, _model_reading("L")], values)


@pytest.mark.parametrize(
    "carrying, key",
    [
        (_lower(), "value"),
        (
            _lower(
                mode="aligned",
                code="def f(c):\n    return ['ran ' + str(v) for v in c[0]]\n",
            ),
            "output",
        ),
    ],
    ids=["row", "aligned"],
)
def test_a_row_or_aligned_lambda_keeps_a_marked_row_in_place(
    carrying: dict[str, Any], key: str, tmp_path: Path
) -> None:
    # A parsed record arrives as its object; only a marker the function returns stays a
    # marker.
    items = _run(
        [_parse(format="json"), carrying],
        {"Answer": ["no idea", '{"a": 1}', "bad"]},
        tmp_path,
    )
    assert items[0][key]["_fail"]["reason"] == "no JSON object"
    assert items[1][key] == "ran {'a': 1}"
    assert items[2][key]["_fail"]["reason"] == "no JSON object"


def test_a_list_lambda_emits_every_value_of_a_returned_generator(
    tmp_path: Path,
) -> None:
    lister = _lower(
        mode="list",
        code="def f(items):\n    return (v['a'].upper() for v in items[0])\n",
    )
    items = _run(
        [_parse(format="json"), lister],
        {"Answer": ['{"a": "x"}', '{"a": "y"}']},
        tmp_path,
    )
    assert [item["output"] for item in items] == ["X", "Y"]


@pytest.mark.parametrize(
    "tail_code, refused",
    [
        ("def f(items):\n    return items[0]\n", False),
        (
            "def f(items):\n    return items[0][:1] + [{'_fail': {'op': 'X', 'reason':"
            " 'new'}}]\n",
            True,
        ),
    ],
    ids=["pass_through", "new_marker"],
)
def test_a_router_error_branch_passes_its_markers_and_refuses_a_new_one(
    tail_code: str, refused: bool, tmp_path: Path
) -> None:
    # The router's list lambda re-encodes each row, so the error branch gets JSON text
    # and passes on its own decoded copy; a marker it builds itself is refused.
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
    # A branch is a list-mode LambdaOp, so its column reaches it decoded.
    ops, _ = expand_routers([_router()])
    for op in ops:
        assert materialize(op["code"], op["fn_name"])(([json.loads(_MARKED)],)) == []


def test_a_user_function_returning_the_reserved_key_fails_the_task(
    tmp_path: Path,
) -> None:
    writer = _lower(code="def f(row):\n    return {'_fail': 'mine'}\n")
    with pytest.raises(ValueError, match="reserved '_fail' key"):
        _run([_parse(format="json"), writer], {"Answer": ['{"a": 1}']}, tmp_path)


@pytest.mark.parametrize(
    "answer, kept",
    [
        ('{"a": 1}', '{"a": 1}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('Here: {"a": 1} done', '{"a": 1}'),
        ("no idea", "no idea"),
        ('{"_fail": 1, "a": 2}', '{"_fail": 1, "a": 2}'),
    ],
    ids=["record", "fenced", "prose", "unparsable", "reserved"],
)
def test_a_parse_op_that_keeps_the_raw_text_is_not_a_marker_source(
    answer: str, kept: str, tmp_path: Path
) -> None:
    """`on_fail: raw` is no marker source, so a model op may read it."""
    items = _run([_parse(format="json", on_fail="raw")], {"Answer": [answer]}, tmp_path)
    assert items[0]["output"] == kept
    values = {"Answer": [answer], "Images": ["img.png"]}
    raw = [_parse(format="json", on_fail="raw"), _model_reading("P")]
    assert _build(raw, values) is not None
    with pytest.raises(ValueError, match="rule or default"):
        _build([_parse(format="json"), _model_reading("P")], values)


_EXPAND_UNITS = _lower(
    mode="expand",
    id="E",
    inputs=["Answer"],
    code="def f(row):\n    return [u for u in row[0]['units'].split(',') if u]\n",
)
_EMPTY_PARENT = ['{"units": "a,b"}', '{"units": ""}', '{"units": "c"}']


def _reduce(on_fail: str) -> dict[str, Any]:
    return _lower(
        mode="reduce",
        by="unit",
        on_fail=on_fail,
        code="def f(rows):\n    return len(rows)\n",
    )


_MARKED_KEY = json.dumps({"unit": "u", "score": json.loads(_MARKED)})


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({"on_empty": "fail"}, ["a@0.0", "b@0.1", "marker@1.0", "c@2.0"]),
        ({}, ["a@0.0", "b@0.1", "c@2.0"]),
    ],
    ids=["fail", "drop"],
)
def test_an_empty_expansion_records_a_failed_parent_or_nothing(
    fields: dict[str, Any], expected: list[str], tmp_path: Path
) -> None:
    items = _run([{**_EXPAND_UNITS, **fields}], {"Answer": _EMPTY_PARENT}, tmp_path)
    got = []
    for item in items:
        output = item["output"]
        if isinstance(output, dict) and list(output) == ["_fail"]:
            output = "marker"
        got.append(f"{output}@{item['parent']}.{item['child']}")
    assert got == expected


def test_an_expand_that_fails_an_empty_parent_is_a_marker_source() -> None:
    values = {"Answer": _EMPTY_PARENT}
    with pytest.raises(ValueError, match="rule or default"):
        _build([{**_EXPAND_UNITS, "on_empty": "fail"}, _model_reading("E")], values)
    assert _build([_EXPAND_UNITS, _model_reading("E")], values) is not None


@pytest.mark.parametrize(
    "op, answers, expected",
    [
        # A row whose key cannot be read is its own group and emits its marker, whether
        # the row is the marker or its key path reaches one.
        (
            _reduce("fail_group"),
            ["bad", json.dumps({"unit": json.loads(_MARKED)}), '{"unit": "u"}'],
            ["no JSON object", "no JSON object", 1],
        ),
        # A row whose key reads joins its group, and on_fail decides that group.
        (_reduce("fail_group"), [_MARKED_KEY], ["no JSON object"]),
        (_reduce("drop_failed"), [_MARKED_KEY, '{"unit": "u"}'], [1]),
        (_reduce("drop_failed"), [_MARKED_KEY], ["empty group"]),
        # An expand emits one tombstone for a marked parent at child zero, and every
        # item names its parent and child.
        (
            {**_EXPAND_UNITS, "inputs": ["P"]},
            ['{"units": "a,b"}', "bad"],
            ["a@0.0", "b@0.1", "no JSON object@1.0"],
        ),
    ],
)
def test_a_marked_row_keeps_its_group_or_its_slot(
    op: dict[str, Any], answers: list[str], expected: list[Any], tmp_path: Path
) -> None:
    got = []
    for item in _run([_parse(format="json"), op], {"Answer": answers}, tmp_path):
        output = item["output"]
        if isinstance(output, dict):
            output = output["_fail"]["reason"]
        got.append(
            f"{output}@{item['parent']}.{item['child']}" if "parent" in item else output
        )
    assert got == expected
