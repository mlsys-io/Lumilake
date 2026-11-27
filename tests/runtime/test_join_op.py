"""Keyed joins through the production path: YAML -> graph -> generated step."""

import json
from pathlib import Path
from typing import Any

import pytest

from lumilake_server.data_specs import ColumnPlanEntry, DataFrame, PythonData
from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime import join_step, python_step
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder

_ROWS = [
    {"case_id": "a", "text": "hello"},
    {"case_id": "b", "text": "DIAGNOSIS READY: flu"},
    {"case_id": "c", "text": "REQUEST TEST: cbc"},
]

_PRELUDE = """
name: j
inputs:
  Rows:
    - '{"case_id": "a", "text": "hello"}'
    - '{"case_id": "b", "text": "DIAGNOSIS READY: flu"}'
    - '{"case_id": "c", "text": "REQUEST TEST: cbc"}'
ops:
  - id: State
    op: LambdaOp
    mode: list
    inputs: [Rows]
    fn_name: state
    code: |
      def state(c):
          return [json.loads(v) for v in c[0]]
  - id: Route
    op: RouterOp
    inputs: [State]
    path: text
    default_branch: patient
    rules:
      - {branch: diagnose, match: contains, value: "DIAGNOSIS READY"}
"""

_TURN = """
  - id: PatientTurn
    op: LLMChatOp
    inputs: [State]
    messages:
      - {role: user, content: "Answer the patient."}
    config: {model: test/model}
    rowwise_template: "Reply to {text}"
    rowwise_columns: [{label: text, node: State, path: items.output.text}]
"""

_REPLY = """
  - id: Reply
    op: LambdaOp
    mode: row
    inputs: [State]
    fn_name: reply
    code: |
      def reply(row):
          return 1
"""


def _parse(ops: str) -> dict[str, Any]:
    return parse_yaml_payload(_PRELUDE + ops)["j"]


def _merge(inputs: str, rest: str = "") -> str:
    """A join over ``inputs`` keyed by ``case_id``, followed by ``rest``."""
    return (
        f"  - id: Merge\n    op: JoinOp\n    inputs: [{inputs}]\n    by:"
        f" case_id\n{rest}"
    )


def _join_of(runtime: Any) -> Any:
    """The runtime node that runs the join."""
    return next(
        node
        for node in runtime.nodes.values()
        if isinstance(node.data_spec, PythonData) and node.data_spec.mode == "join"
    )


def _build(ops: str) -> Any:
    spec = _parse(ops)
    return RuntimeGraphBuilder().build(
        Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    )


def _join_node(ops: str) -> Any:
    return _join_of(_build(ops))


_RUNTIME_MOUNT: dict[str, str] = {}


@pytest.fixture(scope="session", autouse=True)
def _task_runtime_mount(tmp_path_factory: pytest.TempPathFactory) -> None:
    path = tmp_path_factory.mktemp("join-runtime") / "lumilake-runtime.py"
    path.write_bytes(python_step.runtime_blob().body)
    _RUNTIME_MOUNT["path"] = str(path)


def _run(
    tmp_path: Path,
    node: Any,
    upstream: dict[int, list[dict[str, Any]]],
    name: str = "run",
) -> dict[str, Any]:
    """Run the generated step over one staged result list per plan entry."""
    plan = node.data_spec.plan
    stages: dict[str, str] = {}
    for index, items in upstream.items():
        stage = tmp_path / name / f"{plan[index].node}-{index}"
        stage.mkdir(parents=True)
        (stage / "results.json").write_text(
            json.dumps({"task_id": "t", "result": {"items": items}})
        )
        stages[str(plan[index].node)] = str(stage)
    code = join_step.join_source(
        node.node_id, node.data_spec.join, plan, python_step.TASK_RUNTIME_MOUNT
    ).replace(python_step.TASK_RUNTIME_MOUNT, _RUNTIME_MOUNT["path"])
    namespace: dict[str, Any] = {}
    exec(code, namespace)
    return namespace["main"](stages)


def test_join_merges_a_disjoint_branch_and_leaves_its_fields_out(
    tmp_path: Path,
) -> None:
    node = _join_node(_merge("State, Route.diagnose"))
    out = _run(tmp_path, node, {0: [{"output": _ROWS}], 1: [{"output": [_ROWS[1]]}]})
    rows = [item["output"] for item in out["items"]]
    assert [row["case_id"] for row in rows] == ["a", "b", "c"]
    assert rows[1]["text"] == _ROWS[1]["text"]
    assert set(rows[0]) == {"case_id", "text"}


def test_a_join_attaching_a_chat_feeds_a_list_lambda_and_records_its_output() -> None:
    runtime = _build(
        _TURN
        + _merge(
            "State, PatientTurn",
            """
    attach: {patient_reply: PatientTurn}
  - id: Summary
    op: LambdaOp
    mode: list
    inputs: [Merge]
    fn_name: summary
    code: |
      def summary(c):
          return [row["case_id"] for row in c[0]]
outputs:
  - {name: merged, ref: Merge}
""",
        )
    )
    merge = _join_of(runtime)
    summary = next(
        node for node in runtime.nodes.values() if "Summary_" in node.node_id
    )
    (entry,) = summary.data_spec.plan
    assert entry == ColumnPlanEntry(kind="stage", node=merge.node_id)
    assert summary.dependencies == (merge.node_id,)
    assert runtime.output_node_map == {merge.node_id: "merged"}
    assert runtime.output_paths[merge.node_id] == "items.output"


_AMBIGUOUS_TURN = """
  - id: PatientTurn
    op: LLMChatOp
    inputs: [Route.patient]
    messages:
      - {role: user, content: "Reply to the patient"}
    config: {model: test/model}
    rowwise_template: "Reply to {case_id}"
    rowwise_columns:
      - {label: case_id, node: Route.patient, path: items.output.case_id}
      - {label: text, node: State, path: items.output.text}
"""


def test_join_refuses_an_attach_whose_row_source_is_not_one_node() -> None:
    ops = _AMBIGUOUS_TURN + _merge(
        "State, Route.patient, PatientTurn", "    attach: {reply: PatientTurn}\n"
    )
    with pytest.raises(ValueError, match="reply.*several nodes"):
        _join_node(ops)


def test_join_refuses_a_format_op_as_an_attach_target() -> None:
    """A format op is a row source only as a hop on a chain, never a target."""
    ops = (
        '  - {id: Prompt, op: FormatOp, inputs: [State], template: "{0}", format_args:'
        " [State]}\n" + _merge("State, Prompt", "    attach: {prompt: Prompt}\n")
    )
    with pytest.raises(ValueError, match="attach field 'prompt' reading .*hop on a"):
        _join_node(ops)


def test_join_fails_on_a_bad_key(tmp_path: Path) -> None:
    """A repeated first-input key and a further-input row without its key fail."""
    node = _join_node(_merge("State, Route.diagnose"))
    stray = {"text": "no key at all"}
    with pytest.raises(ValueError, match="duplicate key"):
        _run(
            tmp_path,
            node,
            {0: [{"output": [_ROWS[1]] * 2}], 1: [{"output": []}]},
            name="duplicate",
        )
    with pytest.raises(ValueError, match="cannot be placed"):
        _run(
            tmp_path,
            node,
            {0: [{"output": [_ROWS[1]]}], 1: [{"output": [stray]}]},
            name="stray",
        )


def test_join_how_keeps_the_keys_its_choice_names(tmp_path: Path) -> None:
    stray = {"case_id": "z", "text": "not in the first input"}
    staged = {
        0: [{"output": [_ROWS[0], _ROWS[1]]}],
        1: [{"output": [_ROWS[1], stray]}],
    }

    def run(how: str, name: str) -> list[dict[str, Any]]:
        node = _join_node(_merge("State, Route.diagnose", f"    how: {how}\n"))
        return [item["output"] for item in _run(tmp_path, node, staged, name)["items"]]

    left = run("left", "left")
    assert [row["case_id"] for row in left] == ["a", "b"]
    assert left[0]["text"] == "hello"
    assert [row["case_id"] for row in run("inner", "inner")] == ["b"]
    full = run("full", "full")
    assert [row["case_id"] for row in full] == ["a", "b", "z"]
    assert full[2] == stray
    with pytest.raises(ValueError, match="how"):
        _join_node(_merge("State, Route.diagnose", "    how:\n"))


def test_join_require_match_refuses_a_key_an_input_lacks(tmp_path: Path) -> None:
    node = _join_node(_merge("State, Route.diagnose", "    require_match: true\n"))
    with pytest.raises(ValueError, match=r'key "a" is missing from'):
        _run(tmp_path, node, {0: [{"output": _ROWS}], 1: [{"output": [_ROWS[1]]}]})


def test_join_reads_a_table_in_index_order_and_keeps_a_row_value_typed(
    tmp_path: Path,
) -> None:
    node = _join_node(_REPLY + _merge("State, Reply", "    attach: {reply: Reply}\n"))
    table = {"case_id": {"0": "a", "2": "c"}, "text": {"0": "x", "2": "z"}}
    reply = [{"output": '{"n": 1}', "value": '{"n": 1}'}] * 2
    out = _run(
        tmp_path,
        node,
        {0: [{"output": [{"df": json.dumps(table)}]}], 1: reply},
    )
    assert [item["output"] for item in out["items"]] == [
        {"case_id": "a", "text": "x", "reply": {"n": 1}},
        {"case_id": "c", "text": "z", "reply": {"n": 1}},
    ]
    with pytest.raises(ValueError, match="table is not a column table"):
        _run(tmp_path, node, {0: [{"output": [{"df": "[1, 2]"}]}]}, name="bad")


_MARK = {"_fail": {"op": "P", "reason": "no JSON object"}}
_DIAG = _merge("State, Route.diagnose")
_ATTACH = _REPLY + _merge("State, Reply", "    attach: {reply: Reply}\n")
# A row Lambda over the diagnose branch, so its values reach only that key.
_PART = _REPLY.replace("[State]", "[Route.diagnose]") + _merge(
    "State, Route.diagnose, Reply", "    attach: {part: Reply}\n"
)
_A, _B, _C = _ROWS
_NULL = {"case_id": None, "text": "no key"}


def _rows(*values: Any) -> list[dict[str, Any]]:
    return [{"output": list(values)}]


def _cells(*values: Any) -> list[dict[str, Any]]:
    return [{"output": value, "value": value} for value in values]


# A join's staged inputs and its value column, in plan order.
_LIVE = {0: _rows(_A, _C), 1: _cells("hi", _MARK)}
_GAPPED = {0: _rows(_A, _MARK, _C), 1: _cells("hi", _MARK, "bye")}
_PARTED = {0: _rows(_A, _B, _C), 1: _rows(_B), 2: _cells(1), 3: _rows(_B)}


@pytest.mark.parametrize(
    "ops, upstream, expected",
    [
        # A marked row keeps its slot: in the first input the row is the marker, in a
        # further input the key's output is.
        (_DIAG, {0: _rows(_A, _MARK, _B), 1: _rows(_B)}, [_A, _MARK, _B]),
        # A marked attach value marks its paired key, whether the value column covers
        # the live source rows only or every source row. An attach whose values cannot
        # be paired one per row fails the job, as does a live row whose key field is
        # null.
        (_ATTACH, _LIVE, [{**_A, "reply": "hi"}, _MARK]),
        (_ATTACH, _GAPPED, [{**_A, "reply": "hi"}, _MARK, {**_C, "reply": "bye"}]),
        (_ATTACH, {0: _rows(_A, _C), 1: _cells("hi")}, "values attached but"),
        (_DIAG, {0: _rows(_A, _NULL), 1: _rows(_B)}, "row has no key field"),
        # fail_row marks the rows an attach source lacks a key for.
        (
            _PART + "    on_missing: fail_row\n",
            _PARTED,
            ["marked", {**_B, "part": 1}, "marked"],
        ),
    ],
)
def test_a_marked_join_row_keeps_its_key_and_every_other_row(
    ops: str,
    upstream: dict[int, list[dict[str, Any]]],
    expected: Any,
    tmp_path: Path,
) -> None:
    node = _join_node(ops)
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            _run(tmp_path, node, upstream)
        return
    got = []
    for item in _run(tmp_path, node, upstream)["items"]:
        row = item["output"]
        if row.get("_fail", {}).get("op") == node.node_id:
            assert "is missing from" in row["_fail"]["reason"]
            row = "marked"
        got.append(row)
    assert got == expected


def test_join_output_reads_as_a_rowwise_node_column() -> None:
    runtime = _build(
        _merge(
            "State, Route.diagnose",
            """
  - id: Ask
    op: LLMChatOp
    inputs: [Merge]
    messages:
      - {role: user, content: "Go"}
    config: {model: test/model}
    rowwise_template: "{text}"
    rowwise_columns: [{label: text, node: Merge, path: items.output.text}]
""",
        )
    )
    merge = _join_of(runtime)
    ask = next(
        node for node in runtime.nodes.values() if isinstance(node.data_spec, DataFrame)
    )
    (column,) = ask.data_spec.columns
    assert column.node == merge.node_id
    # A join's per-row fields sit under the python envelope, as a list Lambda's do, not
    # at the declared path.
    assert column.path == "value.items.output.text"
