"""Keyed joins through the production path: YAML -> graph -> generated step."""

import json
from pathlib import Path
from typing import Any

import pytest

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


def _merge(inputs: str, rest: str = "", by: str = "case_id") -> str:
    """A join over ``inputs`` keyed by ``by``, followed by ``rest``."""
    return (
        f"  - id: Merge\n    op: JoinOp\n    inputs: [{inputs}]\n"
        f"    by: {by}\n{rest}"
    )


def _find(runtime: Any, **spec: Any) -> Any:
    """The one runtime node whose data_spec matches every ``spec`` entry."""
    return next(
        node
        for node in runtime.nodes.values()
        if all(node.data_spec.get(key) == value for key, value in spec.items())
    )


def _build(ops: str) -> Any:
    spec = _parse(ops)
    return RuntimeGraphBuilder().build(
        Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    )


def _join_node(ops: str) -> Any:
    return _find(_build(ops), mode=join_step.JOIN_MODE)


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
    plan = node.data_spec["plan"]
    stages: dict[str, str] = {}
    for index, items in upstream.items():
        stage = tmp_path / name / f"{plan[index].get('node', 'literal')}-{index}"
        stage.mkdir(parents=True)
        (stage / "results.json").write_text(
            json.dumps({"task_id": "t", "result": {"items": items}})
        )
        stages[plan[index].get("node", "literal")] = str(stage)
    code = join_step.join_source(
        node.node_id, node.data_spec["join"], plan, python_step.TASK_RUNTIME_MOUNT
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
    merge = _find(runtime, mode=join_step.JOIN_MODE)
    summary = next(
        node for node in runtime.nodes.values() if "Summary_" in node.node_id
    )
    (entry,) = summary.data_spec["plan"]
    assert entry == {"kind": "stage", "node": merge.node_id}
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
        '  - {id: Prompt, op: FormatOp, inputs: [State], template: "{0}",'
        " format_args: [State]}\n"
        + _merge("State, Prompt", "    attach: {prompt: Prompt}\n")
    )
    with pytest.raises(ValueError, match="attach field 'prompt' reading .*hop on a"):
        _join_node(ops)


def test_join_fails_on_a_bad_key(tmp_path: Path) -> None:
    node = _join_node(_merge("State, Route.diagnose"))
    stray = {"case_id": "z", "text": "not in the first input"}
    with pytest.raises(ValueError, match="duplicate key"):
        _run(
            tmp_path,
            node,
            {0: [{"output": [_ROWS[1]] * 2}], 1: [{"output": []}]},
            name="duplicate",
        )
    with pytest.raises(ValueError, match=r'key "z" is not in'):
        _run(
            tmp_path,
            node,
            {0: [{"output": [_ROWS[1]]}], 1: [{"output": [_ROWS[1], stray]}]},
            name="stray",
        )


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


_TAG = """
  - id: Tag
    op: LambdaOp
    mode: row
    inputs: [Rows]
    fn_name: tag
    code: |
      def tag(row):
          return row["case_id"].upper()
"""


def test_join_reads_a_workflow_input_as_its_first_column(tmp_path: Path) -> None:
    node = _join_node(_TAG + _merge("Rows, Tag", "    attach: {tag: Tag}\n"))
    assert node.data_spec["plan"][0]["kind"] == "literal"
    out = _run(tmp_path, node, {1: [{"output": v, "value": v} for v in "ABC"]})
    assert [item["output"]["tag"] for item in out["items"]] == ["A", "B", "C"]


def test_join_keys_on_every_field_of_a_composite_key(tmp_path: Path) -> None:
    """Rows sharing the first key field stay distinct under a composite key."""
    node = _join_node(_merge("State, Route.diagnose", by="[case_id, text]"))
    same = [dict(_ROWS[1], extra=1), dict(_ROWS[1], text="another row")]
    out = _run(tmp_path, node, {0: [{"output": _ROWS}], 1: [{"output": same}]})
    assert [item["output"]["case_id"] for item in out["items"]] == ["a", "b", "c"]
    assert out["items"][1]["output"]["extra"] == 1


def test_join_refuses_a_composite_key_that_is_empty_or_not_a_field_name() -> None:
    with pytest.raises(ValueError, match="Merge.*requires 'by'"):
        _join_node("  - {id: Merge, op: JoinOp, inputs: [State], by: []}\n")
    with pytest.raises(ValueError, match="Merge.*by must be dotted field names"):
        _join_node("  - {id: Merge, op: JoinOp, inputs: [State], by: [case_id, 3]}\n")
_MARK = {"_fail": {"op": "P", "reason": "no JSON object"}}
_DIAG = _merge("State, Route.diagnose")
_ATTACH = _REPLY + _merge("State, Reply", "    attach: {reply: Reply}\n")
_ALL = [{"output": list(_ROWS)}]
_LIVE = [{"output": [_ROWS[0], _ROWS[2]]}]


@pytest.mark.parametrize(
    "ops, upstream, expected, fails",
    [
        # A marked first-input row keeps its position; every other row is whole.
        (
            _DIAG,
            {0: [{"output": [_ROWS[0], _MARK, _ROWS[1]]}], 1: [{"output": [_ROWS[1]]}]},
            [_ROWS[0], _MARK, _ROWS[1]],
            "",
        ),
        # A marked further input keeps its key and marks that key's output.
        (
            _DIAG,
            {0: _ALL, 1: [{"output": [{"case_id": "b", "text": _MARK}]}]},
            [_ROWS[0], _MARK, _ROWS[2]],
            "",
        ),
        # One whose key cannot be read cannot be placed.
        (_DIAG, {0: _ALL, 1: [{"output": [_MARK]}]}, None, "cannot be placed"),
        # A marked attach value marks the key its position pairs with.
        (
            _ATTACH,
            {
                0: _LIVE,
                1: [
                    {"output": "hi", "value": "hi"},
                    {"output": json.dumps(_MARK), "value": _MARK},
                ],
                2: _LIVE,
            },
            [{**_ROWS[0], "reply": "hi"}, _MARK],
            "",
        ),
        # fail_row marks only the rows the further input lacks, with its key.
        (
            _DIAG + "    on_missing: fail_row\n",
            {0: _ALL, 1: [{"output": [_ROWS[1]]}]},
            "fail_row",
            "",
        ),
        # An attach length mismatch is a job failure under either policy.
        (
            _ATTACH + "    on_missing: fail_row\n",
            {0: _LIVE, 1: [{"output": "hi", "value": "hi"}]},
            None,
            "values attached but",
        ),
    ],
    ids=[
        "first_input",
        "further_keyed",
        "further_unkeyed",
        "attach_value",
        "fail_row",
        "attach_length",
    ],
)
def test_a_marked_join_row_keeps_its_key_and_every_other_row(
    ops: str,
    upstream: dict[int, list[dict[str, Any]]],
    expected: Any,
    fails: str,
    tmp_path: Path,
) -> None:
    node = _join_node(ops)
    if fails:
        with pytest.raises(ValueError, match=fails):
            _run(tmp_path, node, upstream)
        return
    got = [item["output"] for item in _run(tmp_path, node, upstream)["items"]]
    if expected == "fail_row":
        # The join names itself and the absent key on each row it lacks.
        assert [row.get("_fail", {}).get("op") == node.node_id for row in got] == [
            True,
            False,
            True,
        ]
        assert got[1]["case_id"] == "b"
        return
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
    merge = _find(runtime, mode=join_step.JOIN_MODE)
    ask = _find(runtime, type="dataframe")
    (column,) = ask.data_spec["columns"]
    assert column["node"] == merge.node_id
    # A join's per-row fields sit under the python envelope, as a list
    # Lambda's do, not at the declared path.
    assert column["path"] == "value.items.output.text"
