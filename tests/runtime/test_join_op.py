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


def _merge(inputs: str, rest: str = "", by: str = "case_id") -> str:
    """A join over ``inputs`` keyed by ``by``, followed by ``rest``."""
    return (
        f"  - id: Merge\n    op: JoinOp\n    inputs: [{inputs}]\n"
        f"    by: {by}\n{rest}"
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
        stage = tmp_path / name / f"{plan[index].node or 'literal'}-{index}"
        stage.mkdir(parents=True)
        (stage / "results.json").write_text(
            json.dumps({"task_id": "t", "result": {"items": items}})
        )
        stages[plan[index].node or "literal"] = str(stage)
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
        # A marked row keeps its slot: in the first input the row is the marker,
        # in a further input the key's output is.
        (_DIAG, {0: _rows(_A, _MARK, _B), 1: _rows(_B)}, [_A, _MARK, _B]),
        # A marked attach value marks its paired key, whether the value column
        # covers the live source rows only or every source row. An attach whose
        # values cannot be paired one per row fails the job, as does a live
        # row whose key field is null.
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
    # A join's per-row fields sit under the python envelope, as a list
    # Lambda's do, not at the declared path.
    assert column.path == "value.items.output.text"


# A broadcast join: two rows of unit u1 and one of u2, a summary row per unit
# and a tag per first-input row.
_UNITS = [
    {"unit": "u1", "rid": "a"},
    {"unit": "u1", "rid": "b"},
    {"unit": "u2", "rid": "c"},
]
_SIZES = [{"unit": "u1", "size": 2}, {"unit": "u2", "size": 1}]
_JOINED = [
    {**_UNITS[0], "size": 2, "tag": "A"},
    {**_UNITS[1], "size": 2, "tag": "B"},
    {**_UNITS[2], "size": 1, "tag": "C"},
]
_BROADCAST_YAML = """
name: u
inputs: {Rows: []}
ops:
  - {id: State, op: DataOp, data: __ROWS__}
  - {id: Summary, op: DataOp, data: __SIZES__}
  - {id: Tag, op: LambdaOp, mode: row, inputs: [State], fn_name: tag,
     code: "def tag(row): return row['rid'].upper()"}
  - id: Merge
    op: JoinOp
    inputs: [State, Summary__TAG_INPUT__]
    by: unit
__FIELDS__
__ATTACH__
outputs: [{name: merged, ref: Merge}]
"""


def _broadcast(
    tmp_path: Path,
    rows: list[Any] | None = None,
    sizes: list[Any] | None = None,
    tags: list[str] | None = None,
    fields: tuple[str, ...] = ("row_key: [unit, rid]",),
    attach: bool = True,
    name: str = "run",
) -> list[Any]:
    """Run the broadcast join over one summary row per unit and one tag per row."""
    quote = [f"'{json.dumps(row)}'" for row in rows or _UNITS]
    summary = [f"'{json.dumps(row)}'" for row in sizes or _SIZES]
    code = (
        _BROADCAST_YAML.replace("__ROWS__", f"[{', '.join(quote)}]")
        .replace("__SIZES__", f"[{', '.join(summary)}]")
        .replace("__TAG_INPUT__", ", Tag" if attach else "")
        .replace("__ATTACH__\n", "    attach: {tag: Tag}\n" if attach else "")
        .replace("__FIELDS__\n", "".join(f"    {line}\n" for line in fields))
    )
    spec = parse_yaml_payload(code)["u"]
    runtime = RuntimeGraphBuilder().build(
        Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    )
    node = _join_of(runtime)
    upstream = {2: _cells(*(tags or ["A", "B", "C"]))} if attach else {}
    out = _run(tmp_path, node, upstream, name)
    return [item["output"] for item in out["items"]]


def test_a_broadcast_join_keeps_one_output_row_per_record(tmp_path: Path) -> None:
    """Several first-input rows may share a by key: each keeps a row of its own,
    in its order, and a record's fields follow its row_key, not its position."""
    assert _broadcast(tmp_path) == _JOINED
    reordered = [_UNITS[1], _UNITS[0], _UNITS[2]]
    swapped = _broadcast(tmp_path, rows=reordered, tags=["B", "A", "C"], name="swap")
    assert swapped == [_JOINED[1], _JOINED[0], _JOINED[2]]


def test_a_broadcast_join_refuses_a_repeated_or_partial_row_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate row_key"):
        _broadcast(tmp_path, rows=[_UNITS[0], _UNITS[0]])
    with pytest.raises(ValueError, match="row_key must name every by path"):
        _broadcast(tmp_path, fields=("row_key: rid",))


def test_a_broadcast_join_applies_on_missing_per_output_row(tmp_path: Path) -> None:
    """A further input missing a key leaves that key's rows out under null_fill
    and marks each of them, and only them, under fail_row."""
    only_u1 = [_SIZES[0]]
    missing = {**_UNITS[2], "tag": "C"}
    assert _broadcast(tmp_path, sizes=only_u1) == [_JOINED[0], _JOINED[1], missing]
    fail = ("row_key: [unit, rid]", "on_missing: fail_row")
    rows = _broadcast(tmp_path, sizes=only_u1, fields=fail, name="fail")
    assert rows[:2] == [_JOINED[0], _JOINED[1]]
    assert "key" in rows[2]["_fail"]["reason"]


def test_a_marked_broadcast_row_marks_exactly_its_keys_rows(tmp_path: Path) -> None:
    """A marked further-input row marks every output row of its key and leaves
    the other keys' rows as they were."""
    marked = [{"unit": "u1", "size": _MARK}, _SIZES[1]]
    assert _broadcast(tmp_path, sizes=marked) == [_MARK, _MARK, _JOINED[2]]


def test_a_broadcast_join_passes_a_marked_first_row_through_as_its_marker(
    tmp_path: Path,
) -> None:
    """A marked first-input row passes through as its marker whether the marker is
    a field of it or the whole row, and two such rows in one unit keep their own
    rows while the unit still takes the broadcast."""
    rows = [{"unit": "u1", "rid": _MARK}, _MARK, _UNITS[1], _UNITS[2]]
    expected = [_MARK, _MARK, {**_UNITS[1], "size": 2}, {**_UNITS[2], "size": 1}]
    assert _broadcast(tmp_path, rows=rows, attach=False, name="mark") == expected
