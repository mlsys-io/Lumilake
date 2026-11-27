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
        stage.mkdir(parents=True, exist_ok=True)
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


def test_a_join_that_joins_its_own_rows_sets_their_count() -> None:
    """A join over the job's own rows drops or appends rows under ``inner`` and
    ``full`` and emits a key's pairs under ``many_to_many``, so its output reads as
    a whole-run value there; any other join keeps one row per input row."""
    yaml = """
name: j
inputs:
  Cases: [a, b, c]
ops:
  - {id: Left, op: LambdaOp, mode: row, inputs: [Cases], fn_name: left,
     code: "def left(row):\\n    return {'case_id': row[0], 'text': row[0]}\\n"}
  - {id: Right, op: LambdaOp, mode: row, inputs: [Cases], fn_name: right,
     code: "def right(row):\\n    return {'case_id': row[0], 'n': 1}\\n"}
"""
    cases = (
        ("left", "", False),
        ("inner", "", True),
        ("full", "", True),
        ("left", "    cardinality: many_to_many\n", True),
    )
    for how, extra, whole_run in cases:
        spec = parse_yaml_payload(
            yaml + f"  - id: Merge\n    op: JoinOp\n    inputs: [Left, Right]\n    by:"
            f" case_id\n    how: {how}\n{extra}"
        )["j"]
        runtime = RuntimeGraphBuilder().build(
            Graph.from_json(spec["graph"]).compile(**spec["inputs"]),
            node_prefix="g",
        )
        assert (_join_of(runtime).node_id in runtime.own_row_count_nodes()) is whole_run


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


def test_fail_row_leaves_a_key_a_further_input_lacks_alone(tmp_path: Path) -> None:
    """on_missing governs an attach source: a further input's gap keeps the row."""
    node = _join_node(_merge("State, Route.diagnose", "    on_missing: fail_row\n"))
    out = _run(tmp_path, node, {0: [{"output": _ROWS}], 1: [{"output": [_ROWS[1]]}]})
    assert [item["output"] for item in out["items"]] == _ROWS


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


def test_join_reads_a_workflow_input_as_its_first_column(tmp_path: Path) -> None:
    node = _join_node(_merge("Rows"))
    assert node.data_spec.plan[0].kind == "literal"
    out = _run(tmp_path, node, {})
    assert [item["output"]["case_id"] for item in out["items"]] == ["a", "b", "c"]


def test_join_refuses_an_empty_key_or_one_that_is_not_a_field_name() -> None:
    for by, match in [("[]", "requires 'by'"), ("[case_id, 3]", "dotted field")]:
        with pytest.raises(ValueError, match=f"Merge.*{match}"):
            _join_node(f"  - {{id: Merge, op: JoinOp, inputs: [State], by: {by}}}\n")


_MARK = {"_fail": {"op": "P", "reason": "no JSON object"}}
_DIAG = _merge("State, Route.diagnose")
_ATTACH = _REPLY + _merge("State, Reply", "    attach: {reply: Reply}\n")
_COMPOSITE = _merge("State, Route.diagnose", by="[meta.unit, meta.kind]")
# A row Lambda over the diagnose branch, so its values reach only that key.
_PART = _REPLY.replace("[State]", "[Route.diagnose]") + _merge(
    "State, Route.diagnose, Reply", "    attach: {part: Reply}\n"
)
_A, _B, _C = _ROWS
_META = {"meta": {"unit": "u1", "kind": "x"}}
_META_Y = {"meta": {"unit": "u1", "kind": "y"}}
_META_SIZED = [{**_META, "s": 1}, {**_META_Y, "s": 2}]
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
        # further input the key's output is. A composite key reads each path, so a
        # marked row whose path reaches a marker has no key to be placed by.
        (_DIAG, {0: _rows(_A, _MARK, _B), 1: _rows(_B)}, [_A, _MARK, _B]),
        (_COMPOSITE, {0: _rows(_META), 1: _rows({**_META, "s": _MARK})}, [_MARK]),
        # A composite key keeps rows that share its first field apart.
        (_COMPOSITE, {0: _rows(_META, _META_Y), 1: _rows(*_META_SIZED)}, _META_SIZED),
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


def _quoted(rows: list[Any]) -> str:
    return "[" + ", ".join(f"'{json.dumps(row)}'" for row in rows) + "]"


def _pairs(cases: list[Any], girls: list[Any], fields: str) -> Any:
    """A join of two literal columns keyed by ``unit``, under ``fields``."""
    return _join_node(
        f"  - {{id: Cases, op: DataOp, data: {_quoted(cases)}}}\n"
        f"  - {{id: Girls, op: DataOp, data: {_quoted(girls)}}}\n"
        f"  - {{id: Merge, op: JoinOp, inputs: [Cases, Girls], by: unit, {fields}}}\n"
    )


_CASES = [{"unit": "u1", "q": "a"}, {"unit": "u1", "q": "b"}, {"unit": "u2", "q": "c"}]
_GIRLS = [{"unit": "u1", "g": "x"}, {"unit": "u1", "g": "y"}, {"unit": "u2", "g": "z"}]
_PRODUCT = [("a", "x"), ("a", "y"), ("b", "x"), ("b", "y"), ("c", "z")]


def test_a_many_to_many_join_emits_the_key_product(tmp_path: Path) -> None:
    """A key's rows pair held side first, and a stray key is its input's rows once."""

    def run(girls: list[Any], how: str, name: str) -> list[tuple[Any, Any]]:
        node = _pairs(_CASES, girls, f"cardinality: many_to_many, how: {how}")
        rows = [item["output"] for item in _run(tmp_path, node, {}, name)["items"]]
        return [(row.get("q"), row["g"]) for row in rows]

    assert run(_GIRLS, "left", "left") == _PRODUCT
    stray = [{"unit": "u3", "g": "x"}, {"unit": "u3", "g": "y"}]
    assert run(_GIRLS + stray, "full", "full") == _PRODUCT + [(None, g) for g in "xy"]


def test_a_cardinality_and_a_row_key_declare_each_other(tmp_path: Path) -> None:
    """A repeat needs cardinality and row_key together, and a product no attach."""
    one, many = "cardinality: one_to_many", "cardinality: many_to_many"
    cases = [
        (one, "one_to_many also needs row_key"),
        ("row_key: [unit, q]", "row_key needs cardinality one_to_many"),
        (many + ", attach: {g: Girls}", "reads no attach"),
        ("", "declare one_to_many with a row_key"),
    ]
    for fields, match in cases:
        with pytest.raises(ValueError, match=match):
            _run(tmp_path, _pairs(_CASES, _GIRLS, fields), {})


def test_one_to_many_keeps_every_row_of_a_repeated_key(tmp_path: Path) -> None:
    """Each first-input row of a key keeps its own output row, in its order."""
    node = _pairs(
        _CASES[:2], _GIRLS[:1], "cardinality: one_to_many, row_key: [unit, q]"
    )
    rows = [item["output"] for item in _run(tmp_path, node, {})["items"]]
    assert [(row["q"], row["g"]) for row in rows] == [("a", "x"), ("b", "x")]


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


_UNITS = [{"unit": "u1", "rid": r} for r in "ab"] + [{"unit": "u2", "rid": "c"}]
_SIZES = [{"unit": "u1", "size": 2}, {"unit": "u2", "size": 1}]
_SIZE = {size["unit"]: size["size"] for size in _SIZES}
_JOINED = [{**u, "size": _SIZE[u["unit"]], "tag": u["rid"].upper()} for u in _UNITS]
_ROW_KEY = "cardinality: one_to_many, row_key: [unit, rid]"


def _broadcast(
    tmp_path: Path,
    rows: list[Any] | None = None,
    sizes: list[Any] | None = None,
    second: list[Any] | None = None,
    part: list[Any] | None = None,
    tags: list[Any] | None = None,
    fields: str = _ROW_KEY,
) -> list[Any]:
    """Run a one_to_many, row_key join of the units with a per-row attach."""
    extra, more, inputs, attach = "", "", ", Tag", "tag: Tag"
    if second is not None:
        extra = f"\n  - {{id: Second, op: DataOp, data: {_quoted(second)}}}"
        more = ", Second"
    if part is not None:
        extra = f"\n  - {{id: Extra, op: DataOp, data: {_quoted(part)}}}"
        extra += "\n  - {id: Part, op: LambdaOp, mode: row, inputs: [Extra],"
        extra += " fn_name: part, code: \"def part(row): return row['part']\"}"
        inputs, attach = ", Extra, Part", "part: Part"
    code = (
        "name: u\ninputs: {Rows: []}\nops:\n"
        f"  - {{id: State, op: DataOp, data: {_quoted(rows or _UNITS)}}}\n"
        f"  - {{id: Summary, op: DataOp, data: {_quoted(sizes or _SIZES)}}}{extra}\n"
        "  - {id: Tag, op: LambdaOp, mode: row, inputs: [State], fn_name: tag,\n"
        "     code: \"def tag(row): return row['rid'].upper()\"}\n"
        f"  - {{id: Merge, op: JoinOp, inputs: [State{inputs}, Summary{more}], by:"
        f" unit, {fields}, attach: {{{attach}}}}}\n"
        "outputs: [{name: merged, ref: Merge}]\n"
    )
    spec = parse_yaml_payload(code)["u"]
    graph = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    node = _join_of(RuntimeGraphBuilder().build(graph))
    _, value_index, *_ = node.data_spec.join.attach[0]
    staged = {value_index: _cells(*(tags or ["A", "B", "C"]))}
    return [item["output"] for item in _run(tmp_path, node, staged)["items"]]


def test_a_broadcast_join_keeps_one_output_row_per_record(tmp_path: Path) -> None:
    """Each row of a shared by key keeps its own output row, in its order."""
    assert _broadcast(tmp_path) == _JOINED
    with pytest.raises(ValueError, match="duplicate row_key"):
        _broadcast(tmp_path, [_UNITS[0], _UNITS[0]])
    with pytest.raises(ValueError, match="row_key must name every by path"):
        _broadcast(tmp_path, fields="cardinality: one_to_many, row_key: rid")
    with pytest.raises(ValueError, match="repeats in a further input") as caught:
        _broadcast(tmp_path, sizes=_SIZES + [_SIZES[0]])
    assert "many_to_many" not in str(caught.value)


def test_a_broadcast_join_keeps_the_keys_its_how_names(tmp_path: Path) -> None:
    """inner drops a missing key; full appends a key once, or its marker."""
    fields = _ROW_KEY + ", how: "
    inner = _broadcast(
        tmp_path, [_UNITS[0], _UNITS[2], _UNITS[1]], _SIZES[:1], fields=fields + "inner"
    )
    assert [row["unit"] + row["rid"] for row in inner] == ["u1a", "u1b"]
    stray = {"unit": "u3", "size": 1}
    full = fields + "full"
    assert _broadcast(
        tmp_path, sizes=_SIZES + [stray], second=[stray], fields=full
    ) == (_JOINED + [stray])
    marked = [{"unit": "u4", "n": _MARK}]
    bad = [{"unit": "u4", "n": 2}]
    out = _broadcast(tmp_path, sizes=marked, second=bad, tags=3 * ["A"], fields=full)
    assert out[-1] == _MARK


def test_a_broadcast_join_applies_on_missing_to_an_attach_source_only(
    tmp_path: Path,
) -> None:
    """fail_row marks only the rows an attach source has no key for."""
    missing = {**_UNITS[2], "tag": "C"}
    fields = _ROW_KEY + ", on_missing: fail_row"
    for policy in (_ROW_KEY, fields):  # the further input's gap is not a failure
        kept = _broadcast(tmp_path, sizes=_SIZES[:1], fields=policy)
        assert kept == [_JOINED[0], _JOINED[1], missing]
    parts = _broadcast(tmp_path, part=[{"unit": "u1", "part": "P"}], tags=["P"])
    assert [row.get("part") for row in parts] == ["P", "P", None]
    marked = [{"unit": "u1", "size": _MARK}, _SIZES[1]]
    assert _broadcast(tmp_path, sizes=marked) == [_MARK, _MARK, _JOINED[2]]
    src = [{"unit": "u1", "part": _MARK}, {"unit": "u2"}]
    out = _broadcast(tmp_path, part=src, tags=["C"])
    assert out == [_MARK, _MARK, {**_UNITS[2], "size": 1, "part": "C"}]


def test_a_marked_broadcast_row_keeps_a_readable_key(tmp_path: Path) -> None:
    """A marked first row whose key stays readable is placed by it."""
    rows = [_UNITS[0], {**_UNITS[2], "text": _MARK}]
    match = _ROW_KEY + ", require_match: true"
    with pytest.raises(ValueError, match=r'key "u3" is not in'):
        _broadcast(tmp_path, sizes=_SIZES + [{"unit": "u3", "size": 1}], fields=match)
    assert _broadcast(tmp_path, rows, tags=["A", "C"], fields=match) == [
        _JOINED[0],
        _MARK,
    ]
    stray = {"unit": "u3", "size": 1}
    fields = _ROW_KEY + ", how: full"
    full = _broadcast(tmp_path, rows, _SIZES + [stray], tags=["A", "C"], fields=fields)
    assert full == [_JOINED[0], _MARK, stray]
