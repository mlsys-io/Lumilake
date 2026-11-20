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

_TURNS = """
  - id: PatientTurn
    op: LLMChatOp
    inputs: [State]
    messages:
      - {role: user, content: "Answer the patient."}
    config: {model: test/model}
    rowwise_template: "Reply to {text}"
    rowwise_columns: [{label: text, node: State, path: items.output.text}]
"""


def _parse(turns: str, join: str) -> dict[str, Any]:
    payload = _PRELUDE + turns + join
    return parse_yaml_payload(payload)["j"]


def _join_node(spec: dict[str, Any]) -> Any:
    graph = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    runtime = RuntimeGraphBuilder().build(graph)
    return next(
        node
        for node in runtime.nodes.values()
        if node.data_spec.get("mode") == join_step.JOIN_MODE
    )


_RUNTIME_MOUNT: dict[str, str] = {}


@pytest.fixture(scope="session", autouse=True)
def _task_runtime_mount(tmp_path_factory: pytest.TempPathFactory) -> None:
    path = tmp_path_factory.mktemp("join-runtime") / "lumilake-runtime.py"
    path.write_bytes(python_step.runtime_blob().body)
    _RUNTIME_MOUNT["path"] = str(path)


def _stage(tmp_path: Path, name: str, items: list[dict[str, Any]]) -> str:
    stage = tmp_path / name
    stage.mkdir()
    (stage / "results.json").write_text(
        json.dumps({"task_id": "t", "result": {"items": items}})
    )
    return str(stage)


def _run(tmp_path: Path, node: Any, upstream: dict[int, Any]) -> dict[str, Any]:
    """Run the generated step over one result list per plan entry."""
    plan = node.data_spec["plan"]
    stages: dict[str, str] = {}
    for index, items in upstream.items():
        name = f"{node.node_id}-{index}"
        stages[plan[index]["node"]] = _stage(tmp_path, name, items)
    code = join_step.join_source(
        node.node_id,
        node.data_spec["join"],
        plan,
        python_step.TASK_RUNTIME_MOUNT,
    ).replace(python_step.TASK_RUNTIME_MOUNT, _RUNTIME_MOUNT["path"])
    namespace: dict[str, Any] = {}
    exec(code, namespace)
    return namespace["main"](stages)


def test_join_merges_a_disjoint_branch_and_leaves_its_fields_out(
    tmp_path: Path,
) -> None:
    spec = _parse(
        "",
        """
  - id: Merge
    op: JoinOp
    inputs: [State, Route.diagnose]
    by: case_id
""",
    )
    node = _join_node(spec)
    out = _run(
        tmp_path,
        node,
        {0: [{"output": _ROWS}], 1: [{"output": [_ROWS[1]]}]},
    )
    rows = [item["output"] for item in out["items"]]
    assert [row["case_id"] for row in rows] == ["a", "b", "c"]
    assert rows[1]["text"] == _ROWS[1]["text"]
    assert set(rows[0]) == {"case_id", "text"}


def test_join_attaches_a_rowwise_chat_through_its_row_source(
    tmp_path: Path,
) -> None:
    spec = _parse(
        _TURNS,
        """
  - id: Merge
    op: JoinOp
    inputs: [State, PatientTurn]
    by: case_id
    attach: {patient_reply: PatientTurn}
""",
    )
    node = _join_node(spec)
    out = _run(
        tmp_path,
        node,
        {
            0: [{"output": _ROWS}],
            1: [{"output": ["for-a", "for-b", "for-c"]}],
            2: [{"output": _ROWS}],
        },
    )
    replies = {
        item["output"]["case_id"]: item["output"].get("patient_reply")
        for item in out["items"]
    }
    assert replies == {"a": "for-a", "b": "for-b", "c": "for-c"}


_TURNS_READ_TWO_NODES = """
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

_LIST_HOP_TURNS = """
  - id: Turns
    op: LambdaOp
    mode: list
    inputs: [State]
    fn_name: turns
    code: |
      def turns(c):
          return [row for row in c[0] if "READY" in row["text"]]
  - id: PatientTurn
    op: LLMChatOp
    inputs: [Route.patient]
    messages:
      - {role: user, content: "Reply to the patient"}
    config: {model: test/model}
    rowwise_template: "Reply to {case_id}"
    rowwise_columns: [{label: case_id, node: Turns, path: items.output.case_id}]
"""


@pytest.mark.parametrize(
    ("turns", "match"),
    [
        pytest.param(_TURNS_READ_TWO_NODES, "several nodes", id="two-nodes"),
        pytest.param(_LIST_HOP_TURNS, "not 1:1 over one row column", id="list-hop"),
    ],
)
def test_join_refuses_an_attach_it_cannot_pair_with_rows(
    turns: str, match: str
) -> None:
    spec = _parse(
        turns,
        """
  - id: Merge
    op: JoinOp
    inputs: [State, Route.patient, PatientTurn]
    by: case_id
    attach: {patient_reply: PatientTurn}
""",
    )
    with pytest.raises(ValueError, match=f"patient_reply.*{match}"):
        _join_node(spec)


def test_join_fails_on_a_duplicate_key(tmp_path: Path) -> None:
    spec = _parse(
        "",
        """
  - id: Merge
    op: JoinOp
    inputs: [State, Route.diagnose]
    by: case_id
""",
    )
    node = _join_node(spec)
    with pytest.raises(ValueError, match="duplicate key"):
        _run(
            tmp_path,
            node,
            {0: [{"output": [_ROWS[1], _ROWS[1]]}], 1: [{"output": [_ROWS[1]]}]},
        )


def test_join_missing_error_names_the_key(tmp_path: Path) -> None:
    spec = _parse(
        "",
        """
  - id: Merge
    op: JoinOp
    inputs: [State, Route.diagnose]
    by: case_id
    missing: error
""",
    )
    node = _join_node(spec)
    with pytest.raises(ValueError, match=r'key "a"'):
        _run(tmp_path, node, {0: [{"output": _ROWS}], 1: [{"output": [_ROWS[1]]}]})


_DOWNSTREAM = """
  - id: Merge
    op: JoinOp
    inputs: [State, Route.patient, PatientTurn]
    by: case_id
    attach: {patient_reply: PatientTurn}
  - id: Merged
    op: LambdaOp
    mode: list
    inputs: [Merge]
    fn_name: merged
    code: |
      def merged(c):
          return [row["case_id"] for row in c[0]]
outputs:
  - {name: merged, ref: Merge}
"""


def test_a_join_attaching_a_chat_feeds_a_list_lambda_and_keeps_its_output() -> None:
    spec = parse_yaml_payload(_PRELUDE + _TURNS + _DOWNSTREAM)["j"]
    runtime = RuntimeGraphBuilder().build(
        Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    )
    merge = next(
        node
        for node in runtime.nodes.values()
        if node.data_spec.get("mode") == join_step.JOIN_MODE
    )
    merged = next(
        node
        for node in runtime.nodes.values()
        if node is not merge and "Merged_" in node.node_id
    )
    (entry,) = merged.data_spec["plan"]
    assert entry == {"kind": "stage", "node": merge.node_id}
    assert merged.dependencies == (merge.node_id,)
    assert runtime.output_node_map == {merge.node_id: "merged"}
    assert runtime.output_paths[merge.node_id] == "items.output"


def test_join_output_reads_as_a_rowwise_node_column() -> None:
    spec = parse_yaml_payload(_PRELUDE + """
  - id: Merge
    op: JoinOp
    inputs: [State, Route.diagnose]
    by: case_id
  - id: Ask
    op: LLMChatOp
    inputs: [Merge]
    messages:
      - {role: user, content: "Go"}
    config: {model: test/model}
    rowwise_template: "{text}"
    rowwise_columns: [{label: text, node: Merge, path: items.output.text}]
""")["j"]
    runtime = RuntimeGraphBuilder().build(
        Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    )
    merge = next(
        node
        for node in runtime.nodes.values()
        if node.data_spec.get("mode") == join_step.JOIN_MODE
    )
    ask = next(
        node
        for node in runtime.nodes.values()
        if node.data_spec.get("type") == "dataframe"
    )
    (column,) = ask.data_spec["columns"]
    assert column["node"] == merge.node_id
    # A join's per-row fields sit under the python envelope, as a list
    # Lambda's do, not at the declared path.
    assert column["path"] == "value.items.output.text"


def test_join_missing_error_fails_on_a_further_input_row_the_first_lacks(
    tmp_path: Path,
) -> None:
    spec = _parse(
        "",
        """
  - id: Merge
    op: JoinOp
    inputs: [State, Route.diagnose]
    by: case_id
    missing: error
""",
    )
    node = _join_node(spec)
    stray = {"case_id": "z", "text": "not in the first input"}
    with pytest.raises(ValueError, match=r'key "z" is not in'):
        _run(
            tmp_path,
            node,
            {0: [{"output": [_ROWS[1]]}], 1: [{"output": [_ROWS[1], stray]}]},
        )


_NUMBERED_REPLY = """
  - id: Reply
    op: LambdaOp
    mode: row
    inputs: [State]
    fn_name: reply
    code: |
      def reply(row):
          return 1
"""


def test_join_keeps_a_row_lambda_attach_value_typed(tmp_path: Path) -> None:
    spec = _parse(
        _NUMBERED_REPLY,
        """
  - id: Merge
    op: JoinOp
    inputs: [State, Reply]
    by: case_id
    attach: {reply: Reply}
""",
    )
    node = _join_node(spec)
    reply = next(e for e in node.data_spec["plan"] if "Reply" in e["node"])
    assert reply["path"] == "value"
    out = _run(
        tmp_path,
        node,
        {
            0: [{"output": [_ROWS[1]]}],
            1: [{"output": "1", "value": 1}],
            2: [{"output": [_ROWS[1]]}],
        },
    )
    (row,) = out["items"]
    assert row["output"]["reply"] == 1


def test_join_reads_a_table_by_its_own_index_keys(tmp_path: Path) -> None:
    spec = _parse(
        "",
        """
  - id: Merge
    op: JoinOp
    inputs: [State]
    by: case_id
""",
    )
    node = _join_node(spec)
    table = {"case_id": {"0": "a", "2": "c"}, "text": {"0": "x", "2": "z"}}
    out = _run(
        tmp_path,
        node,
        {0: [{"output": [{"df": json.dumps(table)}]}]},
    )
    assert [item["output"] for item in out["items"]] == [
        {"case_id": "a", "text": "x"},
        {"case_id": "c", "text": "z"},
    ]
