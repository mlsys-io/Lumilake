"""The SQL row-order rule at submit and the `rows` bounds at run time."""

import pytest

from lumilake import envs
from lumilake_server.dynamic.driver import DriverProtocolError, validate_library
from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraph, RuntimeGraphBuilder
from lumilake_server.runtime.runtime_manager import FlowmeshRuntimeManager
from lumilake_server.utils.retrieval_spec import RetrievalSpec


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")


def _graph(template: str, extra: str = "") -> Graph:
    template = template.replace("\n", "\\n")
    doc = (
        "name: order\ninputs:\n  Stock: [NVDA]\nops:\n"
        '  - id: "Retrieve"\n    op: DataRetrievalOp\n    inputs: [Stock]\n'
        f"    data_spec:\n      type: lumid\n      mode: sql\n{extra}"
        f'      template: "{template}"\n'
        '      params:\n        - label: symbol\n          node: "Stock"\n'
    )
    return Graph.from_json(parse_yaml_payload(doc)["order"]["graph"])


@pytest.mark.parametrize(
    ("template", "accepted"),
    [
        ("SELECT a FROM t ORDER BY a", True),
        ("WITH c AS (SELECT a FROM t ORDER BY a) SELECT a FROM c ORDER BY a", True),
        ("SELECT a FROM t UNION SELECT b FROM u ORDER BY 1", True),
        ("SELECT a FROM t WHERE b = 1 ORDER\n  BY a", True),
        ("(SELECT a FROM t ORDER BY a);", True),
        ("((SELECT a FROM t ORDER BY a))", True),
        ("SELECT a FROM t ORDER BY a; -- latest first", True),
        ("SELECT a FROM t", False),
        ("SELECT * FROM (SELECT a FROM t ORDER BY a) x", False),
        ("SELECT row_number() OVER (ORDER BY a) FROM t", False),
    ],
)
def test_only_a_top_level_order_counts(template: str, accepted: bool) -> None:
    if accepted:
        _graph(template)
        return
    with pytest.raises(ValueError, match="DataRetrievalOp '[^']*Retrieve[^']*'"):
        _graph(template)


def test_the_refusals_name_the_library_entry_and_the_field() -> None:
    spec = {"type": "lumid", "mode": "sql", "template": "SELECT 1"}
    with pytest.raises(DriverProtocolError, match="peers.*top-level ORDER BY"):
        validate_library({"peers": {"op": "DataRetrievalOp", "data_spec": spec}})
    with pytest.raises(ValueError, match="where: must be a mapping"):
        RetrievalSpec.load(3, "where")
    with pytest.raises(ValueError, match="where: type: Field required"):
        RetrievalSpec.load({"mode": "sql"}, "where")


def test_rows_one_checks_its_bound() -> None:
    with pytest.raises(ValueError, match="rows: Input should be 'one' or 'table'"):
        _graph("SELECT a FROM t", "      rows: all\n")
    runtime = RuntimeGraphBuilder().build(
        _graph("SELECT a FROM t", "      rows: one\n").compile(Stock=["NVDA"])
    )
    (node_id,) = [
        nid for nid, n in runtime.nodes.items() if n.task_type == "data_retrieval"
    ]
    manager = FlowmeshRuntimeManager()
    items = [{"index": 0, "rows": 1}, {"index": 3, "rows": 2}]
    with pytest.raises(RuntimeError, match=r"_Retrieve_.*\[3\] of this batch"):
        manager._assert_rows_one(runtime, node_id, {"items": items})
    manager._assert_rows_one(runtime, node_id, {"items": [{}]})


@pytest.mark.parametrize("template", ["SELECT a FROM t", "SELECT a FROM t ORDER BY a"])
def test_rows_table_carries_the_whole_result(template: str) -> None:
    runtime = RuntimeGraphBuilder().build(
        _graph(template, "      rows: table\n").compile(Stock=["NVDA"])
    )
    (node,) = [n for n in runtime.nodes.values() if n.task_type == "data_retrieval"]
    declared = node.data_spec.model_dump(include={"rows", "require_non_empty_rows"})
    assert declared == {"rows": "table", "require_non_empty_rows": False}


def test_rows_table_refuses_a_position_inside_it() -> None:
    _reader_graph("items.table.close")
    with pytest.raises(ValueError, match="read the whole value or"):
        _reader_graph("items.table.0.close")


def test_rows_is_declared_only_on_a_sql_retrieval() -> None:
    with pytest.raises(ValueError, match="'rows' is declared only on a mode: sql"):
        _reader_graph("items.table", mode="s3")


def _reader_graph(path: str, mode: str = "sql") -> RuntimeGraph:
    doc = (
        "name: order\ninputs:\n  Stock: [NVDA]\nops:\n"
        '  - id: "Retrieve"\n    op: DataRetrievalOp\n    inputs: [Stock]\n'
        f"    data_spec:\n      type: lumid\n      mode: {mode}\n      rows: table\n"
        '      template: "SELECT close FROM t"\n  - id: "Read"\n'
        "    op: LLMChatOp\n    inputs: [Retrieve]\n    config: {model: m}\n"
        '    messages: [{role: user, content: "close"}]\n'
        '    rowwise_template: "Read {close}"\n'
        f'    rowwise_columns: [{{label: close, node: Retrieve, path: "{path}"}}]\n'
    )
    graph = Graph.from_json(parse_yaml_payload(doc)["order"]["graph"])
    return RuntimeGraphBuilder().build(graph.compile(Stock=["NVDA"]))
