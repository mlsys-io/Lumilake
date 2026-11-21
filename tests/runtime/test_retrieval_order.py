"""The SQL row-order rule at submit, the `rows: one` bound at run time, and the
lexicographic order of a prefix expansion."""

from unittest.mock import AsyncMock, patch

import pytest

from lumilake import envs
from lumilake_server.dynamic.driver import DriverProtocolError, validate_library
from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.runtime.runtime_manager import FlowmeshRuntimeManager
from lumilake_server.utils.lumid_data_client import alist_blob_keys

ORDERED = [
    "SELECT a FROM t ORDER BY a",
    "WITH c AS (SELECT a FROM t ORDER BY a) SELECT a FROM c ORDER BY a",
    "SELECT a FROM t UNION SELECT b FROM u ORDER BY 1",
    "SELECT a FROM t ORDER BY a;",
    "-- latest first\nSELECT a FROM t ORDER BY a",
    "SELECT a FROM t WHERE b = 1 ORDER\n  BY a",
    "(SELECT a FROM t ORDER BY a);",
    "((SELECT a FROM t ORDER BY a))",
    "SELECT a FROM t ORDER BY a; -- latest first",
]
UNORDERED = [
    "SELECT a FROM t",
    "SELECT * FROM (SELECT a FROM t ORDER BY a) x",
    "SELECT row_number() OVER (ORDER BY a) FROM t",
]


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")


def _graph(template: str, extra: str = "") -> Graph:
    """Load one retrieval whose only param is the Stock input, so the row-order
    check runs on the graph a submit loads."""
    template = template.replace("\n", "\\n")
    doc = (
        "name: order\ninputs:\n  Stock: [NVDA]\nops:\n"
        '  - id: "Retrieve"\n    op: DataRetrievalOp\n    inputs: [Stock]\n'
        f"    data_spec:\n      type: lumid\n      mode: sql\n{extra}"
        f'      template: "{template}"\n'
        '      params:\n        - label: symbol\n          node: "Stock"\n'
    )
    return Graph.from_json(parse_yaml_payload(doc)["order"]["graph"])


@pytest.mark.parametrize("template", ORDERED)
def test_a_defined_order_is_accepted(template: str) -> None:
    _graph(template)


@pytest.mark.parametrize("template", UNORDERED)
def test_only_a_top_level_order_by_counts(template: str) -> None:
    with pytest.raises(ValueError, match="DataRetrievalOp '.*_Retrieve_.*'"):
        _graph(template)


def test_the_rule_names_the_library_entry() -> None:
    library = {
        "peers": {
            "op": "DataRetrievalOp",
            "data_spec": {"type": "lumid", "mode": "sql", "template": "SELECT 1"},
        }
    }
    with pytest.raises(DriverProtocolError, match="peers.*top-level ORDER BY"):
        validate_library(library)


def test_rows_one_checks_its_bound() -> None:
    with pytest.raises(ValueError, match="'rows' must be 'one'"):
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


@pytest.mark.asyncio
async def test_prefix_expansion_is_lexicographic() -> None:
    page = {"objects": [{"key": "p/c"}, {"key": "p/a"}, {"key": "p/b"}]}
    with patch(
        "lumilake_server.utils.lumid_data_client.aget_json",
        new=AsyncMock(return_value=page),
    ):
        assert await alist_blob_keys(prefix="p/", recursive=True) == [
            "p/a",
            "p/b",
            "p/c",
        ]
