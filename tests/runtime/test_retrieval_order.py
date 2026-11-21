"""A retrieval's row order: the submit-time SQL rule, the run-time `rows: one`
bound, and the lexicographic order of a prefix expansion.
"""

from unittest.mock import AsyncMock, patch

import pytest

from lumilake import envs
from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.runtime_graph import (
    RuntimeGraph,
    RuntimeGraphBuilder,
)
from lumilake_server.runtime.runtime_manager import FlowmeshRuntimeManager
from lumilake_server.utils.lumid_data_client import alist_blob_keys


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")


def _graph(template: str, extra: str = "") -> Graph:
    """Parse one retrieval whose only param is the workflow's Stock input, and
    load the graph a submit loads so the node-level row-order check runs."""
    body = "\n".join(f"        {line}" if line else "" for line in template.split("\n"))
    doc = (
        "name: order\ninputs:\n  Stock: [NVDA]\nops:\n"
        '  - id: "Retrieve"\n    op: DataRetrievalOp\n    inputs: [Stock]\n'
        f"    data_spec:\n      type: lumid\n      mode: sql\n{extra}"
        f"      template: |\n{body}\n"
        '      params:\n        - label: symbol\n          node: "Stock"\n'
        'outputs:\n  - name: out\n    ref: "Retrieve"\n'
    )
    return Graph.from_json(parse_yaml_payload(doc)["order"]["graph"])


@pytest.mark.parametrize(
    "template",
    [
        "SELECT a FROM t",
        "SELECT * FROM (SELECT a FROM t ORDER BY a) x",
        "SELECT row_number() OVER (ORDER BY a) FROM t",
        "-- latest first\nSELECT a FROM t",
        "SELECT a FROM t; SELECT b FROM u",
    ],
    ids=["no-order", "subquery-order", "window-order", "no-order-comment", "last-part"],
)
def test_only_the_outer_order_by_counts(template: str) -> None:
    with pytest.raises(ValueError, match="DataRetrievalOp '.*_Retrieve_.*'"):
        _graph(template)


@pytest.mark.parametrize(
    "template",
    [
        "SELECT a FROM t ORDER BY a",
        "WITH c AS (SELECT a FROM t ORDER BY a)\nSELECT a FROM c ORDER BY a",
        "SELECT a FROM t UNION SELECT b FROM u ORDER BY 1",
        "SELECT a FROM t ORDER BY a;",
        "-- latest first\nSELECT a FROM t ORDER BY a",
        "SELECT a FROM t WHERE b = 1\nORDER\n  BY a",
        "SELECT a FROM t WHERE b = 1 ORDER\n  BY a",
        "(SELECT a FROM t ORDER BY a);",
        "SELECT a FROM t ORDER BY a; -- latest first",
    ],
    ids=[
        "top-level",
        "cte-outer",
        "union",
        "trailing-semicolon",
        "leading-comment",
        "order-by-newline",
        "order-by-newline-after-where",
        "parenthesized",
        "trailing-comment",
    ],
)
def test_a_defined_order_is_accepted(template: str) -> None:
    _graph(template)


def test_rows_value_is_checked() -> None:
    with pytest.raises(ValueError, match="'rows' must be 'one'"):
        _graph("SELECT a FROM t", "      rows: all\n")


def test_rows_one_fails_only_a_parameter_row_with_two_rows() -> None:
    runtime: RuntimeGraph = RuntimeGraphBuilder().build(
        _graph("SELECT a FROM t", "      rows: one\n").compile(Stock=["NVDA"])
    )
    (node_id,) = [
        nid for nid, n in runtime.nodes.items() if n.task_type == "data_retrieval"
    ]
    response = {"items": [{"index": 0, "rows": 1}, {"index": 3, "rows": 2}]}
    with pytest.raises(RuntimeError, match=r"_Retrieve_.*\[3\] of this batch"):
        FlowmeshRuntimeManager()._assert_rows_one(runtime, node_id, response)
    FlowmeshRuntimeManager()._assert_rows_one(
        runtime, node_id, {"items": [{"index": 0, "rows": 1}]}
    )


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
