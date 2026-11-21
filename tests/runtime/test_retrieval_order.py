"""A retrieval's row order: the submit-time SQL rule, the run-time `rows: one`
bound, and the lexicographic order of a prefix expansion.
"""

from unittest.mock import AsyncMock, patch

import pytest

from lumilake import envs
from lumilake_server.graphs import Graph
from lumilake_server.ops import DataRetrievalOp
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
    doc = (
        "name: order\ninputs:\n  Stock: [NVDA]\nops:\n"
        '  - id: "Retrieve"\n    op: DataRetrievalOp\n    inputs: [Stock]\n'
        f"    data_spec:\n      type: lumid\n      mode: sql\n{extra}"
        f'      template: "{template}"\n'
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
    ],
    ids=["no-order", "subquery-order", "window-order"],
)
def test_only_the_outer_order_by_counts(template: str) -> None:
    with pytest.raises(ValueError, match="DataRetrievalOp '.*_Retrieve_.*'"):
        _graph(template)


def test_rows_one_survives_the_native_round_trip() -> None:
    """A declared single-row lookup needs no ORDER BY, and keeps its
    declaration through serialization and back."""
    graph = _graph("SELECT a FROM t", "      rows: one\n")
    native = Graph.from_json(
        {oid: op.serialize() for oid, op in graph.as_dict().items()}
    )
    (op,) = [o for o in native.as_dict().values() if isinstance(o, DataRetrievalOp)]
    assert op.data_spec["rows"] == "one"


def test_rows_value_is_checked() -> None:
    with pytest.raises(ValueError, match="'rows' must be 'one'"):
        _graph("SELECT a FROM t", "      rows: all\n")


def _rows_one_runtime() -> tuple[RuntimeGraph, str]:
    runtime = RuntimeGraphBuilder().build(
        _graph("SELECT a FROM t", "      rows: one\n").compile(Stock=["NVDA"])
    )
    (node_id,) = [
        nid for nid, n in runtime.nodes.items() if n.task_type == "data_retrieval"
    ]
    return runtime, node_id


def test_rows_one_fails_a_parameter_row_with_two_rows() -> None:
    runtime, node_id = _rows_one_runtime()
    response = {"items": [{"index": 0, "rows": 1}, {"index": 3, "rows": 2}]}
    with pytest.raises(RuntimeError, match=r"_Retrieve_.*\[3\]"):
        FlowmeshRuntimeManager()._assert_rows_one(runtime, node_id, response)


def test_rows_one_accepts_one_row_per_parameter_row() -> None:
    runtime, node_id = _rows_one_runtime()
    FlowmeshRuntimeManager()._assert_rows_one(
        runtime, node_id, {"items": [{"index": 0, "rows": 1}]}
    )


@pytest.mark.asyncio
async def test_prefix_expansion_is_lexicographic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
