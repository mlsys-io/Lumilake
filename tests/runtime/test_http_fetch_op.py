"""Tests for the typed HttpFetchOp and its YAML form."""

import json
import textwrap

import pytest
import yaml

from lumilake import envs
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    DataRetrievalOp,
    HttpFetchOp,
    as_output,
    input_placeholder,
)
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder


@pytest.fixture(autouse=True)
def _envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test" + "-token")


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"url": ""}, "non-empty url"),
        ({"url": "http://x.test/s", "params": None}, "requires '{query}'"),
        ({"method": "PUT"}, "method must be GET or POST"),
        ({"method": "POST"}, "POST requires a body"),
        ({"params": ["q"]}, "params must be a mapping"),
        ({"config": {"timeout_sec": 0}}, "timeout_sec must be greater than zero"),
        ({"config": {"retries": 11}}, "retries must be between 0 and 10"),
        ({"config": {"timeout": 5}}, r"config has unknown fields: \['timeout'\]"),
        ({"config": {"fail_open": True}}, "fail_open needs a flowmesh-sdk"),
    ],
)
def test_http_fetch_yaml_errors_name_the_op(fields: dict, message: str) -> None:
    """The parser runs the constructor's request and config checks, naming the op."""
    entry = {
        "id": "Search",
        "op": "HttpFetchOp",
        "inputs": ["Query"],
        "query": "Query",
        "url": "http://fixture.test/search?query={query}",
        "params": {"query": "{query}"},
        **fields,
    }
    yaml_text = yaml.safe_dump(
        {
            "name": "fetch_errors",
            "inputs": {"Query": ["first"]},
            "ops": [entry],
            "outputs": [{"name": "candidates", "ref": "Search"}],
        }
    )

    with pytest.raises(ValueError, match=rf"HttpFetchOp 'Search' .*{message}"):
        parse_yaml_payload(yaml_text)


def test_http_fetch_op_yaml_lowers_to_an_api_node() -> None:
    yaml_text = textwrap.dedent("""
        name: fetch_candidates
        inputs: {Query: ["graph retrieval"]}
        ops:
          - {id: Search, op: HttpFetchOp, inputs: [Query], query: Query,
             url: "http://fixture.test/paper/search/{query}?q={query}", method: POST,
             params: {query: "{query}"}, body: {query: "{query}"},
             config: {timeout_sec: 9, retries: 2, max_concurrency: 3,
                      request_delay_sec: 0.5,
                      headers: {Accept: application/json, x-api-key: s3cret-key}}}
        outputs: [{name: candidates, ref: Search}]
        """)
    spec = parse_yaml_payload(yaml_text)["fetch_candidates"]
    runtime_graph = RuntimeGraphBuilder().build(
        Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    )
    [node] = [node for node in runtime_graph.nodes.values() if node.task_type == "api"]

    assert node.data_spec == {"type": "list", "items": ["graph retrieval"]}
    assert node.api_spec == {
        "method": "POST",
        "url": "http://fixture.test/paper/search/{{prompt}}?q={{prompt}}",
        "headers": {"Accept": "application/json", "x-api-key": "s3cret-key"},
        "timeout_sec": 9,
        "retries": 2,
        "concurrency": 3,
        "request_delay_sec": 0.5,
        "params": {"query": "{{prompt}}"},
        "json": {"query": "{{prompt}}"},
        "response": {
            "parse_json": True,
            "return_body": True,
            "raise_for_status": True,
            "fail_open": False,
        },
    }
    assert "s3cret-key" not in json.dumps(runtime_graph.serialize())


def test_http_fetch_op_reads_its_upstream_result_path() -> None:
    """A fetch over a retrieval reads its table rows, not its ``output``."""
    stock = input_placeholder("Stock")
    retrieval = DataRetrievalOp(
        data_spec={
            "type": "lumid",
            "mode": "sql",
            "template": "SELECT q FROM t WHERE symbol = :symbol",
            "params": [{"name": "symbol", "node": stock.id}],
        },
        inputs=[stock],
    )
    fetch = HttpFetchOp(url="http://fixture.test/search?q={query}", query=retrieval)
    compiled = Graph.from_ops([as_output("candidates", fetch)]).compile(Stock=["NVDA"])
    node = RuntimeGraphBuilder().build(compiled).nodes[fetch.id]

    assert node.data_spec == {
        "type": "list",
        "node": retrieval.id,
        "path": "items.table",
    }
    assert node.dependencies == (retrieval.id,)
