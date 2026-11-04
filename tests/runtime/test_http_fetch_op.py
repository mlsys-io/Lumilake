"""Focused tests for the typed HttpFetchOp and its YAML form."""

import textwrap

import pytest

from lumilake_server.graphs import Graph
from lumilake_server.ops import HttpFetchConfig, HttpFetchOp, Op
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder


def test_http_fetch_op_serializes_and_round_trips() -> None:
    op = HttpFetchOp(
        url="http://fixture.test/search?query={query}",
        query=["first", "second"],
        method="POST",
        body={"query": "{query}", "limit": 10},
        response_path="data",
        config=HttpFetchConfig(
            headers={"Accept": "application/json"},
            timeout_sec=5,
            retries=1,
            max_concurrency=2,
        ),
    )
    payload = op.serialize()
    parsed = Op.registry.get_class("HttpFetchOp").from_json(
        payload, {op.query.id: op.query}
    )

    assert isinstance(parsed, HttpFetchOp)
    assert parsed.url == op.url
    assert parsed.method == "POST"
    assert parsed.body == {"query": "{query}", "limit": 10}
    assert parsed.response_path == "data"
    assert parsed.config.timeout_sec == 5
    assert parsed.config.retries == 1
    assert parsed.config.max_concurrency == 2


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        ({"timeout_sec": 0}, "timeout_sec"),
        ({"retries": -1}, "retries"),
        ({"max_concurrency": 0}, "max_concurrency"),
        ({"max_concurrency": 9}, "max_concurrency"),
    ],
)
def test_http_fetch_op_validates_config(config_kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        HttpFetchConfig(**config_kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"url": ""}, "non-empty url"),
        ({"url": "http://fixture.test/search"}, "\\{query\\} placeholder"),
        ({"method": "PUT"}, "must be GET or POST"),
        ({"method": "POST"}, "POST requires a body"),
    ],
)
def test_http_fetch_op_validates_request(kwargs: dict, message: str) -> None:
    fields = {
        "url": "http://fixture.test/search?query={query}",
        "query": ["first"],
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        HttpFetchOp(**fields)


def test_http_fetch_yaml_parses_and_builds_expected_op_dict() -> None:
    yaml_text = textwrap.dedent("""
        name: fetch_candidates
        inputs:
          Query: ["graph retrieval", "scientific models"]
        ops:
          - id: Search
            op: HttpFetchOp
            inputs: [Query]
            query: Query
            url: "http://fixture.test/search?query={query}&limit=10"
            method: GET
            response_path: data
            config:
              headers:
                Accept: application/json
              timeout_sec: 12
              retries: 2
              max_concurrency: 4
        outputs:
          - name: candidates
            ref: Search
        """)

    spec = parse_yaml_payload(yaml_text)["fetch_candidates"]
    by_op = {entry["_op"]: entry for entry in spec["graph"].values()}
    fetch = by_op["HttpFetchOp"]

    assert fetch["url"] == "http://fixture.test/search?query={query}&limit=10"
    assert fetch["method"] == "GET"
    assert fetch["response_path"] == "data"
    assert fetch["config"] == {
        "headers": {"Accept": "application/json"},
        "timeout_sec": 12,
        "retries": 2,
        "max_concurrency": 4,
    }
    assert fetch["query"] == by_op["InputOp"]["_id"]
    assert fetch["_inputs"] == [by_op["InputOp"]["_id"]]


def test_http_fetch_yaml_lowers_to_flowmesh_api_node() -> None:
    yaml_text = textwrap.dedent("""
        name: fetch_candidates
        inputs:
          Query: ["graph retrieval", "scientific models"]
        ops:
          - id: Search
            op: HttpFetchOp
            inputs: [Query]
            query: Query
            url: "http://fixture.test/paper/search/{query}?q={query}"
            method: POST
            body:
              query: "{query}"
            response_path: data
            config:
              timeout_sec: 9
              retries: 2
              max_concurrency: 3
        outputs:
          - name: candidates
            ref: Search
        """)
    spec = parse_yaml_payload(yaml_text)["fetch_candidates"]
    runtime_graph = RuntimeGraphBuilder().build(
        Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    )
    [node] = [node for node in runtime_graph.nodes.values() if node.task_type == "api"]

    assert node.data_spec == {
        "type": "list",
        "items": ["graph retrieval", "scientific models"],
    }
    assert node.api_spec["url"] == (
        "http://fixture.test/paper/search/{{prompt}}?q={{prompt}}"
    )
    assert node.api_spec["json"] == {"query": "{{prompt}}"}
    assert node.api_spec["timeout_sec"] == 9
    assert node.api_spec["retries"] == 2
    assert node.api_spec["concurrency"] == 3
