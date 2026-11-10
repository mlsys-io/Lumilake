"""Focused tests for the typed HttpFetchOp and its YAML form."""

import json
import re
import textwrap
from pathlib import Path

import httpx
import pytest
import yaml

from lumilake import envs
from lumilake_server.common import GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    HttpFetchConfig,
    HttpFetchOp,
    LambdaOp,
    LLMChatOp,
    Op,
    OpMessage,
    as_output,
    input_placeholder,
)
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime import python_step
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder


@pytest.fixture(autouse=True)
def _runtime_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "test" + "-pat")


def test_http_fetch_op_serializes_and_round_trips() -> None:
    op = HttpFetchOp(
        url="http://fixture.test/search?query={query}",
        query=["first", "second"],
        method="POST",
        params={"query": "{query}", "limit": 10},
        body={"query": "{query}", "limit": 10},
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
    assert parsed.params == {"query": "{query}", "limit": 10}
    assert parsed.body == {"query": "{query}", "limit": 10}
    assert parsed.config.fail_open is False
    assert parsed.config.timeout_sec == 5
    assert parsed.config.retries == 1
    assert parsed.config.max_concurrency == 2


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        ({"timeout_sec": 0}, "timeout_sec"),
        ({"retries": -1}, "retries"),
        ({"retries": 11}, "retries"),
        ({"max_concurrency": 0}, "max_concurrency"),
        ({"max_concurrency": 9}, "max_concurrency"),
        ({"cache_dir": "/tmp/cache"}, "cache_dir"),
        ({"cache_dir": "worker/../cache"}, "cache_dir"),
    ],
)
def test_http_fetch_op_validates_config(config_kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        HttpFetchConfig(**config_kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"url": ""}, "non-empty url"),
        (
            {"url": "http://fixture.test/search", "params": {"limit": 10}},
            "url, params, or body",
        ),
        ({"method": "PUT"}, "must be GET or POST"),
        ({"method": "POST"}, "POST requires a body"),
    ],
)
def test_http_fetch_op_validates_request(kwargs: dict, message: str) -> None:
    fields = {
        "url": "http://fixture.test/search?query={query}",
        "query": ["first"],
        "params": {"query": "{query}"},
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
    assert fetch["config"] == {
        "headers": {"Accept": "application/json"},
        "timeout_sec": 12,
        "retries": 2,
        "max_concurrency": 4,
    }
    assert fetch["query"] == by_op["InputOp"]["_id"]
    assert fetch["_inputs"] == [by_op["InputOp"]["_id"]]


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            {"url": "http://fixture.test/search", "params": None},
            "requires '{query}' in url",
        ),
        ({"method": "PUT"}, "method must be GET or POST"),
        ({"method": "POST"}, "POST requires body"),
        ({"params": ["q"]}, "params must be a mapping"),
        ({"config": {"retries": 11}}, "retries must be between 0 and 10"),
        ({"config": {"timeout": 5}}, "config has unknown fields: ['timeout']"),
    ],
)
def test_http_fetch_yaml_errors_name_the_op(fields: dict, message: str) -> None:
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

    with pytest.raises(
        ValueError, match=f"HttpFetchOp 'Search' .*{re.escape(message)}"
    ):
        parse_yaml_payload(yaml_text)


def test_http_fetch_params_encode_like_requests() -> None:
    import requests

    query = 'a b,c&d "quoted"'
    params = {
        "query": query,
        "limit": 10,
        "pageSize": 10,
        "sort": "relevance",
        "fields": "title,year,summary",
    }
    expected = (
        requests.Request(
            "GET",
            "https://search.example/records",
            params=params,
        )
        .prepare()
        .url
    )
    spec = parse_yaml_payload(textwrap.dedent("""
            name: fetch_params
            inputs:
                Query: ['a b,c&d "quoted"']
            ops:
              - id: Search
                op: HttpFetchOp
                inputs: [Query]
                query: Query
                url: "https://search.example/records"
                params:
                  query: "{query}"
                  limit: 10
                  pageSize: 10
                  sort: "relevance"
                  fields: "title,year,summary"
            outputs:
              - name: candidates
                ref: Search
            """))["fetch_params"]
    runtime_graph = RuntimeGraphBuilder().build(
        Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    )
    [node] = [
        node for node in runtime_graph.nodes.values() if node.model == "http-fetch"
    ]

    assert node.api_spec["params"] == {
        **params,
        "query": "{{prompt}}",
    }
    assert "{{prompt}}" not in node.api_spec["url"]
    encoded = str(
        httpx.URL(
            node.api_spec["url"],
            params={**params, "query": query},
        )
    )
    assert encoded == expected


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
            params:
              query: "{query}"
              limit: 10
            body:
              query: "{query}"
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
    assert node.api_spec["params"] == {"query": "{{prompt}}", "limit": 10}
    assert node.api_spec["timeout_sec"] == 9
    assert node.api_spec["retries"] == 2
    assert node.api_spec["concurrency"] == 3


def test_http_fetch_default_fail_open_is_false(tmp_path: Path) -> None:
    op = HttpFetchOp(url="http://fixture.test/search?q={query}", query="query")
    assert op.config.fail_open is False


def test_http_fetch_fail_open_reaches_api_spec() -> None:
    yaml_text = textwrap.dedent("""
        name: fetch_candidates
        inputs:
          Query: ["graph retrieval"]
        ops:
          - id: Search
            op: HttpFetchOp
            inputs: [Query]
            query: Query
            url: "http://fixture.test/search?q={query}"
            config:
              fail_open: true
        outputs:
          - name: candidates
            ref: Search
        """)
    spec = parse_yaml_payload(yaml_text)["fetch_candidates"]
    runtime_graph = RuntimeGraphBuilder().build(
        Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    )
    [node] = [node for node in runtime_graph.nodes.values() if node.task_type == "api"]
    assert node.api_spec["response"]["fail_open"] is True


def test_http_fetch_headers_are_redacted_from_archived_graph() -> None:
    fetch = HttpFetchOp(
        url="http://fixture.test/search?q={query}",
        query=["graph retrieval"],
        config=HttpFetchConfig(headers={"x-api-key": "s2-secret-123"}),
    )
    compiled = Graph.from_ops([as_output("candidates", fetch)]).compile()
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    fetch_node = runtime_graph.nodes[fetch.id]

    assert fetch_node.api_spec["headers"] == {"x-api-key": "s2-secret-123"}
    assert "s2-secret-123" not in json.dumps(runtime_graph.serialize())


def test_query_stage_sets_pacing_and_query_cache() -> None:
    yaml_text = textwrap.dedent("""
        name: query_stage
        inputs:
          Query: ["graph retrieval", "scientific models"]
        ops:
          - id: Search
            op: HttpFetchOp
            inputs: [Query]
            query: Query
            url: "https://search.example/records?query={{query}}"
            config:
              timeout_sec: 30
              retries: 2
              max_concurrency: 1
              request_delay_sec: 0.5
              cache_dir: "query-cache"
              fail_open: true
        outputs:
          - name: candidates
            ref: Search
        """)
    spec = parse_yaml_payload(yaml_text)["query_stage"]
    runtime_graph = RuntimeGraphBuilder().build(
        Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    )
    [node] = [node for node in runtime_graph.nodes.values() if node.task_type == "api"]

    assert node.api_spec["request_delay_sec"] == 0.5
    assert node.api_spec["cache_dir"] == "query-cache"
    assert node.api_spec["concurrency"] == 1
    assert node.api_spec["response"]["fail_open"] is True


def test_fail_open_error_row_is_visible_to_downstream_runtime_node() -> None:
    yaml_text = textwrap.dedent("""
        name: fetch_then_consume
        inputs:
          Query: ["graph retrieval"]
        ops:
          - id: Search
            op: HttpFetchOp
            inputs: [Query]
            query: Query
            url: "http://fixture.test/search?q={query}"
            config:
              fail_open: true
          - id: Consume
            op: LLMChatOp
            inputs: [Search]
            messages:
              - role: user
                content: Search
            config:
              model: org/model
              api: {}
        outputs:
          - name: answer
            ref: Consume
        """)
    spec = parse_yaml_payload(yaml_text)["fetch_then_consume"]
    runtime_graph = RuntimeGraphBuilder().build(
        Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    )
    fetch = next(
        node for node in runtime_graph.nodes.values() if node.model == "http-fetch"
    )
    consumer = next(
        node
        for node in runtime_graph.nodes.values()
        if node.task_type == "api" and node is not fetch
    )

    assert fetch.api_spec["response"]["fail_open"] is True
    assert consumer.dependencies == (fetch.node_id,)
    assert fetch.output_spec is None


def test_llm_reads_fetch_response_body_column() -> None:
    fetch = HttpFetchOp(
        url="http://fixture.test/search?q={query}",
        query=["graph retrieval"],
        config=HttpFetchConfig(fail_open=True),
    )
    consume = LLMChatOp(
        [OpMessage(role="user", content=fetch)],
        config=GenerationConfig(model="org/model"),
    )
    compiled = Graph.from_ops([as_output("answer", consume)]).compile()
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    fetch_node = next(
        node for node in runtime_graph.nodes.values() if node.model == "http-fetch"
    )
    llm_node = runtime_graph.nodes[consume.id]

    columns = llm_node.data_spec["template"]["columns"]
    assert any(
        column.get("node") == fetch_node.node_id and column.get("path") == "items.json"
        for column in columns
    )


def test_http_fetch_op_feeds_lambda_with_per_row_response(tmp_path: Path) -> None:
    def collect(items):
        return [{f"row_{index}": value} for index, value in enumerate(items[0])]

    fetch = HttpFetchOp(
        url="http://fixture.test/search?q={query}",
        query=["first", "second"],
        config=HttpFetchConfig(fail_open=True),
    )
    consume = LambdaOp([fetch], fn=collect, mode="list")
    compiled = Graph.from_ops([as_output("rows", consume)]).compile()
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    fetch_node = runtime_graph.nodes[fetch.id]
    fetch_dir = tmp_path / fetch_node.node_id
    fetch_dir.mkdir()
    (fetch_dir / "results.json").write_text(
        json.dumps(
            {
                "result": {
                    "items": [
                        {"json": {"title": "first"}, "error": None},
                        {"json": None, "error": "503 Service Unavailable"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    consumer = next(
        node
        for node in runtime_graph.nodes.values()
        if node.task_type == python_step.TASK_TYPE
    )
    namespace: dict[str, object] = {"json": json}
    exec(consumer.to_flowmesh_node()["spec"]["code"], namespace)
    main = namespace["main"]
    assert callable(main)

    rows = main({fetch_node.node_id: str(fetch_dir)})["items"]

    assert rows == [
        {"output": {"row_0": {"json": {"title": "first"}, "error": None}}},
        {"output": {"row_1": {"json": None, "error": "503 Service Unavailable"}}},
    ]


def test_llm_row_lambda_fetch_chain_builds_in_dependency_order() -> None:
    def extract(result):
        return str(result)

    query = input_placeholder("Query")
    llm = LLMChatOp(
        [OpMessage(role="user", content=query)],
        config=GenerationConfig(model="org/model"),
    )
    lambda_op = LambdaOp([llm], extract)
    fetch = HttpFetchOp("http://fixture.test/search?q={query}", lambda_op)
    compiled = Graph.from_ops([as_output("results", fetch)]).compile(
        Query=["first", "second"]
    )
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    llm_node = runtime_graph.nodes[llm.id]
    lambda_node = runtime_graph.nodes[lambda_op.id]
    fetch_node = runtime_graph.nodes[fetch.id]

    order = runtime_graph.node_order
    assert order.index(llm_node.node_id) < order.index(lambda_node.node_id)
    assert order.index(lambda_node.node_id) < order.index(fetch_node.node_id)
    assert lambda_node.dependencies == (llm_node.node_id,)
    assert fetch_node.dependencies == (lambda_node.node_id,)
