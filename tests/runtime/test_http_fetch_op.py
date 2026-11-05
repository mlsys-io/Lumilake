"""Focused tests for the typed HttpFetchOp and its YAML form."""

import json
import textwrap
from pathlib import Path

import pytest

from lumilake import envs
from lumilake_server.graphs import Graph
from lumilake_server.ops import HttpFetchConfig, HttpFetchOp, LambdaOp, Op, as_output
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


def test_semantic_scholar_stage_sets_pacing_and_query_cache(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "semantic-scholar-cache"
    yaml_text = textwrap.dedent(f"""
        name: semantic_scholar_stage
        inputs:
          Query: ["graph retrieval", "scientific models"]
        ops:
          - id: SSSearch
            op: HttpFetchOp
            inputs: [Query]
            query: Query
            url: "https://api.semanticscholar.org/graph/v1/paper/search?query={{query}}"
            config:
              timeout_sec: 30
              retries: 2
              max_concurrency: 1
              request_delay_sec: 0.5
              cache_dir: "{cache_dir}"
              fail_open: true
        outputs:
          - name: candidates
            ref: SSSearch
        """)
    spec = parse_yaml_payload(yaml_text)["semantic_scholar_stage"]
    runtime_graph = RuntimeGraphBuilder().build(
        Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    )
    [node] = [node for node in runtime_graph.nodes.values() if node.task_type == "api"]

    assert node.api_spec["request_delay_sec"] == 0.5
    assert node.api_spec["cache_dir"] == str(cache_dir)
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
