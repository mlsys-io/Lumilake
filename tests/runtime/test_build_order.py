"""Derived ops (fetch, score and list Lambda) build after the LLM pass, in an
order that satisfies every chain, whatever mix of op kinds it holds."""

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from lumilake import envs
from lumilake_server.common import GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    HttpFetchOp,
    LambdaOp,
    LLMChatOp,
    OpMessage,
    ScoreConfig,
    ScoreOp,
    as_output,
    input_placeholder,
)
from lumilake_server.runtime.runtime_graph import RuntimeGraph, RuntimeGraphBuilder

_MODEL = GenerationConfig(model="org/model-8B")
_SCORER = ScoreConfig(model="org/reranker-0.5B")
_URL = "https://search.example/api?q={query}"


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")


def _first_line(x: Any) -> str:
    return str(x).splitlines()[0]


def _titles(docs: Any) -> list[str]:
    return [str(d) for d in docs]


def _llm(query: Any) -> LLMChatOp:
    return LLMChatOp([OpMessage(role="user", content=query)], config=_MODEL)


def _chain(tail: Callable[[Any, Any], Any]) -> Callable[[], Any]:
    def build() -> Any:
        query = input_placeholder("Query")
        return tail(query, _llm(query))

    return build


def _fetch(feed: Any) -> HttpFetchOp:
    return HttpFetchOp(_URL, feed)


def _score(query: Any, candidates: Any) -> ScoreOp:
    return ScoreOp(query, candidates, _SCORER)


def _rowwise_llm(fetch: HttpFetchOp) -> LLMChatOp:
    return LLMChatOp(
        [OpMessage(role="user", content=fetch)],
        config=_MODEL,
        rowwise_template="{doc}",
        rowwise_columns=[{"label": "doc", "node": fetch.id, "path": "items.output"}],
    )


_CHAINS: dict[str, Callable[[], Any]] = {
    "fetch_row_lambda_score": lambda: (
        lambda q: _score(q, LambdaOp([_fetch(q)], _first_line))
    )(input_placeholder("Query")),
    "fetch_list_lambda_score": lambda: (
        lambda q: _score(q, LambdaOp([_fetch(q)], _titles, mode="list"))
    )(input_placeholder("Query")),
    "llm_row_lambda_fetch": _chain(lambda q, llm: _fetch(LambdaOp([llm], _first_line))),
    "llm_score": _chain(lambda q, llm: _score(q, llm)),
    "llm_score_list_lambda": _chain(
        lambda q, llm: LambdaOp([_score(q, llm)], _titles, mode="list")
    ),
    "llm_score_fetch": _chain(lambda q, llm: _fetch(_score(q, llm))),
    "llm_fetch": _chain(lambda q, llm: _fetch(llm)),
    "llm_list_lambda_fetch": _chain(
        lambda q, llm: _fetch(LambdaOp([llm], _titles, mode="list"))
    ),
    "llm_row_lambda_score": _chain(
        lambda q, llm: _score(q, LambdaOp([llm], _first_line))
    ),
    "llm_fetch_llm": _chain(lambda q, llm: _llm(_fetch(llm))),
    "llm_fetch_rowwise_llm": lambda: (lambda llm: _rowwise_llm(_fetch(llm)))(
        input_placeholder("Query")
    ),
    "llm_fetch_list_lambda": _chain(
        lambda q, llm: LambdaOp([_fetch(llm)], _titles, mode="list")
    ),
    "llm_fetch_row_lambda": _chain(lambda q, llm: LambdaOp([_fetch(llm)], _first_line)),
    "llm_fetch_score": _chain(lambda q, llm: _score(q, _fetch(llm))),
    "llm_fetch_row_lambda_score": _chain(
        lambda q, llm: _score(q, LambdaOp([_fetch(llm)], _first_line))
    ),
    "llm_list_lambda_fetch_list_lambda": _chain(
        lambda q, llm: LambdaOp(
            [_fetch(LambdaOp([llm], _titles, mode="list"))], _titles, mode="list"
        )
    ),
    "fetch_llm": lambda: _llm(_fetch(input_placeholder("Query"))),
    "llm_score_list_lambda_list_lambda": _chain(
        lambda q, llm: LambdaOp(
            [LambdaOp([_score(q, llm)], _titles, mode="list")], _titles, mode="list"
        )
    ),
    "llm_row_lambda_fetch_list_lambda_list_lambda": _chain(
        lambda q, llm: LambdaOp(
            [LambdaOp([_fetch(LambdaOp([llm], _first_line))], _titles, mode="list")],
            _titles,
            mode="list",
        )
    ),
}


def _build(tail: Any) -> RuntimeGraph:
    compiled = Graph.from_ops([as_output("out", tail)]).compile(Query=["a b", "c d"])
    return RuntimeGraphBuilder().build(compiled)


def _spec_refs(spec: Any) -> Iterator[str]:
    """Every runtime node a spec names through a ``node`` key."""
    if isinstance(spec, dict):
        for key, value in spec.items():
            if key == "node" and isinstance(value, str):
                yield value
            else:
                yield from _spec_refs(value)
    elif isinstance(spec, list):
        for value in spec:
            yield from _spec_refs(value)


@pytest.mark.parametrize("name", sorted(_CHAINS))
def test_chain_builds_without_dangling_references(name: str) -> None:
    runtime_graph = _build(_CHAINS[name]())

    node_ids = set(runtime_graph.nodes)
    assert node_ids
    for node_id in runtime_graph.node_order:
        node = runtime_graph.nodes[node_id]
        for dependency in node.dependencies:
            assert (
                dependency in node_ids
            ), f"{name}: node {node_id} depends on missing node {dependency}"
        for reference in _spec_refs(node.data_spec):
            assert (
                reference in node_ids
            ), f"{name}: node {node_id} references missing node {reference}"
        assert node.data_spec.get("mode") != "flatten"


def test_derived_ops_build_after_the_llm_pass() -> None:
    query = input_placeholder("Query")
    llm = _llm(query)
    fetch = _fetch(llm)
    score = _score(query, fetch)
    tail = LambdaOp([LambdaOp([score], _titles, mode="list")], _titles, mode="list")

    runtime_graph = _build(tail)

    order = {node_id: idx for idx, node_id in enumerate(runtime_graph.node_order)}
    fetch_node = runtime_graph.dsl_to_runtime[fetch.id][0]
    pairs_node = f"{score.id}__pairs"
    score_node = runtime_graph.dsl_to_runtime[score.id][0]
    assert order[llm.id] < order[fetch_node] < order[pairs_node] < order[score_node]
    assert fetch_node in runtime_graph.nodes[pairs_node].dependencies
    assert pairs_node in runtime_graph.nodes[score_node].dependencies
