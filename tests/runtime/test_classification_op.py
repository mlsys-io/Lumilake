"""Tests for the Lumilake classification op."""

import pytest
import yaml

from lumilake import envs
from lumilake_server.common import GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    ClassificationConfig,
    ClassificationOp,
    HttpFetchOp,
    LambdaOp,
    LLMChatOp,
    OpMessage,
    as_output,
    input_placeholder,
)
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraph, RuntimeGraphBuilder


@pytest.fixture(autouse=True)
def _runtime_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "test" + "-pat")
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test" + "-token")


def _classification_yaml(fields: dict) -> str:
    return yaml.safe_dump(
        {
            "name": "classify_pairs",
            "inputs": {"Query": ["first question"], "Candidates": ["first candidate"]},
            "ops": [
                {
                    "id": "Rank",
                    "op": "ClassificationOp",
                    "inputs": ["Query", "Candidates"],
                    "query": "Query",
                    "candidates": "Candidates",
                    "config": {"model": "org/reranker"},
                    **fields,
                }
            ],
            "outputs": [{"name": "labels", "ref": "Rank"}],
        }
    )


@pytest.mark.parametrize(
    ("backend", "model_block"),
    [("vllm", {"convert": "classify"}), ("transformers", {"device_map": "auto"})],
)
def test_classification_yaml_lowers_to_a_classification_node(
    backend: str, model_block: dict[str, str]
) -> None:
    spec = parse_yaml_payload(
        _classification_yaml(
            {
                "inputs": ["Query", "Query"],
                "candidates": "Query",
                "config": {"model": "org/reranker", "backend": backend},
            }
        )
    )["classify_pairs"]
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    [node] = [
        node
        for node in runtime_graph.nodes.values()
        if node.task_type == "classification"
    ]
    model_spec = node.to_flowmesh_node()["spec"]["model"]

    assert node.data_spec == {
        "type": "list",
        "items": [{"text": "first question", "text_pair": "first question"}],
    }
    assert model_spec["source"]["identifier"] == "org/reranker"
    assert model_spec[backend] == model_block


@pytest.mark.parametrize(
    ("query", "candidates", "message"),
    [
        (["q1", "q2"], ["c1"], "ClassificationOp query and candidates must have equal"),
        (["q1"], [], "non-empty candidate"),
    ],
)
def test_classification_op_rejects_bad_literal_inputs(
    query: list[str], candidates: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ClassificationOp(
            query, candidates, config=ClassificationConfig(model="org/reranker")
        )


def test_classification_yaml_errors_name_the_op() -> None:
    fields = {"config": {"model": "org/reranker", "device": "cpu"}}
    with pytest.raises(ValueError, match="ClassificationOp 'Rank' .*unknown fields"):
        parse_yaml_payload(_classification_yaml(fields))


def test_mixed_derived_chain_builds_in_dependency_order() -> None:
    """A chain that mixes an LLM, a fetch, a row Lambda, a classification and
    list Lambdas lowers in dependency order, with no reference to a missing
    node."""

    def first_line(value):
        return str(value).splitlines()[0]

    def titles(docs):
        return [str(doc) for doc in docs]

    query = input_placeholder("Query")
    llm = LLMChatOp(
        [OpMessage(role="user", content=query)], config=GenerationConfig(model="org/m")
    )
    fetch = HttpFetchOp("https://search.example/api?q={query}", llm)
    classification = ClassificationOp(
        query,
        LambdaOp([fetch], first_line),
        config=ClassificationConfig(model="org/reranker"),
    )
    head = LambdaOp(
        [LambdaOp([classification], titles, mode="list")], titles, mode="list"
    )
    compiled = Graph.from_ops([as_output("out", head)]).compile(Query=["a", "b"])
    runtime_graph: RuntimeGraph = RuntimeGraphBuilder().build(compiled)

    order = {node_id: i for i, node_id in enumerate(runtime_graph.node_order)}
    pairs_id = f"{classification.id}__pairs"
    pairs = runtime_graph.nodes[pairs_id]
    # The pairs step reads its upstream at the item-relative stage path.
    candidate_stage = pairs.data_spec["plan"][1]
    assert candidate_stage["kind"] == "stage" and candidate_stage["path"] == "output"
    assert runtime_graph.nodes[classification.id].dependencies == (pairs_id,)
    for node in runtime_graph.nodes.values():
        for dependency in node.dependencies:
            assert dependency in runtime_graph.nodes
    chain = [llm.id, fetch.id, pairs_id]
    chain += [runtime_graph.dsl_to_runtime[classification.id][0], head.id]
    assert [order[node_id] for node_id in chain] == sorted(order[n] for n in chain)
