"""Tests for the Lumilake score op."""

import json
from pathlib import Path

import pytest
import yaml

from lumilake import envs
from lumilake_server.common import GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    HttpFetchOp,
    LambdaOp,
    LLMChatOp,
    LLMVisionOp,
    OpMessage,
    ScoreConfig,
    ScoreOp,
    as_output,
    input_placeholder,
)
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime import python_step
from lumilake_server.runtime.runtime_graph import RuntimeGraph, RuntimeGraphBuilder


@pytest.fixture(autouse=True)
def _runtime_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "test" + "-pat")
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test" + "-token")


def _score_yaml(fields: dict) -> str:
    return yaml.safe_dump(
        {
            "name": "score_pairs",
            "inputs": {"Query": ["first question"], "Candidates": ["first candidate"]},
            "ops": [
                {
                    "id": "Rank",
                    "op": "ScoreOp",
                    "inputs": ["Query", "Candidates"],
                    "query": "Query",
                    "candidates": "Candidates",
                    "config": {"model": "org/reranker"},
                    **fields,
                }
            ],
            "outputs": [{"name": "scores", "ref": "Rank"}],
        }
    )


@pytest.mark.parametrize(
    ("backend", "model_block"),
    [("vllm", {"convert": "classify"}), ("transformers", {"device_map": "auto"})],
)
def test_score_yaml_lowers_to_a_classification_node(
    backend: str, model_block: dict[str, str]
) -> None:
    spec = parse_yaml_payload(
        _score_yaml(
            {
                "inputs": ["Query", "Query"],
                "candidates": "Query",
                "config": {"model": "org/reranker", "backend": backend},
            }
        )
    )["score_pairs"]
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
        (["q1", "q2"], ["c1"], "ScoreOp query and candidates must have equal"),
        (["q1"], [], "non-empty candidate"),
    ],
)
def test_score_op_rejects_bad_literal_inputs(
    query: list[str], candidates: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ScoreOp(query, candidates, config=ScoreConfig(model="org/reranker"))


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"config": {"model": "org/reranker", "backend": "tfjs"}}, "backend must be"),
        ({"config": {"model": "org/reranker", "device": "cpu"}}, "unknown fields"),
        ({"candidates": "Missing"}, "requires a 'candidates' reference"),
    ],
)
def test_score_yaml_errors_name_the_op(fields: dict, message: str) -> None:
    with pytest.raises(ValueError, match=f"ScoreOp 'Rank' .*{message}"):
        parse_yaml_payload(_score_yaml(fields))


def test_upstream_score_routes_through_a_pairs_step(tmp_path: Path) -> None:
    """A whole-column Lambda emits its values as python-result items, so the
    pairs step reads them at the item-relative ``output`` path and builds one
    pair per row."""

    def prepare(items):
        return [str(item) for item in items]

    prepare_op = LambdaOp([input_placeholder("Texts")], prepare, mode="list")
    score = ScoreOp(
        query="question",
        candidates=prepare_op,
        config=ScoreConfig(model="org/reranker"),
    )
    compiled = Graph.from_ops([as_output("scores", score)]).compile(
        Texts=["first", "second"]
    )
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    pairs = runtime_graph.nodes[f"{score.id}__pairs"]
    candidate_entry = pairs.data_spec["plan"][1]
    score_node = runtime_graph.nodes[score.id]
    stage_dir = tmp_path / "prep"
    stage_dir.mkdir()
    (stage_dir / "results.json").write_text(
        json.dumps({"value": {"items": [{"output": "c1"}, {"output": "c2"}]}}),
        encoding="utf-8",
    )
    namespace: dict[str, object] = {}
    exec(pairs.to_flowmesh_node()["spec"]["code"], namespace)
    main = namespace["main"]
    assert callable(main)

    assert score_node.dependencies == (pairs.node_id,)
    assert score_node.data_spec == {
        "type": "list",
        "items": f"${{{pairs.node_id}.value.items}}",
    }
    assert candidate_entry == {
        "kind": "stage",
        "node": pairs.dependencies[0],
        "path": "output",
    }
    assert main({candidate_entry["node"]: str(stage_dir)}) == {
        "items": [
            {"text": "question", "text_pair": "c1"},
            {"text": "question", "text_pair": "c2"},
        ]
    }


def test_score_op_reads_vision_llm_candidates_from_the_model_node() -> None:
    """A vision LLM compiles to an embedding stage and the model, so the pairs
    step must read the model's output, not the embedding's."""
    stock = input_placeholder("Stock")
    vision = LLMVisionOp(
        [OpMessage(role="user", content="Describe the image.")],
        image_source=stock.id,
        image_source_op=stock,
        config=GenerationConfig(model="org/vision-model"),
    )
    score = ScoreOp(
        query="question",
        candidates=vision,
        config=ScoreConfig(model="org/reranker"),
    )
    compiled = Graph.from_ops([as_output("scores", score)]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    pairs = runtime_graph.nodes[f"{score.id}__pairs"]
    assert pairs.data_spec["plan"][1] == {
        "kind": "stage",
        "node": vision.id,
        "path": "output",
    }


def test_mixed_derived_chain_builds_in_dependency_order() -> None:
    """A chain that mixes an LLM, a fetch, a row Lambda, a score and list
    Lambdas lowers in dependency order, with no reference to a missing node."""

    def first_line(value):
        return str(value).splitlines()[0]

    def titles(docs):
        return [str(doc) for doc in docs]

    query = input_placeholder("Query")
    llm = LLMChatOp(
        [OpMessage(role="user", content=query)], config=GenerationConfig(model="org/m")
    )
    fetch = HttpFetchOp("https://search.example/api?q={query}", llm)
    score = ScoreOp(
        query, LambdaOp([fetch], first_line), config=ScoreConfig(model="org/reranker")
    )
    head = LambdaOp([LambdaOp([score], titles, mode="list")], titles, mode="list")
    compiled = Graph.from_ops([as_output("out", head)]).compile(Query=["a", "b"])
    runtime_graph: RuntimeGraph = RuntimeGraphBuilder().build(compiled)

    order = {node_id: i for i, node_id in enumerate(runtime_graph.node_order)}
    for node in runtime_graph.nodes.values():
        for dependency in node.dependencies:
            assert dependency in runtime_graph.nodes
    pairs_id = f"{score.id}__pairs"
    chain = [llm.id, fetch.id, pairs_id]
    chain += [runtime_graph.dsl_to_runtime[score.id][0], head.id]
    assert [order[node_id] for node_id in chain] == sorted(order[n] for n in chain)


@pytest.mark.parametrize(
    ("query", "candidates", "message"),
    [
        (["  "], ["first"], r"^pairs: row 0 query must be"),
        (["same question"], [], r"^pairs: candidate column is empty"),
    ],
)
def test_pairs_step_rejects_bad_columns(
    query: list[str], candidates: list[str], message: str
) -> None:
    """A blank fail-open fetch row or an upstream stage that returns nothing
    reaches the pairs step unchanged; it rejects both instead of letting
    FlowMesh fail the whole classification task."""
    plan: python_step.ColumnPlan = [
        {"kind": "literal", "values": query},
        {"kind": "literal", "values": candidates},
    ]
    spec = python_step.flowmesh_spec(
        "pairs",
        {
            "mode": "pairs",
            "plan": plan,
            "timeout_s": python_step.DEFAULT_TIMEOUT_SECONDS,
        },
    )
    namespace: dict[str, object] = {}
    exec(spec["code"], namespace)
    main = namespace["main"]
    assert callable(main)

    with pytest.raises(ValueError, match=message):
        main({})
