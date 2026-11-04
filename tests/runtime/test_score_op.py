"""Tests for the Lumilake score op."""

import pytest

from lumilake_server.graphs import Graph
from lumilake_server.ops import ScoreConfig, ScoreOp, as_output
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder


def test_score_op_requires_config_with_named_error() -> None:
    with pytest.raises(ValueError, match="ScoreOp requires config"):
        ScoreOp("question", ["candidate"])


def test_score_op_lowers_literal_pairs_in_order() -> None:
    op = ScoreOp(
        query=["first question", "second question"],
        candidates=["first candidate", "second candidate"],
        config=ScoreConfig(model="org/reranker", backend="vllm"),
    )
    compiled = Graph.from_ops([as_output("scores", op)]).compile()
    runtime_op = RuntimeGraphBuilder().build(compiled).nodes[op.id]

    assert runtime_op.task_type == "score"
    assert runtime_op.data_spec == {
        "type": "list",
        "items": [
            {"query": "first question", "candidate": "first candidate"},
            {"query": "second question", "candidate": "second candidate"},
        ],
    }
    assert runtime_op.model_spec["vllm"]["convert"] == "classify"


def test_score_op_yaml_lowers_to_score_task() -> None:
    payload = """
    name: score_pairs
    inputs:
      Query: ["first question", "second question"]
      Candidates: ["first candidate", "second candidate"]
    ops:
      - id: Score
        op: ScoreOp
        inputs: [Query, Candidates]
        query: Query
        candidates: Candidates
        config:
          model: org/reranker
          backend: vllm
    outputs:
      - name: scores
        ref: Score
    """
    spec = parse_yaml_payload(payload)["score_pairs"]
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    [runtime_op] = [
        node for node in runtime_graph.nodes.values() if node.task_type == "score"
    ]

    assert runtime_op.data_spec["items"] == [
        {"query": "first question", "candidate": "first candidate"},
        {"query": "second question", "candidate": "second candidate"},
    ]


def test_score_op_literal_length_mismatch_names_op() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        ScoreOp(
            query=["q1", "q2"],
            candidates=["c1"],
            config=ScoreConfig(model="org/reranker"),
        )


def test_score_op_empty_candidates_name_op() -> None:
    with pytest.raises(ValueError, match="non-empty candidate"):
        ScoreOp(
            query=["q1"],
            candidates=[],
            config=ScoreConfig(model="org/reranker"),
        )
