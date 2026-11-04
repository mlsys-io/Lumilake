"""Tests for the Lumilake score op."""

import json
from pathlib import Path

import pytest

from lumilake_server.graphs import Graph
from lumilake_server.ops import ScoreConfig, ScoreOp, as_output
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime import python_step
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
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    runtime_op = runtime_graph.nodes[op.id]

    assert runtime_op.task_type == "score"
    assert runtime_op.data_spec == {
        "type": "list",
        "items": [
            {"query": "first question", "candidate": "first candidate"},
            {"query": "second question", "candidate": "second candidate"},
        ],
    }
    assert runtime_op.model_spec["vllm"]["convert"] == "classify"
    assert runtime_op.dependencies == ()
    assert all(
        dep in runtime_graph.nodes
        for dep in runtime_op.to_flowmesh_node().get("dependsOn", [])
    )


def test_score_runtime_node_emits_flowmesh_model_block() -> None:
    op = ScoreOp(
        query="question",
        candidates=["first", "second"],
        config=ScoreConfig(model="org/reranker", backend="vllm"),
    )
    compiled = Graph.from_ops([as_output("scores", op)]).compile()
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    runtime_op = runtime_graph.nodes[op.id]
    flowmesh_spec = runtime_op.to_flowmesh_node()["spec"]

    assert flowmesh_spec["taskType"] == "score"
    assert flowmesh_spec["model"]["vllm"]["convert"] == "classify"
    assert all(
        dep in runtime_graph.nodes
        for dep in runtime_op.to_flowmesh_node().get("dependsOn", [])
    )


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


def test_upstream_score_uses_pairs_node_placeholder() -> None:
    payload = """
    name: score_upstream
    inputs:
      Unused: []
    ops:
      - id: Query
        op: LLMChatOp
        messages:
          - role: user
            content: "same question"
        config:
          model: org/model
          api: {}
      - id: Candidates
        op: LLMChatOp
        messages:
          - role: user
            content: "candidate source"
        config:
          model: org/model
          api: {}
      - id: Score
        op: ScoreOp
        inputs: [Query, Candidates]
        query: Query
        candidates: Candidates
        config:
          model: org/reranker
    outputs:
      - name: scores
        ref: Score
    """
    spec = parse_yaml_payload(payload)["score_upstream"]
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    score = next(
        node for node in runtime_graph.nodes.values() if node.task_type == "score"
    )
    pairs = next(
        node
        for node in runtime_graph.nodes.values()
        if node.task_type == python_step.TASK_TYPE
        and node.data_spec.get("mode") == "pairs"
    )

    assert score.data_spec == {
        "type": "list",
        "items": f"${{{pairs.node_id}.value.items}}",
    }
    assert score.dependencies == (pairs.node_id,)
    assert score.to_flowmesh_node()["dependsOn"] == [pairs.node_id]
    assert pairs.to_flowmesh_node()["dependsOn"] == list(pairs.dependencies)
    assert all(dep in runtime_graph.nodes for dep in pairs.dependencies)
    assert all(
        entry.get("node") in runtime_graph.nodes
        for entry in pairs.data_spec["plan"]
        if entry.get("kind") == "stage"
    )


def test_pairs_step_output_shape_broadcasts_single_query() -> None:
    plan = [
        {"kind": "literal", "values": ["same question"]},
        {"kind": "literal", "values": ["first", "second"]},
    ]
    code = python_step.pairs_source("pairs", plan)
    namespace: dict[str, object] = {}
    exec(code, namespace)
    main = namespace["main"]
    assert callable(main)
    assert main({}) == {
        "items": [
            {"query": "same question", "candidate": "first"},
            {"query": "same question", "candidate": "second"},
        ]
    }


def test_upstream_score_resolved_spec_is_dispatchable_shape(
    tmp_path: Path,
) -> None:
    upstream = _upstream_score_runtime_graph()
    score = next(node for node in upstream.nodes.values() if node.task_type == "score")
    pairs = next(
        node
        for node in upstream.nodes.values()
        if node.task_type == python_step.TASK_TYPE
        and node.data_spec.get("mode") == "pairs"
    )
    pairs_spec = pairs.to_flowmesh_node()["spec"]
    inputs: dict[str, str] = {}
    for plan, content in zip(
        pairs.data_spec["plan"],
        ["same question", "candidate"],
        strict=True,
    ):
        node_id = plan["node"]
        node_dir = tmp_path / node_id
        node_dir.mkdir()
        (node_dir / "results.json").write_text(
            json.dumps(
                {
                    "result": {
                        "items": [
                            {"json": {"choices": [{"message": {"content": content}}]}}
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        inputs[node_id] = str(node_dir)
    namespace: dict[str, object] = {"json": json}
    exec(pairs_spec["code"], namespace)
    main = namespace["main"]
    assert callable(main)
    pairs_output = main(inputs)["items"]
    score_spec = score.to_flowmesh_node()["spec"]
    resolved_items = pairs_output

    assert isinstance(resolved_items, list) and resolved_items
    assert all(
        isinstance(item, dict)
        and isinstance(item.get("query"), str)
        and item["query"].strip()
        and isinstance(item.get("candidate"), str)
        and item["candidate"].strip()
        for item in resolved_items
    )
    model = score_spec["model"]
    assert model["source"]["identifier"] == "org/reranker"
    assert model["vllm"]["convert"] == "classify"
    assert "diffusers" not in model and "adapters" not in model


def _upstream_score_runtime_graph():
    payload = """
    name: score_upstream_contract
    inputs:
      Unused: []
    ops:
      - id: Query
        op: LLMChatOp
        messages:
          - role: user
            content: "same question"
        config:
          model: org/model
          api: {}
      - id: Candidates
        op: LLMChatOp
        messages:
          - role: user
            content: "candidate source"
        config:
          model: org/model
          api: {}
      - id: Score
        op: ScoreOp
        inputs: [Query, Candidates]
        query: Query
        candidates: Candidates
        config:
          model: org/reranker
    outputs:
      - name: scores
        ref: Score
    """
    spec = parse_yaml_payload(payload)["score_upstream_contract"]
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    return RuntimeGraphBuilder().build(compiled)
