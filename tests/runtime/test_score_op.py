"""Tests for the Lumilake score op."""

import json
from pathlib import Path

import pytest

from lumilake import envs
from lumilake_server.common import GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    LambdaOp,
    LLMChatOp,
    OpMessage,
    ScoreConfig,
    ScoreOp,
    as_output,
    input_placeholder,
)
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime import python_step
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder


@pytest.fixture(autouse=True)
def _runtime_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "test" + "-pat")
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")


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

    assert runtime_op.task_type == "classification"
    assert runtime_op.data_spec == {
        "type": "list",
        "items": [
            {"text": "first question", "text_pair": "first candidate"},
            {"text": "second question", "text_pair": "second candidate"},
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

    assert flowmesh_spec["taskType"] == "classification"
    assert flowmesh_spec["model"]["vllm"]["convert"] == "classify"
    assert all(
        dep in runtime_graph.nodes
        for dep in runtime_op.to_flowmesh_node().get("dependsOn", [])
    )


def test_score_op_yaml_lowers_to_classification_task() -> None:
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
        node
        for node in runtime_graph.nodes.values()
        if node.task_type == "classification"
    ]

    assert runtime_op.data_spec["items"] == [
        {"text": "first question", "text_pair": "first candidate"},
        {"text": "second question", "text_pair": "second candidate"},
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
        node
        for node in runtime_graph.nodes.values()
        if node.task_type == "classification"
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


def test_score_op_feeds_list_lambda_with_ordered_items(tmp_path: Path) -> None:
    score = ScoreOp(
        query=["question"],
        candidates=["first", "second"],
        config=ScoreConfig(model="org/reranker"),
    )

    def collect(items):
        return [{"scores": items[0]}]

    scores = LambdaOp([score], fn=collect, mode="list")
    compiled = Graph.from_ops([as_output("scores", scores)]).compile()
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    score_node = runtime_graph.nodes[score.id]
    score_dir = tmp_path / score_node.node_id
    score_dir.mkdir()
    (score_dir / "results.json").write_text(
        json.dumps(
            {
                "result": {
                    "items": [
                        {"index": 0, "score": 0.9},
                        {"index": 1, "score": 0.2},
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
    runtime = python_step.runtime_blob()
    runtime_path = tmp_path / "lumilake-runtime.py"
    runtime_path.write_bytes(runtime.body)
    namespace: dict[str, object] = {"json": json}
    plan = consumer.data_spec["plan"]
    published = python_step.literal_blobs(plan) | {
        python_step.runtime_blob().key: python_step.runtime_blob()
    }
    code = python_step.flowmesh_spec(consumer.node_id, consumer.data_spec, published)[
        "code"
    ].replace(python_step.TASK_RUNTIME_MOUNT, str(runtime_path))
    exec(code, namespace)
    main = namespace["main"]
    assert callable(main)

    rows = main({score_node.node_id: str(score_dir)})["items"]

    assert rows == [
        {
            "output": {
                "scores": [
                    {"index": 0, "score": 0.9},
                    {"index": 1, "score": 0.2},
                ]
            }
        }
    ]


def test_score_op_feeds_llm_chat_op() -> None:
    score = ScoreOp(
        query=["question"],
        candidates=["first", "second"],
        config=ScoreConfig(model="org/reranker"),
    )
    answer = LLMChatOp(
        [OpMessage(role="user", content=score)],
        config=GenerationConfig(model="org/model"),
    )
    compiled = Graph.from_ops([as_output("answer", answer)]).compile()
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    score_node = runtime_graph.nodes[score.id]
    llm_node = runtime_graph.nodes[answer.id]

    columns = llm_node.data_spec["template"]["columns"]
    assert any(
        column.get("node") == score_node.node_id and column.get("path") == "items"
        for column in columns
    )
    assert score_node.node_id in llm_node.dependencies


def test_score_op_reads_list_lambda_candidates_inside_items(tmp_path: Path) -> None:
    """A whole-column Lambda emits its values as python-result items, so the
    pairs step reads them at the item-relative ``output`` path."""

    def prepare(items):
        return [str(item) for item in items]

    stock = input_placeholder("Texts")
    prepare_op = LambdaOp([stock], prepare, mode="list")
    score = ScoreOp(
        query="question",
        candidates=prepare_op,
        config=ScoreConfig(model="org/reranker", backend="vllm"),
    )
    compiled = Graph.from_ops([as_output("scores", score)]).compile(
        Texts=["first", "second"]
    )
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    pairs = next(
        node
        for node in runtime_graph.nodes.values()
        if node.task_type == python_step.TASK_TYPE
        and node.data_spec.get("mode") == "pairs"
    )
    plan = pairs.data_spec["plan"]
    candidate_entry = next(entry for entry in plan if entry["kind"] == "stage")

    assert candidate_entry["path"] == "output"

    code = python_step.pairs_source("pairs", plan)
    runtime_path = tmp_path / "lumilake-runtime.py"
    runtime_path.write_bytes(python_step.runtime_blob().body)
    code = code.replace(python_step.TASK_RUNTIME_MOUNT, str(runtime_path))
    namespace: dict[str, object] = {}
    exec(code, namespace)
    main = namespace["main"]
    assert callable(main)
    node_id = candidate_entry["node"]
    node_dir = tmp_path / node_id
    node_dir.mkdir()
    (node_dir / "results.json").write_text(
        json.dumps({"value": {"items": [{"output": "first"}, {"output": "second"}]}}),
        encoding="utf-8",
    )

    assert main({node_id: str(node_dir)}) == {
        "items": [
            {"text": "question", "text_pair": "first"},
            {"text": "question", "text_pair": "second"},
        ]
    }


def test_pairs_step_output_shape_broadcasts_single_query(tmp_path: Path) -> None:
    plan = [
        {"kind": "literal", "values": ["same question"]},
        {"kind": "literal", "values": ["first", "second"]},
    ]
    code = python_step.pairs_source("pairs", plan)
    runtime = python_step.runtime_blob()
    runtime_path = tmp_path / "lumilake-runtime.py"
    runtime_path.write_bytes(runtime.body)
    code = code.replace(python_step.TASK_RUNTIME_MOUNT, str(runtime_path))
    namespace: dict[str, object] = {}
    exec(code, namespace)
    main = namespace["main"]
    assert callable(main)
    assert main({}) == {
        "items": [
            {"text": "same question", "text_pair": "first"},
            {"text": "same question", "text_pair": "second"},
        ]
    }


def test_upstream_score_resolved_spec_is_dispatchable_shape(
    tmp_path: Path,
) -> None:
    upstream = _upstream_score_runtime_graph()
    score = next(
        node for node in upstream.nodes.values() if node.task_type == "classification"
    )
    pairs = next(
        node
        for node in upstream.nodes.values()
        if node.task_type == python_step.TASK_TYPE
        and node.data_spec.get("mode") == "pairs"
    )
    published = python_step.literal_blobs(pairs.data_spec["plan"]) | {
        python_step.runtime_blob().key: python_step.runtime_blob()
    }
    pairs_spec = python_step.flowmesh_spec(pairs.node_id, pairs.data_spec, published)
    runtime_path = tmp_path / "lumilake-runtime.py"
    runtime_path.write_bytes(python_step.runtime_blob().body)
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
    exec(
        pairs_spec["code"].replace(python_step.TASK_RUNTIME_MOUNT, str(runtime_path)),
        namespace,
    )
    main = namespace["main"]
    assert callable(main)
    pairs_output = main(inputs)["items"]
    score_spec = score.to_flowmesh_node()["spec"]
    resolved_items = pairs_output

    assert isinstance(resolved_items, list) and resolved_items
    assert all(
        isinstance(item, dict)
        and isinstance(item.get("text"), str)
        and item["text"].strip()
        and isinstance(item.get("text_pair"), str)
        and item["text_pair"].strip()
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
