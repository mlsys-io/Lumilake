"""The news branch of the trading workflow compiles with several symbols in
one graph; the per-row VLM over a retrieval runs as flatten -> embedding +
inference -> regroup.
"""

from typing import Any

import pytest
import yaml

from lumilake import envs
from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraph, RuntimeGraphBuilder

_WORKFLOW = """\
name: trading-news
inputs:
  Symbols:
  - NVDA
ops:
- id: News Query
  op: DataRetrievalOp
  inputs:
  - Symbols
  data_spec:
    type: lumid
    mode: sql
    output_format: jsonl
    verify: false
    template: >-
      SELECT news_id, title, publisheddate, category, synopsis FROM news WHERE
      symbol = '{symbol}' LIMIT 10;
    params:
    - label: symbol
      node: Symbols
- id: News Artifact Query
  op: DataRetrievalOp
  inputs:
  - News Query
  data_spec:
    type: lumid
    mode: s3
    output_format: jsonl
    verify: false
    template: lumilake_experiment_ai4finance_10g/unstructured/news-images/{news_id}.png
    params:
    - label: news_id
      node: News Query
      path: items.table.news_id
- id: News Per-row Summary
  op: LLMVisionOp
  inputs:
  - Symbols
  - News Query
  - News Artifact Query
  messages:
  - role: system
    content: You are a concise data summarization assistant.
  rowwise_template: >-
    Summarize the following retrieved news item in two concise sentences focused
    on investment impact for {symbol}. Title: {title} Published: {published}
    Category: {category} Synopsis: {synopsis} Artifact: {artifact}
  rowwise_columns:
  - label: symbol
    node: Symbols
  - label: title
    node: News Query
    path: items.table.title
  - label: published
    node: News Query
    path: items.table.publisheddate
  - label: category
    node: News Query
    path: items.table.category
  - label: synopsis
    node: News Query
    path: items.table.synopsis
  - label: artifact
    node: News Artifact Query
    path: items.keys
  image_source: News Artifact Query
  image_path: images
  config:
    model: llava-hf/llava-1.5-7b-hf
    max_tokens: 512
    temperature: 0.1
    top_p: 1
- id: News Report
  op: LLMChatOp
  inputs:
  - Symbols
  - News Query
  - News Per-row Summary
  messages:
  - role: system
    content: You are a careful financial analyst.
  - role: user
    content: ''
  config:
    model: Qwen/Qwen3-8B
    max_tokens: 1024
    temperature: 0.7
    top_p: 1
    max_model_len: 16384
    chat_template_kwargs:
      enable_thinking: false
  prompt:
    template: >-
      Given the following per-article summaries, generate an aggregated news
      summary for {ref0}.

      ```table
      {df}
      ```

      Return a structured report with sections: Summary, Key Developments,
      Impact, Open Questions.
    format_kwargs:
      ref0: Symbols
  aggregate_table:
  - label: title
    node: News Query
    path: items.table.title
  - label: publishedDate
    node: News Query
    path: items.table.publisheddate
  - label: category
    node: News Query
    path: items.table.category
  - label: summary
    node: News Per-row Summary
    path: items.output
outputs:
- name: report
  ref: News Report
"""


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "test-runtime")


def _build(symbols: list[str]) -> RuntimeGraph:
    payload = yaml.safe_load(_WORKFLOW)
    payload["inputs"] = {"Symbols": symbols}
    specs = parse_yaml_payload(payload)
    spec = next(iter(specs.values()))
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    return RuntimeGraphBuilder().build(compiled)


def _node(runtime_graph: RuntimeGraph, name: str) -> Any:
    matches = [
        node
        for node_id, node in runtime_graph.nodes.items()
        if f"_{name.replace(' ', '_').replace('-', '_')}_" in f"{node_id}_"
        and not node_id.endswith("_embedding")
        and not node_id.endswith("_flatten")
        and not node_id.endswith("_infer")
    ]
    assert len(matches) == 1, (name, [n for n in runtime_graph.nodes])
    return matches[0]


def _columns(node: Any) -> dict[str, dict[str, Any]]:
    return {col["label"]: col for col in node.data_spec["template"]["columns"]}


def _node_columns(node: Any) -> dict[str, tuple[str, str]]:
    return {
        label: (col["node"], col["path"])
        for label, col in _columns(node).items()
        if "node" in col
    }


@pytest.mark.parametrize("symbols", [["NVDA", "MSFT", "AAPL"], ["NVDA"]])
def test_per_row_vlm_exports_flatten_infer_regroup(symbols: list[str]) -> None:
    runtime_graph = _build(symbols)

    (vlm_id,) = [
        node_id
        for node_id, node in runtime_graph.nodes.items()
        if node.task_type == "python" and node.data_spec["mode"] == "regroup"
    ]
    vlm_ids = {
        node_id for node_id in runtime_graph.nodes if "News_Per_row_Summary" in node_id
    }
    assert vlm_ids == {
        f"{vlm_id}_flatten",
        f"{vlm_id}_embedding",
        f"{vlm_id}_infer",
        vlm_id,
    }

    flatten = runtime_graph.nodes[f"{vlm_id}_flatten"]
    assert flatten.task_type == "python"
    assert flatten.data_spec["mode"] == "flatten"
    assert runtime_graph.nodes[f"{vlm_id}_embedding"].task_type == "embedding"
    assert runtime_graph.nodes[f"{vlm_id}_infer"].task_type == "inference"

    plan = flatten.data_spec["plan"]
    assert {
        "kind": "literal",
        "values": symbols,
        "label": "symbol",
        "grouped": False,
    } in plan

    report = _node(runtime_graph, "News Report")
    df = next(
        col for col in report.data_spec["template"]["columns"] if col["label"] == "df"
    )
    summary = next(col for col in df["data"]["columns"] if col["label"] == "summary")
    assert summary["node"] == vlm_id
    assert summary["path"] == "value.items.output"


def test_per_row_vlm_binds_images_and_prompts_per_input_row() -> None:
    """News Query returns a variable number of rows per symbol, so every link
    from the retrieval to the per-row VLM and the news report keeps one group
    per input row: the artifact retrieval takes its keys from News Query, the
    embedding step reads the artifact contents, the VLM reads the embedding as
    its image source next to one Symbols value per row, and the report reads
    the news columns and the VLM outputs."""
    runtime_graph = _build(["A", "B"])
    news = _node(runtime_graph, "News Query")
    artifacts = _node(runtime_graph, "News Artifact Query")
    vlm = _node(runtime_graph, "News Per-row Summary")
    report = _node(runtime_graph, "News Report")

    (vlm_id,) = [
        node_id for node_id, node in runtime_graph.nodes.items() if node is vlm
    ]
    embedding_id = f"{vlm_id}_embedding"
    infer_id = f"{vlm_id}_infer"

    news_id = next(
        param for param in artifacts.data_spec["params"] if param["label"] == "news_id"
    )
    assert news_id["path"] == "items.table.news_id"
    assert news_id["node"] in {
        node_id for node_id, node in runtime_graph.nodes.items() if node is news
    }

    embedding = runtime_graph.nodes[embedding_id]
    assert embedding.data_spec["path"] == "items.content"
    assert embedding.data_spec["node"] in {
        node_id for node_id, node in runtime_graph.nodes.items() if node is artifacts
    }

    assert runtime_graph.nodes[infer_id].data_spec["image_embedding"] == {
        "node": embedding_id,
        "path": "embedding_file",
    }

    flatten = runtime_graph.nodes[f"{vlm_id}_flatten"]
    plan = flatten.data_spec["plan"]
    assert {
        "kind": "literal",
        "values": ["A", "B"],
        "label": "symbol",
        "grouped": False,
    } in plan

    df = next(
        col for col in report.data_spec["template"]["columns"] if col["label"] == "df"
    )
    summary = next(col for col in df["data"]["columns"] if col["label"] == "summary")
    assert summary["node"] == vlm_id
    assert summary["path"] == "value.items.output"


@pytest.mark.parametrize("symbols", [["NVDA", "MSFT", "AAPL"], ["NVDA"]])
def test_per_row_vlm_nodes_export_to_flowmesh(symbols: list[str]) -> None:
    runtime_graph = _build(symbols)
    for node_id, node in runtime_graph.nodes.items():
        flowmesh_node = node.to_flowmesh_node()
        if node.task_type == "python":
            compile(flowmesh_node["spec"]["code"], node_id, "exec")
