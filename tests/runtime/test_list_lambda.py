import textwrap
from typing import Any

import pytest

from lumilake import envs
from lumilake_server.common import ApiConfig, GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    DataRetrievalOp,
    EmbeddingOp,
    ImageGenerationOp,
    LambdaOp,
    LLMChatOp,
    LLMCompletionOp,
    LLMVisionOp,
    OpMessage,
    as_output,
    data,
    input_placeholder,
)
from lumilake_server.parser.yaml_parser import parse_yaml_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraph, RuntimeGraphBuilder

_LUMID_URL = "http://lumid-data"
_LUMID_TOKEN = "test-token"
_RUNTIME_TOKEN = "test" + "-pat"


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", _LUMID_URL)
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", _LUMID_TOKEN)
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", _RUNTIME_TOKEN)


def _build(yaml_text: str) -> RuntimeGraph:
    specs = parse_yaml_payload(yaml_text)
    spec = next(iter(specs.values()))
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    return RuntimeGraphBuilder().build(compiled)


def _explode_fn(items: tuple[list[Any], ...]) -> list[dict[str, str]]:
    return [{"value": v} for v in items[0]]


def _row_fn(items: tuple[Any, ...]) -> str:
    return str(items[0])


def test_list_lambda_explodes_input_feeds_rowwise_llm() -> None:
    yaml_text = textwrap.dedent("""
        name: list_lambda_rowwise
        inputs:
          stock: ["NVDA", "AAPL"]
        ops:
          - id: explode
            op: LambdaOp
            inputs: [stock]
            fn_name: explode
            code: |
              def explode(items):
                  return [{"value": v} for v in items[0]]
            mode: list
          - id: summarize
            op: LLMChatOp
            inputs: [explode]
            messages:
              - role: user
                content: "hello"
            rowwise_template: "Summarize {value}"
            rowwise_columns:
              - { label: value, node: explode, path: "items.output.value" }
            config: { model: meta-llama/Llama-3.1-8B-Instruct }
        outputs:
          - { name: out, ref: summarize }
        """)
    runtime_graph = _build(yaml_text)

    explode_node = next(
        n for n in runtime_graph.nodes.values() if n.task_type == "python"
    )
    assert explode_node.data_spec["mode"] == "list"
    assert explode_node.data_spec["plan"] == [
        {"kind": "literal", "values": ["NVDA", "AAPL"]}
    ]

    summarize_node = next(
        n
        for n in runtime_graph.nodes.values()
        if n.task_type == "inference" and "columns" in n.data_spec
    )
    assert explode_node.node_id in summarize_node.dependencies
    columns = summarize_node.data_spec["columns"]
    assert {
        "label": "value",
        "node": explode_node.node_id,
        "path": "value.items.output.value",
    } in columns


def test_list_lambda_groups_then_collapses() -> None:
    yaml_text = textwrap.dedent("""
        name: list_lambda_groups_collapse
        inputs:
          stock: ["NVDA", "AAPL"]
        ops:
          - id: group
            op: LambdaOp
            inputs: [stock]
            fn_name: group
            code: |
              def group(items):
                  return [[{"value": v}] for v in items[0]]
            mode: list
          - id: summarize
            op: LLMChatOp
            inputs: [group]
            messages:
              - role: user
                content: "hello"
            rowwise_template: "Summarize {value}"
            rowwise_columns:
              - { label: value, node: group, path: "items.output.value" }
            config: { model: meta-llama/Llama-3.1-8B-Instruct }
          - id: collapse
            op: LambdaOp
            inputs: [summarize]
            fn_name: collapse
            code: |
              def collapse(items):
                  return [{"summary": items[0]}]
            mode: list
        outputs:
          - { name: out, ref: collapse }
        """)
    runtime_graph = _build(yaml_text)

    python_nodes = [n for n in runtime_graph.nodes.values() if n.task_type == "python"]
    assert len(python_nodes) == 2
    group_node = next(
        n
        for n in python_nodes
        if n.data_spec["plan"] == [{"kind": "literal", "values": ["NVDA", "AAPL"]}]
    )
    collapse_node = next(n for n in python_nodes if n is not group_node)
    summarize_node = next(
        n
        for n in runtime_graph.nodes.values()
        if n.task_type == "inference" and "columns" in n.data_spec
    )
    (collapse_plan,) = collapse_node.data_spec["plan"]
    assert collapse_plan == {
        "kind": "stage",
        "node": summarize_node.node_id,
        "flatten": True,
    }
    assert summarize_node.node_id in collapse_node.dependencies
    assert collapse_node.node_id in runtime_graph.output_node_map


def test_list_lambda_as_workflow_output() -> None:
    yaml_text = textwrap.dedent("""
        name: list_lambda_output
        inputs:
          stock: ["NVDA", "AAPL"]
        ops:
          - id: explode
            op: LambdaOp
            inputs: [stock]
            fn_name: explode
            code: |
              def explode(items):
                  return [{"value": v} for v in items[0]]
            mode: list
        outputs:
          - { name: out, ref: explode }
        """)
    runtime_graph = _build(yaml_text)

    (explode_node,) = [
        n for n in runtime_graph.nodes.values() if n.task_type == "python"
    ]
    assert explode_node.data_spec["mode"] == "list"
    assert explode_node.node_id in runtime_graph.output_node_map
    assert runtime_graph.output_paths[explode_node.node_id] == "items.output"


def test_list_lambda_in_message_chain_raises() -> None:
    yaml_text = textwrap.dedent("""
        name: list_lambda_in_chain
        inputs:
          stock: ["NVDA"]
        ops:
          - id: explode
            op: LambdaOp
            inputs: [stock]
            fn_name: explode
            code: |
              def explode(items):
                  return [{"value": v} for v in items[0]]
            mode: list
          - id: summarize
            op: LLMChatOp
            inputs: [explode]
            messages:
              - role: user
                content: explode
            config: { model: meta-llama/Llama-3.1-8B-Instruct }
        outputs:
          - { name: out, ref: summarize }
        """)
    with pytest.raises(ValueError, match="node column"):
        _build(yaml_text)


def test_per_row_lambda_behaviour_unchanged() -> None:
    yaml_text = textwrap.dedent("""
        name: per_row_lambda
        inputs:
          stock: ["NVDA"]
        ops:
          - id: tag
            op: LambdaOp
            inputs: [stock]
            fn_name: tag
            code: |
              def tag(items):
                  return "tagged:" + items[0]
          - id: summarize
            op: LLMChatOp
            inputs: [tag]
            messages:
              - role: user
                content: tag
            config: { model: meta-llama/Llama-3.1-8B-Instruct }
        outputs:
          - { name: out, ref: summarize }
        """)
    runtime_graph = _build(yaml_text)

    assert not any(n.task_type == "echo" for n in runtime_graph.nodes.values())
    assert not any(n.task_type == "python" for n in runtime_graph.nodes.values())
    summarize_node = next(
        n for n in runtime_graph.nodes.values() if n.task_type == "inference"
    )
    template = summarize_node.data_spec["template"]
    steps = template["options"]["format"]["steps"]
    assert any(step.get("function") is not None for step in steps)


def test_list_lambda_over_llm_output_feeds_rowwise_llm() -> None:
    yaml_text = textwrap.dedent("""
        name: list_lambda_over_llm
        inputs:
          stock: ["NVDA"]
        ops:
          - id: extract
            op: LLMChatOp
            inputs: [stock]
            messages:
              - role: user
                content: "extract text"
            config: { model: meta-llama/Llama-3.1-8B-Instruct }
          - id: explode
            op: LambdaOp
            inputs: [extract]
            fn_name: explode
            code: |
              def explode(items):
                  return [{"value": v} for v in items[0]]
            mode: list
          - id: summarize
            op: LLMChatOp
            inputs: [explode]
            messages:
              - role: user
                content: "hello"
            rowwise_template: "Summarize {value}"
            rowwise_columns:
              - { label: value, node: explode, path: "items.output.value" }
            config: { model: meta-llama/Llama-3.1-8B-Instruct }
        outputs:
          - { name: out, ref: summarize }
        """)
    runtime_graph = _build(yaml_text)

    explode_node = next(
        n for n in runtime_graph.nodes.values() if n.task_type == "python"
    )
    assert explode_node.data_spec["mode"] == "list"
    extract_node = next(
        n
        for n in runtime_graph.nodes.values()
        if n.task_type == "inference" and "columns" not in n.data_spec
    )
    (extract_plan,) = explode_node.data_spec["plan"]
    assert extract_plan == {"kind": "stage", "node": extract_node.node_id}
    assert extract_node.node_id in explode_node.dependencies

    summarize_node = next(
        n
        for n in runtime_graph.nodes.values()
        if n.task_type == "inference" and "columns" in n.data_spec
    )
    assert explode_node.node_id in summarize_node.dependencies
    columns = summarize_node.data_spec["columns"]
    assert {
        "label": "value",
        "node": explode_node.node_id,
        "path": "value.items.output.value",
    } in columns

    order = {node_id: idx for idx, node_id in enumerate(runtime_graph.node_order)}
    assert order[extract_node.node_id] < order[explode_node.node_id]
    assert order[explode_node.node_id] < order[summarize_node.node_id]


def test_list_lambda_over_rowwise_llm_flattens_grouped_item() -> None:
    """A list-mode Lambda reading a rowwise LLM must mark the plan entry
    ``flatten`` so the python step expands the grouped item back to one value
    per row (instead of handing the list-mode Lambda a single grouped element)."""
    stock = input_placeholder("Stock")
    shout = LambdaOp([stock], fn=lambda inputs: str(inputs[0]).upper())
    reply = LLMChatOp(
        [OpMessage(role="user", content="Reply {r}")],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
        rowwise_template="Reply {r}",
        rowwise_columns=[{"label": "r", "node": shout.id, "path": "items.output"}],
    )
    reply.inputs.append(shout)
    tally = LambdaOp(
        [reply],
        fn=lambda items: [{"n": len(items[0])}],
        mode="list",
    )
    output = as_output("out", tally)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA", "AAPL", "MSFT"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    tally_node = next(
        n for n in runtime_graph.nodes.values() if n.task_type == "python"
    )
    assert tally_node.data_spec["mode"] == "list"
    reply_node = next(
        n
        for n in runtime_graph.nodes.values()
        if n.task_type == "inference" and "columns" in n.data_spec
    )
    (reply_plan,) = tally_node.data_spec["plan"]
    assert reply_plan == {
        "kind": "stage",
        "node": reply_node.node_id,
        "flatten": True,
    }
    assert reply_node.node_id in tally_node.dependencies


def test_list_lambda_over_rowwise_vlm_does_not_flatten() -> None:
    """A list-mode Lambda reading a rowwise VLM must not mark the plan entry
    ``flatten``: the VLM's regroup step already emits one output list per source
    symbol, and flattening would expand image-level values into per-image rows."""
    stock = input_placeholder("Stock")
    news_s3 = DataRetrievalOp(
        data_spec={
            "type": "lumid",
            "mode": "s3",
            "template": "example-data/news/images/{symbol}.png",
            "params": [{"name": "symbol", "node": stock.id}],
        },
        inputs=[stock],
    )
    vision = LLMVisionOp(
        [OpMessage(role="user", content="ignored")],
        image_source=news_s3.id,
        image_source_op=news_s3,
        config=GenerationConfig(model="llava-hf/llava-1.5-7b-hf"),
        rowwise_template="Summarize {Stock}.",
        rowwise_columns=[
            {"label": "Stock", "data": {"type": "list", "items": ["NVDA"]}},
        ],
    )
    tally = LambdaOp(
        [vision],
        fn=lambda items: [{"n": len(items[0])}],
        mode="list",
    )
    output = as_output("out", tally)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    tally_node = next(
        n for n in runtime_graph.nodes.values() if n.task_type == "python"
    )
    assert tally_node.data_spec["mode"] == "list"
    (vision_plan,) = tally_node.data_spec["plan"]
    assert vision_plan == {"kind": "stage", "node": vision.id}


def test_list_lambda_data_op_input_compiles() -> None:
    constant = data(["NVDA", "AAPL"])
    explode = LambdaOp([constant], _explode_fn, mode="list")
    output = as_output("out", explode)
    compiled = Graph.from_ops([output]).compile()

    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (explode_node,) = [
        n for n in runtime_graph.nodes.values() if n.task_type == "python"
    ]
    assert explode_node.data_spec["plan"] == [
        {"kind": "literal", "values": ["NVDA", "AAPL"]}
    ]


def test_list_lambda_accepts_retrieval_input_feeding_llm() -> None:
    """A list-mode Lambda reading [row-wise LLM, DataRetrievalOp] that feeds an
    LLM must compile, with a stage entry for the retrieval (one element per
    retrieval item) alongside the LLM entry, matching the standalone path."""
    stock = input_placeholder("Stock")
    retrieval = DataRetrievalOp(
        data_spec={
            "type": "lumid",
            "mode": "sql",
            "template": "SELECT * FROM news WHERE symbol = :symbol",
            "params": [{"name": "symbol", "node": stock.id}],
        },
        inputs=[stock],
    )
    judge = LLMChatOp(
        [OpMessage(role="user", content="Judge {r}")],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
        rowwise_template="Judge {r}",
        rowwise_columns=[{"label": "r", "node": retrieval.id, "path": "items.table"}],
    )
    judge.inputs.append(retrieval)
    combine = LambdaOp(
        [judge, retrieval],
        fn=lambda items: [{"n": len(items[0])}],
        mode="list",
    )
    llm = LLMChatOp(
        [OpMessage(role="user", content="Summarize {c}")],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
        rowwise_template="Summarize {c}",
        rowwise_columns=[{"label": "c", "node": combine.id, "path": "items.output"}],
    )
    llm.inputs.append(combine)
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    combine_node = next(
        n for n in runtime_graph.nodes.values() if n.task_type == "python"
    )
    assert combine_node.data_spec["mode"] == "list"
    judge_node = next(
        n
        for n in runtime_graph.nodes.values()
        if n.task_type == "inference" and "columns" in n.data_spec
    )
    assert combine_node.data_spec["plan"] == [
        {"kind": "stage", "node": judge_node.node_id, "flatten": True},
        {"kind": "stage", "node": retrieval.id},
    ]
    assert judge_node.node_id in combine_node.dependencies
    assert retrieval.id in combine_node.dependencies

    llm_node = next(
        n
        for n in runtime_graph.nodes.values()
        if n.task_type == "inference"
        and "columns" in n.data_spec
        and combine_node.node_id in n.dependencies
    )
    assert combine_node.node_id in llm_node.dependencies

    order = {node_id: idx for idx, node_id in enumerate(runtime_graph.node_order)}
    assert order[retrieval.id] < order[judge_node.node_id]
    assert order[judge_node.node_id] < order[combine_node.node_id]
    assert order[combine_node.node_id] < order[llm_node.node_id]


def test_api_op_reads_list_lambda_at_value_items_output() -> None:
    """An API-backed rowwise LLMChatOp reading a list-mode Lambda must bind the
    column to the python stage at ``value.items.output``."""
    stock = input_placeholder("Stock")
    explode = LambdaOp([stock], _explode_fn, mode="list")
    llm = LLMChatOp(
        [OpMessage(role="user", content="Summarize the rows.")],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {value}.",
        rowwise_columns=[
            {"label": "value", "node": explode.id, "path": "items.output.value"}
        ],
    )
    llm.inputs.append(explode)
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA", "AAPL"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (explode_node,) = [
        n for n in runtime_graph.nodes.values() if n.task_type == "python"
    ]
    assert explode_node.data_spec["mode"] == "list"
    (llm_node,) = runtime_graph.dsl_to_runtime[llm.id]
    node = runtime_graph.nodes[llm_node]
    assert node.data_spec["columns"][0] == {
        "label": "value",
        "node": explode_node.node_id,
        "path": "value.items.output.value",
    }


def _runtime_node(graph: RuntimeGraph, op_id: str) -> str:
    (node_id,) = graph.dsl_to_runtime[op_id][-1:]
    return node_id


def test_list_lambda_direct_read_nodes_classify_by_binding() -> None:
    """A node whose ``list`` column binds a list-mode Lambda one hop away runs
    once per element, so its output folds: a completion prompt, an embedding
    content and an image content. A node that reads the same Lambda through a
    template, a row-wise column or a retrieval parameter binds it elsewhere and
    keeps its own cardinality."""
    config = GenerationConfig(model="Qwen/Qwen3-8B")
    stock = input_placeholder("Stock")
    explode = LambdaOp([stock], _explode_fn, mode="list")
    completion = LLMCompletionOp(prompt=explode, config=config)
    embedding = EmbeddingOp(
        content=explode, config=GenerationConfig(model="BAAI/bge-small-en-v1.5")
    )
    image = ImageGenerationOp(
        content=explode, config=GenerationConfig(model="Tongyi-MAI/Z-Image-Turbo")
    )
    graph = RuntimeGraphBuilder().build(
        Graph.from_ops(
            [
                as_output("answer", completion),
                as_output("vectors", embedding),
                as_output("image", image),
            ]
        ).compile(Stock=["NVDA", "AAPL"])
    )
    direct = graph.list_lambda_direct_read_nodes()
    for op in (completion, embedding, image):
        spec = graph.nodes[_runtime_node(graph, op.id)].data_spec
        assert spec["type"] == "list"
        assert spec["node"] == _runtime_node(graph, explode.id)
        assert _runtime_node(graph, op.id) in direct
    assert _runtime_node(graph, explode.id) not in direct


def test_list_lambda_direct_read_skips_two_hops() -> None:
    """A completion prompted by another completion reads that completion, not
    the Lambda, so its list column does not bind a list-mode Lambda and it is
    not folded."""
    config = GenerationConfig(model="Qwen/Qwen3-8B")
    stock = input_placeholder("Stock")
    explode = LambdaOp([stock], _explode_fn, mode="list")
    first = LLMCompletionOp(prompt=explode, config=config)
    second = LLMCompletionOp(prompt=first, config=config)
    graph = RuntimeGraphBuilder().build(
        Graph.from_ops([as_output("second", second)]).compile(Stock=["NVDA", "AAPL"])
    )
    direct = graph.list_lambda_direct_read_nodes()
    assert _runtime_node(graph, first.id) in direct
    assert _runtime_node(graph, second.id) not in direct


def test_list_lambda_direct_read_skips_row_mode_lambda() -> None:
    """A completion whose prompt binds a row-mode Lambda's column keeps one
    value per row: only a list-mode Lambda emits one item per element behind
    that column."""
    stock = input_placeholder("Stock")
    rows = LambdaOp([stock], _row_fn)
    completion = LLMCompletionOp(
        prompt=rows, config=GenerationConfig(model="Qwen/Qwen3-8B")
    )
    graph = RuntimeGraphBuilder().build(
        Graph.from_ops([as_output("answer", completion)]).compile(
            Stock=["NVDA", "AAPL", "MSFT"]
        )
    )
    direct = graph.list_lambda_direct_read_nodes()
    assert _runtime_node(graph, completion.id) not in direct


def test_list_lambda_direct_read_skips_other_read_shapes() -> None:
    """An aggregate chat, a row-wise chat and a retrieval parameter read the
    Lambda through their own columns, so none of them folds by the list
    column rule."""
    stock = input_placeholder("Stock")
    explode = LambdaOp([stock], _explode_fn, mode="list")
    summary = LLMChatOp(
        [OpMessage(role="user", content="Summarize {df}")],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
        aggregate_table=[
            {"label": "value", "node": explode.id, "path": "items.output"}
        ],
    )
    summary.inputs.append(explode)
    rowwise = LLMChatOp(
        [OpMessage(role="user", content="Say {v}")],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
        rowwise_template="Say {v}",
        rowwise_columns=[
            {"label": "v", "node": explode.id, "path": "items.output.value"}
        ],
    )
    rowwise.inputs.append(explode)
    retrieval = DataRetrievalOp(
        data_spec={
            "type": "lumid",
            "mode": "s3",
            "template": "docs/{k}.txt",
            "params": [{"name": "k", "node": explode.id, "path": "items.output.value"}],
        },
        inputs=[explode],
    )
    graph = RuntimeGraphBuilder().build(
        Graph.from_ops(
            [
                as_output("summary", summary),
                as_output("said", rowwise),
                as_output("docs", retrieval),
            ]
        ).compile(Stock=["A", "B", "C"])
    )
    direct = graph.list_lambda_direct_read_nodes()
    for op, spec_type in ((summary, "graph_template"), (rowwise, "dataframe")):
        node_id = _runtime_node(graph, op.id)
        assert graph.nodes[node_id].data_spec["type"] == spec_type
        assert node_id not in direct
    assert _runtime_node(graph, retrieval.id) not in direct
