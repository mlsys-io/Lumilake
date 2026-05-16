from lumilake_server.common import GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    DataRetrievalOp,
    LLMChatOp,
    LLMVisionOp,
    OpMessage,
    as_output,
    input_placeholder,
)
from lumilake_server.parser.n8n import parse_n8n_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder


def test_runtime_graph_builder_accepts_input_used_only_by_retrieval() -> None:
    stock = input_placeholder("Stock")
    retrieval = DataRetrievalOp(
        data_spec={
            "type": "sql",
            "connection_string": "postgresql://example",
            "template": "SELECT * FROM t WHERE symbol = :symbol",
            "params": [{"name": "symbol", "node": stock.id}],
        },
        inputs=[stock],
    )
    llm = LLMChatOp(
        [OpMessage(role="user", content=retrieval)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)

    assert len(runtime_graph.nodes) == 2
    assert retrieval.id in runtime_graph.nodes
    assert llm.id in runtime_graph.nodes


def test_runtime_graph_builder_emits_topological_runtime_node_order() -> None:
    stock = input_placeholder("Stock")
    planner = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    retrieval = DataRetrievalOp(
        data_spec={
            "type": "sql",
            "connection_string": "postgresql://example",
            "template": "SELECT * FROM t WHERE query = :query",
            "params": [
                {
                    "label": "query",
                    "node": planner.id,
                    "path": "items.output",
                }
            ],
        },
        inputs=[planner],
    )
    report = LLMChatOp(
        [OpMessage(role="user", content=retrieval)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    output = as_output("result", report)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)

    order_index = {
        node_id: idx for idx, node_id in enumerate(runtime_graph.node_order)
    }
    assert runtime_graph.node_order == runtime_graph.topological_order()
    assert order_index[planner.id] < order_index[retrieval.id]
    assert order_index[retrieval.id] < order_index[report.id]


def test_runtime_graph_builder_supports_s3_retrieval_as_vlm_image_source() -> None:
    stock = input_placeholder("Stock")
    retrieval = DataRetrievalOp(
        data_spec={
            "type": "s3",
            "connection_string": "s3://example",
            "template": "example-data/news/images/{symbol}.png",
            "params": [{"name": "symbol", "node": stock.id}],
        },
        inputs=[stock],
    )
    vision = LLMVisionOp(
        [OpMessage(role="user", content="Describe the image briefly.")],
        image_source=retrieval.id,
        image_source_op=retrieval,
        config=GenerationConfig(model="llava-hf/llava-1.5-7b-hf"),
    )
    output = as_output("result", vision)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)

    embedding_id = f"{vision.id}_embedding"
    assert embedding_id in runtime_graph.nodes
    embedding = runtime_graph.nodes[embedding_id]
    assert embedding.task_type == "embedding"
    assert embedding.data_spec["node"] == retrieval.id
    assert embedding.data_spec["path"] == "items.content"
    assert embedding.dependencies == (retrieval.id,)


def test_runtime_graph_builder_supports_rowwise_vlm_template() -> None:
    stock = input_placeholder("Stock")
    news_sql = DataRetrievalOp(
        data_spec={
            "type": "sql",
            "connection_string": "postgresql://example",
            "template": "SELECT title FROM news WHERE symbol = :symbol",
            "params": [{"name": "symbol", "node": stock.id}],
        },
        inputs=[stock],
    )
    news_s3 = DataRetrievalOp(
        data_spec={
            "type": "s3",
            "connection_string": "s3://example",
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
        rowwise_template="Summarize {Stock} with title {title}.",
        rowwise_columns=[
            {"label": "Stock", "data": {"type": "list", "items": ["NVDA"]}},
            {
                "label": "title",
                "node": news_sql.id,
                "path": "items.table.title",
            },
        ],
        system_messages=["You are concise."],
    )
    output = as_output("result", vision)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA"])

    runtime_graph = RuntimeGraphBuilder().build(compiled)

    embedding_id = f"{vision.id}_embedding"
    vlm = runtime_graph.nodes[vision.id]
    template = vlm.data_spec["template"]
    columns = template["columns"]
    assert {"label": "Stock", "data": {"type": "list", "items": ["NVDA"]}} in columns
    assert {
        "label": "title",
        "node": news_sql.id,
        "path": "items.table.title",
    } in columns
    assert {"role": "user", "content": "Summarize {Stock} with title {title}."} in (
        template["options"]["format"]["messages"]
    )
    assert embedding_id in vlm.dependencies
    assert news_sql.id in vlm.dependencies


def test_runtime_graph_builder_omits_unused_upstream_context_column() -> None:
    workflow = {
        "nodes": [
            {
                "parameters": {"options": {}},
                "type": "@n8n/n8n-nodes-langchain.chatTrigger",
                "name": "Input",
                "id": "input-node",
                "typeVersion": 1.4,
            },
            {
                "parameters": {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "options": {"maxTokens": 64},
                },
                "type": "@n8n/n8n-nodes-langchain.lmOpenHuggingFaceInference",
                "name": "Model",
                "id": "model-node",
                "typeVersion": 1,
            },
            {
                "parameters": {
                    "promptType": "define",
                    "text": "=First stage for {{ $('Input') }}",
                    "messages": {"messageValues": []},
                },
                "type": "@n8n/n8n-nodes-langchain.chainLlm",
                "name": "Upstream",
                "id": "upstream-node",
                "typeVersion": 1.9,
                "notes": '{"op-type":"text-generation","is-output":false}',
            },
            {
                "parameters": {
                    "promptType": "define",
                    "text": "=Second stage for {{ $('Upstream') }}",
                    "messages": {"messageValues": []},
                },
                "type": "@n8n/n8n-nodes-langchain.chainLlm",
                "name": "Downstream",
                "id": "downstream-node",
                "typeVersion": 1.9,
                "notes": '{"op-type":"text-generation","is-output":false}',
            },
        ],
        "connections": {
            "Input": {"main": [[{"node": "Upstream", "type": "main", "index": 0}]]},
            "Upstream": {
                "main": [[{"node": "Downstream", "type": "main", "index": 0}]]
            },
            "Model": {
                "ai_languageModel": [
                    [
                        {"node": "Upstream", "type": "ai_languageModel", "index": 0},
                        {"node": "Downstream", "type": "ai_languageModel", "index": 0},
                    ]
                ]
            },
        },
    }
    payload = {
        "graphs": [
            {"name": "graph", "workflow": workflow, "inputs": {"Input": ["NVDA"]}}
        ]
    }
    parsed = parse_n8n_payload(payload)
    spec = parsed["graph"]
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    upstream_id = next(nid for nid in runtime_graph.nodes if "Upstream" in nid)
    downstream_id = next(nid for nid in runtime_graph.nodes if "Downstream" in nid)
    columns = runtime_graph.nodes[downstream_id].data_spec["template"]["columns"]
    upstream_columns = [col for col in columns if col.get("node") == upstream_id]
    assert any(col.get("path") == "items.output" for col in upstream_columns)
    assert all(col.get("path") != "items.metadata.prompt" for col in upstream_columns)
