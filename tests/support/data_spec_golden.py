"""Build the golden data_spec dicts with builder code shared by both trees."""

import json
import sys

from lumilake import envs

envs.LUMID_DATA_URL = "http://lumid-data"
envs.LUMID_DATA_TOKEN = "test-token"
envs.RUNTIME_TOKEN = "test-token"

from lumilake_server.common import ApiConfig, GenerationConfig  # noqa: E402
from lumilake_server.graphs import Graph  # noqa: E402
from lumilake_server.ops import (  # noqa: E402
    DataOp,
    DataRetrievalOp,
    EmbeddingOp,
    ImageGenerationOp,
    LambdaOp,
    LLMChatOp,
    LLMCompletionOp,
    LLMVisionOp,
    OpMessage,
    RowwiseSeed,
    as_output,
    input_placeholder,
)
from lumilake_server.ops.classification_ops import (  # noqa: E402
    ClassificationConfig,
    ClassificationOp,
)
from lumilake_server.ops.http_fetch_ops import HttpFetchOp  # noqa: E402
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder  # noqa: E402

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def config(model: str = MODEL) -> GenerationConfig:
    return GenerationConfig(model=model)


def sql_spec(template: str, params: list) -> dict:
    return {"type": "lumid", "mode": "sql", "template": template, "params": params}


def s3_spec(template: str, params: list) -> dict:
    return {"type": "lumid", "mode": "s3", "template": template, "params": params}


def explode_rows(items):
    return [{"value": v} for v in items[0]]


def shout_rows(items):
    return [{"output": str(v)} for v in items[0]]


def python_rowwise():
    stock = input_placeholder("Stock")
    tone = input_placeholder("Tone")
    retrieval = DataRetrievalOp(
        data_spec=sql_spec(
            "SELECT * FROM t WHERE s = '{label}'",
            [{"label": "label", "node": stock.id}],
        ),
        inputs=[stock],
    )
    explode = LambdaOp([stock], fn=explode_rows, mode="list")
    rowwise = LLMChatOp(
        [OpMessage(role="user", content="hello")],
        rowwise_template="Summarize {value}",
        rowwise_columns=[
            {"label": "value", "node": explode.id, "path": "items.output.value"},
            {"label": "tone", "data": {"type": "list", "items": ["a", "b"]}},
        ],
        rowwise_seed=RowwiseSeed(node=explode.id, path="items.output.value"),
        rowwise_seed_op=explode,
        config=config(),
    )
    plain = LLMChatOp([OpMessage(role="user", content="plain")], config=config())
    row_lambda = LambdaOp([plain], fn=shout_rows)
    fetch = HttpFetchOp("https://example.com/{query}", stock)
    classification = ClassificationOp(
        stock, tone, config=ClassificationConfig(model="cross-encoder/ms-marco")
    )
    embedding = EmbeddingOp(
        content=DataOp(data=["one", "two"]), config=config("BAAI/bge-m3")
    )
    api_chat = LLMChatOp(
        [OpMessage(role="user", content="api")],
        config=GenerationConfig(
            model="gpt-x",
            api=ApiConfig(
                url="https://llm.example/v1/chat/completions",
                authorization="Bearer x",
            ),
        ),
    )
    outputs = [
        as_output("a", embedding),
        as_output("b", row_lambda),
        as_output("c", rowwise),
        as_output("d", retrieval),
        as_output("e", fetch),
        as_output("f", classification),
        as_output("g", api_chat),
    ]
    return RuntimeGraphBuilder().build(
        Graph.from_ops(outputs).compile(Stock=["NVDA", "AAPL"], Tone=["x", "y"])
    )


def completion_vision_image():
    stock = input_placeholder("Stock")
    prompt = input_placeholder("Prompt")
    completion = LLMCompletionOp(prompt=prompt, config=config())
    news_s3 = DataRetrievalOp(
        data_spec=s3_spec(
            "example-data/news/images/{symbol}.png",
            [{"label": "symbol", "node": stock.id}],
        ),
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
    image = ImageGenerationOp(
        content=prompt, config=GenerationConfig(model="black-forest-labs/FLUX.1-dev")
    )
    outputs = [
        as_output("a", completion),
        as_output("b", vision),
        as_output("c", image),
    ]
    return RuntimeGraphBuilder().build(
        Graph.from_ops(outputs).compile(Stock=["NVDA"], Prompt=["hi"])
    )


GRAPHS = {
    "python_rowwise": python_rowwise,
    "completion_vision_image": completion_vision_image,
}


def main() -> int:
    payload: dict[str, dict[str, dict[str, object]]] = {}
    for name, build in GRAPHS.items():
        graph = build()
        payload[name] = {}
        for node_id, op in graph.nodes.items():
            spec = op.data_spec
            if hasattr(spec, "model_dump"):
                spec = spec.model_dump(exclude_none=True)
            payload[name][node_id] = {"task_type": op.task_type, "data_spec": spec}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
