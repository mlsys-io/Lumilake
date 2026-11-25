"""The golden builder graphs and the wire shapes this checkout builds for them.
The golden fixture holds the wire origin/main's builders produced for the same
graphs."""

from typing import Any, Literal

from lumilake_server.common import ApiConfig, GenerationConfig
from lumilake_server.data_specs import (
    PythonData,
    RetrievalParam,
    build_column_table,
)
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
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
from lumilake_server.ops.classification_ops import (
    ClassificationConfig,
    ClassificationOp,
)
from lumilake_server.ops.http_fetch_ops import HttpFetchOp
from lumilake_server.runtime import python_step
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.runtime.runtime_ops import RuntimeOp
from lumilake_server.utils.retrieval_spec import RetrievalSpec

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


class FixedIds:
    """Id counters, so a graph builds the same node ids on any checkout."""

    counter = 0

    @classmethod
    def reset(cls) -> None:
        cls.counter = 0

    @classmethod
    def next(cls) -> str:
        cls.counter += 1
        return f"node{cls.counter:04d}"


def _config(model: str = MODEL) -> GenerationConfig:
    return GenerationConfig(model=model)


def _lumid_spec(
    mode: Literal["sql", "s3", "agent"], template: str, params: list
) -> RetrievalSpec:
    return RetrievalSpec(type="lumid", mode=mode, template=template, params=params)


def explode_rows(items):
    return [{"value": v} for v in items[0]]


def shout_rows(items):
    return [{"output": str(v)} for v in items[0]]


def _python_rowwise():
    stock = input_placeholder("Stock")
    tone = input_placeholder("Tone")
    retrieval = DataRetrievalOp(
        data_spec=_lumid_spec(
            "sql",
            "SELECT * FROM t WHERE s = '{label}'",
            [{"label": "label", "node": stock.id}],
        ),
        inputs=[stock],
    )
    explode = LambdaOp([stock], fn=explode_rows, mode="list")
    rowwise = LLMChatOp(
        [OpMessage(role="user", content="hello")],
        rowwise_template="Summarize {value}",
        rowwise_columns=build_column_table(
            [
                {"label": "value", "node": explode.id, "path": "items.output.value"},
                {"label": "tone", "data": {"type": "list", "items": ["a", "b"]}},
            ],
            "rowwise_columns",
        ),
        rowwise_seed=RowwiseSeed(node=explode.id, path="items.output.value"),
        rowwise_seed_op=explode,
        config=_config(),
    )
    plain = LLMChatOp([OpMessage(role="user", content="plain")], config=_config())
    row_lambda = LambdaOp([plain], fn=shout_rows)
    fetch = HttpFetchOp("https://example.com/{query}", stock)
    classification = ClassificationOp(
        stock, tone, config=ClassificationConfig(model="cross-encoder/ms-marco")
    )
    embedding = EmbeddingOp(
        content=DataOp(data=["one", "two"]), config=_config("BAAI/bge-m3")
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


def _completion_vision_image():
    stock = input_placeholder("Stock")
    prompt = input_placeholder("Prompt")
    completion = LLMCompletionOp(prompt=prompt, config=_config())
    images = DataRetrievalOp(
        data_spec=_lumid_spec(
            "s3",
            "example-data/news/images/{symbol}.png",
            [{"label": "symbol", "node": stock.id}],
        ),
        inputs=[stock],
    )
    vision = LLMVisionOp(
        [OpMessage(role="user", content="ignored")],
        image_source=images.id,
        image_source_op=images,
        config=GenerationConfig(model="llava-hf/llava-1.5-7b-hf"),
        rowwise_template="Summarize {Stock}.",
        rowwise_columns=build_column_table(
            [{"label": "Stock", "data": {"type": "list", "items": ["NVDA"]}}],
            "rowwise_columns",
        ),
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


def _grouped_fetch_classify():
    """A grouped upstream: the fetch issues one request per row and the
    classification scores one pair per row, each reading a flatten step."""
    stock = input_placeholder("Stock")
    prompt = input_placeholder("Prompt")
    rowwise = LLMChatOp(
        [OpMessage(role="user", content=prompt)],
        config=_config(),
        rowwise_template="{prompt}",
        rowwise_columns=build_column_table(
            [{"label": "prompt", "node": prompt.id, "path": ""}], "rowwise_columns"
        ),
    )
    fetch = HttpFetchOp("https://example.com/{query}", rowwise)
    candidates = LambdaOp([stock], fn=shout_rows, mode="list")
    classification = ClassificationOp(
        fetch,
        candidates,
        config=ClassificationConfig(model="cross-encoder/ms-marco"),
    )
    outputs = [
        as_output("a", fetch),
        as_output("b", classification),
    ]
    return RuntimeGraphBuilder().build(
        Graph.from_ops(outputs).compile(Stock=["NVDA", "AAPL"], Prompt=["one", "two"])
    )


def _aggregate_retrieval_python():
    """An aggregate-table chat, an agent retrieval and a large-literal python
    step, whose bytes land in generated code."""
    stock = input_placeholder("Stock")
    upstream = LLMChatOp([OpMessage(role="user", content=stock)], config=_config())
    digest = LLMChatOp(
        [OpMessage(role="user", content="Summarize the table")],
        config=_config(),
        aggregate_table=build_column_table(
            [{"label": "summary", "node": upstream.id, "path": "items.output"}],
            "aggregate_table",
        ),
    )
    agent = DataRetrievalOp(
        data_spec=RetrievalSpec(
            type="lumid",
            mode="agent",
            description="Find the revenue for {symbol}",
            params=[RetrievalParam(label="symbol", node=stock.id)],
            schema_scope="finance",
            max_steps=4,
        ),
        inputs=[stock],
    )
    big = LambdaOp(
        [DataOp(data=[f"row-{index}" for index in range(200)])], fn=shout_rows
    )
    outputs = [
        as_output("a", digest),
        as_output("b", agent),
        as_output("c", big),
    ]
    return RuntimeGraphBuilder().build(Graph.from_ops(outputs).compile(Stock=["NVDA"]))


GRAPHS = {
    "python_rowwise": _python_rowwise,
    "completion_vision_image": _completion_vision_image,
    "grouped_fetch_classify": _grouped_fetch_classify,
    "aggregate_retrieval_python": _aggregate_retrieval_python,
}


def _generated_code(op: RuntimeOp) -> str | None:
    """The python task code a python step generates, or None for other tasks."""
    spec = op.data_spec
    if not isinstance(spec, PythonData):
        return None
    runtime = python_step.runtime_blob()
    published = {runtime.key: runtime, **python_step.literal_blobs(spec.plan)}
    return python_step.flowmesh_spec(op.node_id, spec, published)["code"]


def build(graph_name: str) -> dict[str, dict[str, Any]]:
    """One graph's node specs and generated code, keyed by node id."""
    FixedIds.reset()
    built = GRAPHS[graph_name]()
    entries: dict[str, dict[str, Any]] = {}
    for node_id, op in built.nodes.items():
        entry: dict[str, Any] = {
            "task_type": op.task_type,
            "data_spec": op.data_spec.model_dump(exclude_none=True),
        }
        code = _generated_code(op)
        if code is not None:
            entry["code"] = code
        entries[node_id] = entry
    return entries
