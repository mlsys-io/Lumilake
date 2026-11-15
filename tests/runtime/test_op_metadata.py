"""Each op declares the row shape and profile eligibility its lowering relies on."""

import pytest

from lumilake import envs
from lumilake_server.common import GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    DataRetrievalOp,
    EmbeddingOp,
    ImageGenerationOp,
    LambdaOp,
    LLMChatOp,
    LLMCompletionOp,
    LLMVisionOp,
    Op,
    OpMessage,
    RowShape,
    as_output,
    input_placeholder,
)
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder

_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")


def _row(items):
    return items[0]


def _whole(items):
    return list(items)


def _sql_retrieval() -> DataRetrievalOp:
    return DataRetrievalOp(
        data_spec={
            "type": "lumid",
            "mode": "sql",
            "template": "select a from t",
            "params": [],
        }
    )


def _cases() -> dict[str, tuple[Op, RowShape, bool]]:
    stock = input_placeholder("Stock")
    chat = LLMChatOp(
        [OpMessage(role="user", content=stock)], config=GenerationConfig(model=_MODEL)
    )
    rowwise_chat = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model=_MODEL),
        rowwise_template="{symbol}",
        rowwise_columns=[{"label": "symbol", "data": {"type": "list", "items": ["a"]}}],
    )
    return {
        "chat": (chat, RowShape.PER_ROW, False),
        "rowwise_chat": (rowwise_chat, RowShape.GROUPED, False),
        "completion": (
            LLMCompletionOp(prompt="hi", config=GenerationConfig(model=_MODEL)),
            RowShape.PER_ROW,
            False,
        ),
        "vision": (
            LLMVisionOp(
                [OpMessage(role="user", content="describe")],
                image_source=stock.id,
                image_source_op=stock,
                config=GenerationConfig(model=_MODEL),
            ),
            RowShape.PER_ROW,
            False,
        ),
        "image_generation": (
            ImageGenerationOp("a cat", config=GenerationConfig(model=_MODEL)),
            RowShape.PER_ROW,
            False,
        ),
        "embedding": (
            EmbeddingOp("text", config=GenerationConfig(model="bge-small")),
            RowShape.PER_ROW,
            False,
        ),
        "sql_retrieval": (_sql_retrieval(), RowShape.GROUPED, True),
        "s3_retrieval": (
            DataRetrievalOp(
                data_spec={"type": "lumid", "mode": "s3", "template": "s3://b/{k}"}
            ),
            RowShape.PER_ROW,
            True,
        ),
        "agent_retrieval": (
            DataRetrievalOp(
                data_spec={
                    "type": "lumid",
                    "mode": "agent",
                    "description": "Find a price.",
                }
            ),
            RowShape.GROUPED,
            False,
        ),
        "list_lambda": (
            LambdaOp([stock], _whole, mode="list"),
            RowShape.WHOLE_LIST,
            False,
        ),
        "row_lambda": (LambdaOp([stock], _row), RowShape.PER_ROW, False),
        "aligned_lambda": (
            LambdaOp([stock], _whole, mode="aligned"),
            RowShape.PER_ROW,
            False,
        ),
    }


@pytest.mark.parametrize("name", sorted(_cases()))
def test_op_declares_the_shape_its_lowering_emits(name: str) -> None:
    op, shape, profilable = _cases()[name]

    assert op.row_shape is shape
    assert op.profilable is profilable


@pytest.mark.parametrize("name", sorted(_cases()))
def test_metadata_matches_the_node_its_lowering_emits(name: str) -> None:
    op, shape, profilable = _cases()[name]
    graph = Graph.from_ops([as_output("result", op)])
    inputs = {"Stock": ["NVDA", "AAPL"]} if graph.input_ops else {}
    compiled = graph.compile(**inputs)

    runtime = RuntimeGraphBuilder().build(compiled)
    node = runtime.nodes[op.id]

    assert runtime.dsl_to_runtime[op.id][-1] == op.id
    if isinstance(op, LLMChatOp) and shape is RowShape.GROUPED:
        assert node.data_spec["type"] == "dataframe"
    if isinstance(op, LambdaOp):
        assert (node.data_spec.get("mode") == "list") is (shape is RowShape.WHOLE_LIST)
    if isinstance(op, DataRetrievalOp):
        assert node.data_spec["mode"] == op.data_spec["mode"]

    profile = RuntimeGraphBuilder().build(compiled, task_type_override="data_profile")
    profiled = [
        candidate
        for candidate in profile.nodes.values()
        if candidate.task_type == "data_profiling"
    ]
    assert bool(profiled) is profilable
