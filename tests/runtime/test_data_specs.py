"""The typed ``data_spec`` each task type carries, and its wire shape."""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from lumilake import envs
from lumilake_server.common import ApiConfig, GenerationConfig
from lumilake_server.data_specs import dump_data_spec, parse_data_spec
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
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder

_GOLDEN = Path(__file__).resolve().parent / "fixtures" / "data_spec_golden.json"
_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
_LUMID_URL = "http://lumid-data"
_LUMID_TOKEN = "test-token"
# A builder-assigned node id, including its derived step suffix.
_NODE_ID = re.compile(r"[A-Za-z0-9_]{20,}")


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", _LUMID_URL)
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", _LUMID_TOKEN)
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "test-token")


def _config(model: str = _MODEL) -> GenerationConfig:
    return GenerationConfig(model=model)


def _lumid_spec(mode: str, template: str, params: list) -> dict[str, Any]:
    return {"type": "lumid", "mode": mode, "template": template, "params": params}


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
        rowwise_columns=[
            {"label": "value", "node": explode.id, "path": "items.output.value"},
            {"label": "tone", "data": {"type": "list", "items": ["a", "b"]}},
        ],
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
        rowwise_columns=[
            {"label": "Stock", "data": {"type": "list", "items": ["NVDA"]}}
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


_GRAPHS = {
    "python_rowwise": _python_rowwise,
    "completion_vision_image": _completion_vision_image,
}


def _normalized(value: Any) -> Any:
    """The spec with random node ids replaced by one placeholder."""
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    if isinstance(value, str):
        return _NODE_ID.sub("<node>", value)
    return value


def _shapes(specs: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Task type -> its sorted serialized specs, node ids normalized."""
    shapes: dict[str, list[str]] = {}
    for entry in specs.values():
        wire = _normalized(entry["data_spec"])
        shapes.setdefault(entry["task_type"], []).append(
            json.dumps(wire, sort_keys=True)
        )
    return {task: sorted(group) for task, group in shapes.items()}


@pytest.mark.parametrize("graph_name", sorted(_GRAPHS))
def test_every_task_spec_round_trips(graph_name: str) -> None:
    """Each task's serialized spec validates back into a model that serializes
    identically."""
    for op in _GRAPHS[graph_name]().nodes.values():
        wire = dump_data_spec(op.data_spec)
        assert dump_data_spec(parse_data_spec(op.task_type, wire)) == wire


@pytest.mark.parametrize("graph_name", sorted(_GRAPHS))
def test_built_data_specs_match_the_golden_wire_shape(graph_name: str) -> None:
    """The serialized spec of every task type matches the builders' original output."""
    golden = json.loads(_GOLDEN.read_text())[graph_name]
    built = _GRAPHS[graph_name]()
    built_specs = {
        node_id: {"task_type": op.task_type, "data_spec": dump_data_spec(op.data_spec)}
        for node_id, op in built.nodes.items()
    }
    assert _shapes(built_specs) == _shapes(golden)


def test_unknown_task_type_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown task type"):
        parse_data_spec("query", {"type": "list", "items": []})


def test_unknown_data_spec_key_is_refused() -> None:
    with pytest.raises(ValueError, match="invalid data_spec"):
        parse_data_spec("python", {"plan": [], "bogus": 1})
