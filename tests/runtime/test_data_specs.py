"""Every task type's ``data_spec`` round-trips through its typed model.

The expected dicts are what the graph builders wrote before the models existed,
so the serialized wire shape stays byte-identical.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from lumilake import envs
from lumilake_server.common import ApiConfig, GenerationConfig
from lumilake_server.data_specs import (
    DataFrame,
    S3ProfileData,
    SqlProfileData,
    dump_data_spec,
    parse_data_spec,
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
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from tests.support import data_spec_golden

_LUMID_URL = "http://lumid-data"
_LUMID_TOKEN = "test-token"
_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", _LUMID_URL)
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", _LUMID_TOKEN)
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "test-token")


def _config(model: str = _MODEL) -> GenerationConfig:
    return GenerationConfig(model=model)


def _sql_spec(template: str, params: list) -> dict[str, Any]:
    return {"type": "lumid", "mode": "sql", "template": template, "params": params}


def _s3_spec(template: str, params: list) -> dict[str, Any]:
    return {"type": "lumid", "mode": "s3", "template": template, "params": params}


def _build(outputs: list, **inputs: list[str]):
    return RuntimeGraphBuilder().build(Graph.from_ops(outputs).compile(**inputs))


def _by_task(graph, task_type: str):
    return [op for op in graph.nodes.values() if op.task_type == task_type]


def _assert_round_trips(op) -> None:
    wire = dump_data_spec(op.data_spec)
    assert dump_data_spec(parse_data_spec(op.task_type, wire)) == wire


def test_python_step_shapes_round_trip() -> None:
    stock = input_placeholder("Stock")
    explode = LambdaOp(
        [stock], fn=lambda items: [{"value": v} for v in items[0]], mode="list"
    )
    row = LambdaOp([explode], fn=lambda items: [{"output": str(v)} for v in items[0]])
    graph = _build(
        [as_output("a", explode), as_output("b", row)], Stock=["NVDA", "AAPL"]
    )
    for op in _by_task(graph, "python"):
        _assert_round_trips(op)


def test_list_data_shapes_round_trip() -> None:
    prompt = input_placeholder("Prompt")
    embedding = EmbeddingOp(content=DataOp(data=["one", "two"]), config=_config("bge"))
    completion = LLMCompletionOp(prompt=prompt, config=_config())
    feed = LLMChatOp([OpMessage(role="user", content="x")], config=_config())
    embed_from_op = EmbeddingOp(content=feed, config=_config("bge"))
    graph = _build(
        [
            as_output("a", embedding),
            as_output("b", completion),
            as_output("c", embed_from_op),
        ],
        Prompt=["hi"],
    )
    assert _by_task(graph, "embedding")
    for op in _by_task(graph, "embedding") + _by_task(graph, "inference"):
        _assert_round_trips(op)


def test_rowwise_dataframe_round_trips() -> None:
    stock = input_placeholder("Stock")
    explode = LambdaOp(
        [stock], fn=lambda items: [{"value": v} for v in items[0]], mode="list"
    )
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
    graph = _build([as_output("a", rowwise)], Stock=["NVDA", "AAPL"])
    (op,) = _by_task(graph, "inference")
    assert isinstance(op.data_spec, DataFrame)
    _assert_round_trips(op)


def test_graph_template_and_api_shapes_round_trip() -> None:
    stock = input_placeholder("Stock")
    retrieval = DataRetrievalOp(
        data_spec=_sql_spec("SELECT * FROM t", [{"label": "s", "node": stock.id}]),
        inputs=[stock],
    )
    chat = LLMChatOp([OpMessage(role="user", content=retrieval)], config=_config())
    api = LLMChatOp(
        [OpMessage(role="user", content="api")],
        config=GenerationConfig(
            model="gpt-x",
            api=ApiConfig(
                url="https://llm.example/v1/chat/completions",
                authorization="Bearer x",
            ),
        ),
    )
    graph = _build([as_output("a", chat), as_output("b", api)], Stock=["NVDA"])
    assert _by_task(graph, "api")
    for op in graph.nodes.values():
        if op.task_type in {"api", "inference", "data_retrieval"}:
            _assert_round_trips(op)


def test_retrieval_and_profile_shapes_round_trip() -> None:
    stock = input_placeholder("Stock")
    sql = DataRetrievalOp(
        data_spec=_sql_spec(
            "SELECT title FROM news WHERE symbol = '{symbol}'",
            [{"label": "symbol", "node": stock.id}],
        ),
        inputs=[stock],
    )
    s3 = DataRetrievalOp(
        data_spec=_s3_spec(
            "data/{symbol}.json", [{"label": "symbol", "node": stock.id}]
        ),
        inputs=[stock],
    )
    online = _build([as_output("a", sql), as_output("b", s3)], Stock=["NVDA"])
    for op in _by_task(online, "data_retrieval"):
        _assert_round_trips(op)

    graph = RuntimeGraphBuilder().build(
        Graph.from_ops([as_output("a", sql), as_output("b", s3)]).compile(
            Stock=["NVDA"]
        ),
        task_type_override="data_profile",
    )
    profiles = _by_task(graph, "data_profiling")
    assert {type(op.data_spec) for op in profiles} == {SqlProfileData, S3ProfileData}
    for op in profiles:
        _assert_round_trips(op)


def test_vision_and_image_shapes_round_trip() -> None:
    stock = input_placeholder("Stock")
    prompt = input_placeholder("Prompt")
    images = DataRetrievalOp(
        data_spec=_s3_spec("images/{s}.png", [{"label": "s", "node": stock.id}]),
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
    graph = _build(
        [as_output("a", vision), as_output("b", image)], Stock=["NVDA"], Prompt=["hi"]
    )
    for op in graph.nodes.values():
        if op.task_type in {"inference", "embedding", "python", "omni_text2image"}:
            _assert_round_trips(op)


def test_classification_and_fetch_shapes_round_trip() -> None:
    stock = input_placeholder("Stock")
    tone = input_placeholder("Tone")
    fetch = HttpFetchOp("https://example.com/{query}", stock)
    classification = ClassificationOp(
        stock, tone, config=ClassificationConfig(model="cross-encoder/ms-marco")
    )
    graph = _build(
        [as_output("a", fetch), as_output("b", classification)],
        Stock=["NVDA", "AAPL"],
        Tone=["x", "y"],
    )
    for op in graph.nodes.values():
        if op.task_type in {"api", "classification"}:
            _assert_round_trips(op)


def test_unknown_data_spec_key_is_refused() -> None:
    with pytest.raises(ValueError, match="invalid data_spec"):
        parse_data_spec("python", {"plan": [], "bogus": 1})

    with pytest.raises(ValueError, match="invalid data_spec"):
        parse_data_spec("inference", {"type": "list", "items": [], "bogus": 1})


_GOLDEN = Path(__file__).resolve().parent / "fixtures" / "data_spec_golden.json"
# A builder-assigned node id, including its derived step suffix.
_NODE_ID = re.compile(r"[A-Za-z0-9_]{20,}")


def _normalized(spec: dict[str, Any]) -> Any:
    """The spec with random node ids replaced by one placeholder, so ids that
    differ between runs do not affect the comparison."""
    if isinstance(spec, dict):
        return {key: _normalized(value) for key, value in spec.items()}
    if isinstance(spec, list):
        return [_normalized(value) for value in spec]
    if isinstance(spec, str):
        return _NODE_ID.sub("<node>", spec)
    return spec


def _shapes(specs: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Task type -> its sorted serialized specs, node ids normalized."""
    shapes: dict[str, list[str]] = {}
    for entry in specs.values():
        wire = _normalized(entry["data_spec"])
        shapes.setdefault(entry["task_type"], []).append(
            json.dumps(wire, sort_keys=True)
        )
    return {task: sorted(group) for task, group in shapes.items()}


@pytest.mark.parametrize("graph_name", ["python_rowwise", "completion_vision_image"])
def test_built_data_specs_match_the_golden_wire_shape(graph_name: str) -> None:
    """The serialized ``data_spec`` of every task type is byte-identical to what
    the builders wrote before the models, node ids aside."""
    golden = json.loads(_GOLDEN.read_text())[graph_name]
    built = _GRAPHS[graph_name]()
    built_specs = {
        node_id: {
            "task_type": op.task_type,
            "data_spec": dump_data_spec(op.data_spec),
        }
        for node_id, op in built.nodes.items()
    }
    assert _shapes(built_specs) == _shapes(golden)


def _python_rowwise_graph():
    return data_spec_golden.python_rowwise()


def _completion_vision_image_graph():
    return data_spec_golden.completion_vision_image()


_GRAPHS = {
    "python_rowwise": _python_rowwise_graph,
    "completion_vision_image": _completion_vision_image_graph,
}
