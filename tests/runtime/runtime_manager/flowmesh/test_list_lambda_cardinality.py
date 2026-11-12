"""A node whose list column binds a list-mode ``LambdaOp`` directly runs once
per element, so its several items collapse into exactly one whole-list value
per run: the server's slice merge accepts only one value for such an output.
The row-wise arm keeps folding an output in the list-Lambda cardinality set, as
it does for any other shape."""

import json
from pathlib import Path
from typing import Any

import pytest

import lumilake_server.runtime.runtime_manager.flowmesh as fm_mod
from lumilake_server.common import GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    DataRetrievalOp,
    LambdaOp,
    LLMChatOp,
    LLMCompletionOp,
    OpMessage,
    as_output,
    input_placeholder,
)
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager

_MODEL = "BAAI/bge-small-en-v1.5"


class _StubResults:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.downloaded: list[tuple[str, str]] = []

    async def download_file(
        self, task_id: str, remote_path: str, local_path: Path
    ) -> None:
        self.downloaded.append((task_id, remote_path))
        Path(local_path).write_bytes(self._payload)


class _StubFm:
    def __init__(self, payload: bytes) -> None:
        self.results = _StubResults(payload)


class _StubStorage:
    def __init__(self) -> None:
        self.saved: list[tuple[str, str, bytes, str]] = []

    def save_artifact(
        self, request_id: str, filename: str, data: bytes, content_type: str
    ) -> str:
        self.saved.append((request_id, filename, data, content_type))
        return f"blob://{request_id}/{filename}"


@pytest.fixture
def _archive_env(monkeypatch: pytest.MonkeyPatch) -> tuple[_StubFm, _StubStorage]:
    stub_fm = _StubFm(b"safetensors-bytes")
    stub_storage = _StubStorage()
    monkeypatch.setattr(fm_mod, "flowmesh_for_context", lambda: stub_fm)
    monkeypatch.setattr(fm_mod, "get_job_storage", lambda: stub_storage)
    monkeypatch.setattr(fm_mod.envs, "S3_ARCHIVE_PREFIX", "s3://bucket/prefix")
    return stub_fm, stub_storage


async def _aggregate(
    items: list[dict],
    *,
    list_lambda_direct_read: bool,
    list_lambda_cardinality: bool = False,
) -> list[str]:
    return await FlowmeshRuntimeManager()._aggregate_output_node(
        output_op_id="output",
        output_task_id="task-1",
        request_id="req-1",
        items=items,
        output_path=None,
        list_lambda=False,
        list_lambda_cardinality=list_lambda_cardinality,
        list_lambda_direct_read=list_lambda_direct_read,
    )


@pytest.mark.asyncio
async def test_embedding_output_of_list_lambda_folds_to_one_value(
    _archive_env: tuple[_StubFm, _StubStorage],
) -> None:
    _, stub_storage = _archive_env
    items = [
        {
            "embedding_file": {"path": "embeddings.safetensors"},
            "model": _MODEL,
            "usage": {
                "prompt_tokens": 6,
                "num_requests": 3,
                "embedding_dim": 4,
                "latency_sec": 0.1,
            },
        }
    ]
    outputs = await _aggregate(items, list_lambda_direct_read=True)
    assert len(outputs) == 1
    rows = json.loads(outputs[0])
    assert [row["row"] for row in rows] == [0, 1, 2]
    for row in rows:
        assert row["model"] == _MODEL
        assert row["output"].endswith("embeddings.safetensors")
    assert len(stub_storage.saved) == 1


@pytest.mark.asyncio
async def test_image_output_of_list_lambda_folds_to_one_value(
    _archive_env: tuple[_StubFm, _StubStorage],
) -> None:
    _, stub_storage = _archive_env
    items = [{"image": {"path": "a.png"}}, {"image": {"path": "b.png"}}]
    outputs = await _aggregate(items, list_lambda_direct_read=True)
    assert len(outputs) == 1
    entries = json.loads(outputs[0])
    assert len(entries) == 2
    for entry, name in zip(entries, ("a.png", "b.png")):
        assert entry.startswith("blob://req-1/output-")
        assert entry.endswith(name)
    assert len(stub_storage.saved) == 2


@pytest.mark.asyncio
async def test_one_item_per_row_folds_to_one_value() -> None:
    items = [{"output": "a"}, {"output": "b"}]
    assert await _aggregate(items, list_lambda_direct_read=True) == [
        json.dumps(["a", "b"])
    ]
    assert await _aggregate(items, list_lambda_direct_read=False) == ["a", "b"]


@pytest.mark.asyncio
async def test_rowwise_output_folds_once() -> None:
    outputs = await FlowmeshRuntimeManager()._aggregate_output_node(
        output_op_id="output",
        output_task_id="task-1",
        request_id="req-1",
        items=[{"rows": ["a", "b"]}],
        output_path="items.rows",
        list_lambda=False,
        list_lambda_cardinality=True,
        list_lambda_direct_read=False,
        rowwise_llm=True,
    )
    assert outputs == [json.dumps(["a", "b"])]


def _explode_fn(items: tuple[list[str], ...]) -> list[dict[str, str]]:
    return [{"value": v} for v in items[0]]


def _probe_shapes() -> list[tuple[str, Any, str, list[dict[str, Any]], Any, list[str]]]:
    """The shapes a list Lambda reaches without binding a node's list column:
    an aggregate chat's answer read by a completion, a row-wise judge over a
    retrieval parameter, an S3 retrieval, and a row-wise chat read by another
    row-wise chat. Each carries the output path, FlowMesh-shaped items and the
    result the aggregator produces for it."""
    config = GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct")
    completion_config = GenerationConfig(model="Qwen/Qwen3-8B")
    shapes: list[tuple[str, Any, str, list[dict[str, Any]], Any, list[str]]] = []

    stock = input_placeholder("Stock")
    explode = LambdaOp([stock], _explode_fn, mode="list")
    summary = LLMChatOp(
        [OpMessage(role="user", content="Summarize {df}")],
        config=config,
        aggregate_table=[
            {"label": "value", "node": explode.id, "path": "items.output"}
        ],
    )
    summary.inputs.append(explode)
    title = LLMCompletionOp(prompt=summary, config=completion_config)
    graph = RuntimeGraphBuilder().build(
        Graph.from_ops([as_output("title", title)]).compile(Stock=["A", "B"])
    )
    shapes.append(
        (
            "aggregate chat -> completion",
            graph,
            "items.output",
            [{"output": "s"}],
            title,
            ["s"],
        )
    )

    stock = input_placeholder("Stock")
    explode = LambdaOp([stock], _explode_fn, mode="list")
    retrieval = DataRetrievalOp(
        data_spec={
            "type": "lumid",
            "mode": "sql",
            "template": "SELECT * FROM news WHERE symbol = :symbol",
            "params": [
                {"name": "symbol", "node": explode.id, "path": "items.output.value"}
            ],
        },
        inputs=[explode],
    )
    judge = LLMChatOp(
        [OpMessage(role="user", content="Judge {r}")],
        config=config,
        rowwise_template="Judge {r}",
        rowwise_columns=[{"label": "r", "node": retrieval.id, "path": "items.table"}],
    )
    judge.inputs.append(retrieval)
    graph = RuntimeGraphBuilder().build(
        Graph.from_ops([as_output("rated", judge)]).compile(Stock=["A", "B", "C"])
    )
    shapes.append(
        (
            "list Lambda -> retrieval param -> row-wise judge",
            graph,
            "items.rows",
            [{"rows": ["good", "bad"]}],
            judge,
            ['["good", "bad"]'],
        )
    )

    stock = input_placeholder("Stock")
    explode = LambdaOp([stock], _explode_fn, mode="list")
    s3 = DataRetrievalOp(
        data_spec={
            "type": "lumid",
            "mode": "s3",
            "template": "docs/{k}.txt",
            "params": [{"name": "k", "node": explode.id, "path": "items.output.value"}],
        },
        inputs=[explode],
    )
    graph = RuntimeGraphBuilder().build(
        Graph.from_ops([as_output("docs", s3)]).compile(Stock=["A", "B", "C"])
    )
    shapes.append(
        (
            "list Lambda -> S3 retrieval",
            graph,
            "items.content",
            [{"keys": ["docs/A.txt", "docs/B.txt"], "content": ["A", "B"]}],
            s3,
            ['["A", "B"]'],
        )
    )

    stock = input_placeholder("Stock")
    explode = LambdaOp([stock], _explode_fn, mode="list")
    rowwise = LLMChatOp(
        [OpMessage(role="user", content="Say {v}")],
        config=config,
        rowwise_template="Say {v}",
        rowwise_columns=[
            {"label": "v", "node": explode.id, "path": "items.output.value"}
        ],
    )
    rowwise.inputs.append(explode)
    rewrite = LLMChatOp(
        [OpMessage(role="user", content="Rewrite {r}")],
        config=config,
        rowwise_template="Rewrite {r}",
        rowwise_columns=[{"label": "r", "node": rowwise.id, "path": "items.output"}],
    )
    rewrite.inputs.append(rowwise)
    graph = RuntimeGraphBuilder().build(
        Graph.from_ops([as_output("rewritten", rewrite)]).compile(Stock=["A", "B", "C"])
    )
    shapes.append(
        (
            "list Lambda -> row-wise chat -> row-wise chat",
            graph,
            "items.rows",
            [{"rows": ["a0", "a1", "a2"]}],
            rewrite,
            ['["a0", "a1", "a2"]'],
        )
    )
    return shapes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape",
    [
        "aggregate chat -> completion",
        "list Lambda -> retrieval param -> row-wise judge",
        "list Lambda -> S3 retrieval",
        "list Lambda -> row-wise chat -> row-wise chat",
    ],
)
async def test_probe_shapes_aggregate_as_on_main(
    shape: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A list Lambda reached without a node's list column keeps the branch it
    takes for any other upstream: the fold comes only from the row-wise arm's
    cardinality set."""
    monkeypatch.setattr(fm_mod.envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(fm_mod.envs, "LUMID_DATA_TOKEN", "token")
    expected_shape = next(s for s in _probe_shapes() if s[0] == shape)
    _, graph, output_path, items, op, expected = expected_shape
    node_id = graph.dsl_to_runtime[op.id][-1]
    manager = FlowmeshRuntimeManager()
    flags = {
        "output_op_id": node_id,
        "output_task_id": "task-1",
        "request_id": "req-1",
        "items": items,
        "output_path": output_path,
        "list_lambda": False,
        "list_lambda_cardinality": node_id in graph.list_lambda_cardinality_nodes(),
        "list_lambda_direct_read": node_id in graph.list_lambda_direct_read_nodes(),
        "rowwise_llm": graph.nodes[node_id].data_spec.get("type") == "dataframe",
    }
    assert node_id not in graph.list_lambda_direct_read_nodes(), shape
    assert await manager._aggregate_output_node(**flags) == expected
