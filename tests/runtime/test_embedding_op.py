import json
import struct
import textwrap
from pathlib import Path
from typing import Any

import pytest
from lumilake import envs

import lumilake_server.runtime.runtime_manager.flowmesh as fm_mod
from lumilake_server.common import GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    EmbeddingArtifact,
    EmbeddingOp,
    as_output,
    input_placeholder,
)
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager

_MODEL = "BAAI/bge-small-en-v1.5"


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")


def _build_literal_graph(**config_kwargs):
    cfg = GenerationConfig(model=_MODEL, **config_kwargs)
    emb = EmbeddingOp(content=["hello world", "second doc"], config=cfg)
    output = as_output("vectors", emb)
    compiled = Graph.from_ops([output]).compile()
    return RuntimeGraphBuilder().build(compiled), emb.id


def test_embedding_op_emits_embedding_task_spec() -> None:
    runtime_graph, emb_id = _build_literal_graph()
    node = runtime_graph.nodes[emb_id]
    assert node.task_type == "embedding"
    assert node.backend == "vllm"
    assert node.model == _MODEL
    assert node.model_spec["source"]["identifier"] == _MODEL
    assert node.model_spec["vllm"]["convert"] == "embed"
    # The executor sets runner="pooling" internally; the op must not emit it.
    assert "runner" not in node.model_spec
    assert "runner" not in node.model_spec["vllm"]
    assert node.data_spec == {
        "type": "list",
        "items": ["hello world", "second doc"],
    }


def test_embedding_flowmesh_node_omits_inference_and_runner() -> None:
    runtime_graph, emb_id = _build_literal_graph()
    fm = runtime_graph.nodes[emb_id].to_flowmesh_node()
    assert fm["spec"]["taskType"] == "embedding"
    assert fm["spec"]["model"]["vllm"]["convert"] == "embed"
    assert fm["spec"]["data"] == {
        "type": "list",
        "items": ["hello world", "second doc"],
    }
    assert "inference" not in fm["spec"]
    assert "runner" not in fm["spec"]


def test_embedding_op_requests_embedding_artifact() -> None:
    runtime_graph, emb_id = _build_literal_graph()
    node = runtime_graph.nodes[emb_id]
    # Vectors come back as a safetensors artifact, not inline; the op must
    # not project an inline ``items.embedding`` payload.
    assert emb_id not in runtime_graph.output_paths
    assert node.output_spec is not None
    assert "embeddings.safetensors" in node.output_spec["artifacts"]


def test_embedding_op_passes_vllm_engine_kwargs() -> None:
    runtime_graph, emb_id = _build_literal_graph(
        gpu_memory_utilization=0.5, tensor_parallel_size=2
    )
    vllm = runtime_graph.nodes[emb_id].model_spec["vllm"]
    assert vllm["convert"] == "embed"
    assert vllm["gpu_memory_utilization"] == 0.5
    assert vllm["tensor_parallel_size"] == 2


def test_embedding_op_from_input_placeholder() -> None:
    cfg = GenerationConfig(model=_MODEL)
    emb = EmbeddingOp(content=input_placeholder("Docs"), config=cfg)
    output = as_output("vectors", emb)
    compiled = Graph.from_ops([output]).compile(Docs=["a", "b", "c"])
    node = RuntimeGraphBuilder().build(compiled).nodes[emb.id]
    assert node.task_type == "embedding"
    assert node.data_spec == {"type": "list", "items": ["a", "b", "c"]}


def test_embedding_op_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        EmbeddingOp(content=[], config=GenerationConfig(model=_MODEL))
    with pytest.raises(ValueError):
        EmbeddingOp(content="", config=GenerationConfig(model=_MODEL))
    with pytest.raises(ValueError):
        EmbeddingOp(content=["ok", "  "], config=GenerationConfig(model=_MODEL))


@pytest.mark.parametrize("docs", [[""], ["ok", ""], ["  "]])
def test_embedding_op_rejects_blank_input_values_at_build(docs: list[str]) -> None:
    cfg = GenerationConfig(model=_MODEL)
    emb = EmbeddingOp(content=input_placeholder("Docs"), config=cfg)
    output = as_output("vectors", emb)
    compiled = Graph.from_ops([output]).compile(Docs=docs)
    with pytest.raises(ValueError, match="must be a non-empty string"):
        RuntimeGraphBuilder().build(compiled)


def test_embedding_op_builds_from_valid_input_values() -> None:
    cfg = GenerationConfig(model=_MODEL)
    emb = EmbeddingOp(content=input_placeholder("Docs"), config=cfg)
    output = as_output("vectors", emb)
    compiled = Graph.from_ops([output]).compile(Docs=["ok", "second"])
    node = RuntimeGraphBuilder().build(compiled).nodes[emb.id]
    assert node.task_type == "embedding"
    assert node.model_spec["vllm"]["convert"] == "embed"
    assert node.data_spec == {"type": "list", "items": ["ok", "second"]}


def _worker_response() -> dict:
    # Merged FlowMesh contract: the result carries `embedding_file` + a
    # `usage` block (num_requests/embedding_dim replace the old top-level
    # count/dim), and the framework stamps `_artifacts`. No prompts_file.
    return {
        "ok": True,
        "model": _MODEL,
        "embedding_file": {"path": "embeddings.safetensors"},
        "usage": {
            "prompt_tokens": 12,
            "total_tokens": 12,
            "num_requests": 2,
            "embedding_dim": 384,
            "latency_sec": 0.5,
        },
        "_artifacts": {
            "embeddings.safetensors": {"uri": "blob://job/embeddings.safetensors"},
        },
    }


def test_parse_response_surfaces_artifact_ref_and_metadata() -> None:
    artifact = EmbeddingOp.parse_response(_worker_response())
    assert isinstance(artifact, EmbeddingArtifact)
    assert artifact.model == _MODEL
    assert artifact.embedding_file == "embeddings.safetensors"
    # Row/dimension counts are carried on `usage`, not as artifact fields.
    assert artifact.usage["num_requests"] == 2
    assert artifact.usage["embedding_dim"] == 384
    assert artifact.usage["prompt_tokens"] == 12
    assert "embeddings.safetensors" in artifact.artifacts


def test_parse_response_carries_no_redundant_count_dim() -> None:
    # count/dim were dropped from the artifact (they duplicate `usage`); the
    # dropped prompts sidecar must not reappear either.
    artifact = EmbeddingOp.parse_response(_worker_response())
    assert not hasattr(artifact, "count")
    assert not hasattr(artifact, "dim")
    assert not hasattr(artifact, "prompts_file")


def test_parse_response_fails_when_embedding_file_missing() -> None:
    result = _worker_response()
    del result["embedding_file"]
    with pytest.raises(ValueError, match="embedding_file"):
        EmbeddingOp.parse_response(result)


def test_parse_response_fails_when_artifacts_context_missing() -> None:
    result = _worker_response()
    del result["_artifacts"]
    with pytest.raises(ValueError, match="_artifacts"):
        EmbeddingOp.parse_response(result)


def test_parse_response_fails_when_usage_missing() -> None:
    result = _worker_response()
    del result["usage"]
    with pytest.raises(ValueError, match="usage"):
        EmbeddingOp.parse_response(result)


def test_embedding_op_yaml_end_to_end() -> None:
    yaml_text = textwrap.dedent(
        """
        name: embed_docs
        inputs:
          Docs: ["hello", "world"]
        ops:
          - id: Embed
            op: EmbeddingOp
            inputs: [Docs]
            input: Docs
            config:
              model: BAAI/bge-small-en-v1.5
        outputs:
          - name: vectors
            ref: Embed
        """
    )
    spec = parse_yaml_payload(yaml_text)["embed_docs"]
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)
    embedding_nodes = [
        node for node in runtime_graph.nodes.values() if node.task_type == "embedding"
    ]
    assert len(embedding_nodes) == 1
    node = embedding_nodes[0]
    assert node.model_spec["source"]["identifier"] == "BAAI/bge-small-en-v1.5"
    assert node.model_spec["vllm"]["convert"] == "embed"
    assert node.data_spec == {"type": "list", "items": ["hello", "world"]}


def test_embedding_op_yaml_requires_model() -> None:
    yaml_text = textwrap.dedent(
        """
        name: embed_docs
        inputs:
          Docs: ["hello"]
        ops:
          - id: Embed
            op: EmbeddingOp
            inputs: [Docs]
            input: Docs
            config: {}
        outputs:
          - name: vectors
            ref: Embed
        """
    )
    with pytest.raises(ValueError, match="requires 'config' with a 'model'"):
        parse_yaml_payload(yaml_text)


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


def _embedding_items() -> list[dict]:
    # Merged FlowMesh result shape: count/dim live inside `usage`, no
    # prompts_file, no top-level count/dim.
    return [
        {
            "embedding_file": {"path": "embeddings.safetensors"},
            "model": _MODEL,
            "usage": {
                "prompt_tokens": 12,
                "num_requests": 2,
                "embedding_dim": 384,
                "latency_sec": 0.5,
            },
        }
    ]


@pytest.fixture
def _archive_env(monkeypatch: pytest.MonkeyPatch) -> tuple[_StubFm, _StubStorage]:
    stub_fm = _StubFm(b"safetensors-bytes")
    stub_storage = _StubStorage()
    monkeypatch.setattr(fm_mod, "flowmesh_for_context", lambda: stub_fm)
    monkeypatch.setattr(fm_mod, "get_job_storage", lambda: stub_storage)
    monkeypatch.setattr(fm_mod.envs, "S3_ARCHIVE_PREFIX", "s3://bucket/prefix")
    return stub_fm, stub_storage


@pytest.mark.asyncio
async def test_aggregation_surfaces_artifact_uri_and_metadata(
    _archive_env: tuple[_StubFm, _StubStorage],
) -> None:
    stub_fm, stub_storage = _archive_env
    manager = FlowmeshRuntimeManager()
    outputs = await manager._aggregate_output_node(
        output_op_id="Embed",
        output_task_id="task-1",
        request_id="req-1",
        items=_embedding_items(),
        output_path=None,
    )
    # usage.num_requests=2 covers 2 input rows via one artifact: the
    # per-row output contract requires 2 output items, not 1.
    assert len(outputs) == 2
    rows = [json.loads(o) for o in outputs]
    for idx, obj in enumerate(rows):
        # The archived entry surfaces the artifact uri + model ident +
        # row index. Redundant count/dim are no longer copied onto the
        # archive surface.
        assert obj["model"] == _MODEL
        assert obj["row"] == idx
        assert "count" not in obj
        assert "dim" not in obj
        assert obj["output"].endswith("embeddings.safetensors")
    # The shared safetensors artifact is downloaded/archived only once,
    # even though it is referenced by every row.
    assert len(stub_storage.saved) == 1
    # FlowMesh's `/files/{filename}` route resolves relative to the task's
    # `artifacts/` dir itself; the SDK call must pass the bare ref path.
    assert stub_fm.results.downloaded == [("task-1", "embeddings.safetensors")]


@pytest.mark.asyncio
async def test_aggregation_fails_fast_without_archive_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fm_mod, "flowmesh_for_context", lambda: _StubFm(b"x"))
    monkeypatch.setattr(fm_mod, "get_job_storage", lambda: _StubStorage())
    monkeypatch.setattr(fm_mod.envs, "S3_ARCHIVE_PREFIX", "")
    manager = FlowmeshRuntimeManager()
    with pytest.raises(
        RuntimeError, match="S3_ARCHIVE_PREFIX is required for embedding"
    ):
        await manager._aggregate_output_node(
            output_op_id="Embed",
            output_task_id="task-1",
            request_id="req-1",
            items=_embedding_items(),
            output_path=None,
        )


@pytest.mark.asyncio
async def test_aggregation_fails_fast_on_missing_embedding_path(
    _archive_env: tuple[_StubFm, _StubStorage],
) -> None:
    manager = FlowmeshRuntimeManager()
    bad_items = [
        {
            "embedding_file": {},
            "model": _MODEL,
            "usage": {"num_requests": 1, "embedding_dim": 4},
        }
    ]
    with pytest.raises(RuntimeError, match=r"embedding_file\.path"):
        await manager._aggregate_output_node(
            output_op_id="Embed",
            output_task_id="task-1",
            request_id="req-1",
            items=bad_items,
            output_path=None,
        )


def test_resolve_output_items_accepts_flat_embedding_result() -> None:
    # Live FlowMesh embedding tasks return a flat result with no `items`
    # list: {"ok": true, "embedding_file": {...}, "usage": {...}}.
    manager = FlowmeshRuntimeManager()
    results_json = {
        "ok": True,
        "embedding_file": {"path": "embeddings.safetensors"},
        "usage": {"num_requests": 3, "embedding_dim": 384},
    }
    items = manager._resolve_output_items(results_json, "Embed")
    assert items == [results_json]


def test_resolve_output_items_prefers_items_list_when_present() -> None:
    manager = FlowmeshRuntimeManager()
    results_json = {"items": _embedding_items()}
    assert manager._resolve_output_items(results_json, "Embed") == _embedding_items()


def test_resolve_output_items_fails_fast_without_items_or_embedding_file() -> None:
    manager = FlowmeshRuntimeManager()
    with pytest.raises(RuntimeError, match="produced no items"):
        manager._resolve_output_items({"ok": True}, "Embed")


@pytest.mark.asyncio
async def test_flat_embedding_result_aggregates_to_artifact_ref(
    _archive_env: tuple[_StubFm, _StubStorage],
) -> None:
    """End-to-end regression: a flat embedding result covering a slice of 3
    input rows (no `items`, `usage.num_requests=3`) must resolve to 3
    per-row `vectors` output items, not 1 — the bug that produced
    "Output length mismatch ... expected=3 got=1" downstream."""
    manager = FlowmeshRuntimeManager()
    results_json = {
        "ok": True,
        "embedding_file": {"path": "embeddings.safetensors"},
        "usage": {"num_requests": 3, "embedding_dim": 384},
    }
    items = manager._resolve_output_items(results_json, "Embed")
    outputs = await manager._aggregate_output_node(
        output_op_id="Embed",
        output_task_id="task-1",
        request_id="req-1",
        items=items,
        output_path=None,
        expected_row_count=3,
    )
    assert len(outputs) == 3
    for idx, raw in enumerate(outputs):
        obj = json.loads(raw)
        assert obj["row"] == idx
        assert obj["output"].endswith("embeddings.safetensors")


@pytest.mark.asyncio
async def test_embedding_row_count_mismatch_fails_fast(
    _archive_env: tuple[_StubFm, _StubStorage],
) -> None:
    """If the artifact's vector count doesn't match the expected slice
    length, fail fast instead of silently padding/truncating the output."""
    manager = FlowmeshRuntimeManager()
    results_json = {
        "ok": True,
        "embedding_file": {"path": "embeddings.safetensors"},
        "usage": {"num_requests": 3, "embedding_dim": 384},
    }
    items = manager._resolve_output_items(results_json, "Embed")
    with pytest.raises(RuntimeError, match="row count mismatch"):
        await manager._aggregate_output_node(
            output_op_id="Embed",
            output_task_id="task-1",
            request_id="req-1",
            items=items,
            output_path=None,
            expected_row_count=5,
        )


def test_embedding_row_count_from_literal_data_spec() -> None:
    runtime_graph, emb_id = _build_literal_graph()
    manager = FlowmeshRuntimeManager()
    assert manager._embedding_row_count(runtime_graph, emb_id) == 2


def test_embedding_row_count_none_for_non_embedding_node() -> None:
    runtime_graph, emb_id = _build_literal_graph()
    manager = FlowmeshRuntimeManager()
    assert manager._embedding_row_count(runtime_graph, "not-a-node") is None


def _pack_safetensors(name: str, shape: tuple[int, int], values: list[float]) -> bytes:
    # Minimal safetensors writer (stdlib only, no `safetensors`/`torch`
    # dependency): 8-byte LE header length, JSON header, then raw F32 data.
    body = struct.pack(f"<{len(values)}f", *values)
    header = {
        name: {"dtype": "F32", "shape": list(shape), "data_offsets": [0, len(body)]}
    }
    header_bytes = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(header_bytes)) + header_bytes + body


def _unpack_safetensors_tensor(
    data: bytes, name: str
) -> tuple[tuple[int, ...], list[float]]:
    (header_len,) = struct.unpack_from("<Q", data, 0)
    header: dict[str, Any] = json.loads(data[8 : 8 + header_len])
    meta = header[name]
    start, end = meta["data_offsets"]
    body_start = 8 + header_len
    values = list(
        struct.unpack(
            f"<{(end - start) // 4}f", data[body_start + start : body_start + end]
        )
    )
    return tuple(meta["shape"]), values


@pytest.mark.asyncio
async def test_embedding_artifact_is_fetchable_and_loads_real_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The archived artifact isn't just a path: downloading and parsing it
    yields the real per-row float vectors at the documented [count, dim]
    shape (docs/OPS.md), proving a consumer can actually load `vectors`."""
    rows = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    payload = _pack_safetensors("embeddings", (2, 4), [v for row in rows for v in row])
    stub_fm = _StubFm(payload)
    stub_storage = _StubStorage()
    monkeypatch.setattr(fm_mod, "flowmesh_for_context", lambda: stub_fm)
    monkeypatch.setattr(fm_mod, "get_job_storage", lambda: stub_storage)
    monkeypatch.setattr(fm_mod.envs, "S3_ARCHIVE_PREFIX", "s3://bucket/prefix")

    manager = FlowmeshRuntimeManager()
    await manager._aggregate_output_node(
        output_op_id="Embed",
        output_task_id="task-1",
        request_id="req-1",
        items=_embedding_items(),
        output_path=None,
    )

    _, _, archived_bytes, _ = stub_storage.saved[0]
    shape, values = _unpack_safetensors_tensor(archived_bytes, "embeddings")
    assert shape == (2, 4)
    assert values == pytest.approx([v for row in rows for v in row], abs=1e-6)
