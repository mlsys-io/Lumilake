import contextlib
import json
import struct
from pathlib import Path

import httpx2
import pytest
from huggingface_hub.utils import NotASafetensorsRepoError

from lumilake_server.data_profile_models import (
    DataProfileCostEstimate,
    DataProfileResultRow,
)
from lumilake_server.graphs import Graph
from lumilake_server.ops import LambdaOp, as_output, input_placeholder
from lumilake_server.runtime.optimizer.halo import HaloOptimizer
from lumilake_server.runtime.optimizer.schedule.models import Node
from lumilake_server.runtime.runtime_graph import RuntimeGraph, RuntimeGraphBuilder
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _simple_graph() -> RuntimeGraph:
    nodes = {
        "n1": RuntimeOp(
            node_id="n1",
            task_type="inference",
            backend="vllm",
            model="meta-llama/Llama-3.1-8B-Instruct",
            data_spec={"type": "list", "items": ["a"]},
            model_spec={},
            inference_spec={"max_tokens": 16},
        ),
        "n2": RuntimeOp(
            node_id="n2",
            task_type="inference",
            backend="vllm",
            model="meta-llama/Llama-3.1-8B-Instruct",
            data_spec={
                "type": "graph_template",
                "template": {
                    "name": "format",
                    "columns": [{"label": "x", "node": "n1", "path": "items.output"}],
                    "options": {"format": {"template": "{x}"}},
                },
            },
            model_spec={},
            inference_spec={"max_tokens": 16},
            dependencies=("n1",),
        ),
    }
    return RuntimeGraph(
        nodes=nodes,
        node_order=["n1", "n2"],
        output_node_map={},
        dsl_to_runtime={},
    )


def test_halo_optimizer_generates_schedule_covering_all_nodes() -> None:
    optimizer = HaloOptimizer()
    graph = _simple_graph()

    schedule = optimizer.generate_schedule(
        graph=graph,
        worker_names=["gpu-0", "cpu-0"],
        worker_profiles={"gpu-0": {"has_gpu": True}, "cpu-0": {"has_gpu": False}},
    )

    scheduled = {
        node_id
        for worker_nodes in schedule.worker_assignment.values()
        for node_id in worker_nodes
    }
    assert scheduled == {"n1", "n2"}


def test_halo_optimizer_requires_workers() -> None:
    optimizer = HaloOptimizer()
    with pytest.raises(ValueError, match="requires at least one worker"):
        optimizer.generate_schedule(
            graph=_simple_graph(),
            worker_names=[],
            worker_profiles={},
        )


def _retrieval_graph(*, task_type: str, backend: str) -> RuntimeGraph:
    return RuntimeGraph(
        nodes={
            "r1": RuntimeOp(
                node_id="r1",
                task_type=task_type,
                backend=backend,
                model="data_retrieval",
                data_spec={"type": "sql", "template": "SELECT 1"},
                model_spec={},
                inference_spec={},
            )
        },
        node_order=["r1"],
        output_node_map={},
        dsl_to_runtime={},
    )


def test_halo_optimizer_allows_data_retrieval_cpu_nodes() -> None:
    optimizer = HaloOptimizer()
    graph = _retrieval_graph(task_type="data_retrieval", backend="data_retrieval")
    optimizer._validate_supported_runtime_nodes(graph)


def test_halo_optimizer_rejects_non_retrieval_cpu_nodes() -> None:
    optimizer = HaloOptimizer()
    graph = _retrieval_graph(task_type="query", backend="data_retrieval")
    with pytest.raises(ValueError, match="only supports data_retrieval nodes on CPU"):
        optimizer._validate_supported_runtime_nodes(graph)


def test_halo_optimizer_rejects_data_profiling_nodes() -> None:
    optimizer = HaloOptimizer()
    graph = _retrieval_graph(task_type="data_profiling", backend="data_profiling")
    with pytest.raises(ValueError, match="unsupported runtime node"):
        optimizer._validate_supported_runtime_nodes(graph)


def _api_graph() -> RuntimeGraph:
    return RuntimeGraph(
        nodes={
            "a1": RuntimeOp(
                node_id="a1",
                task_type="api",
                backend="api",
                model="meta-llama/Llama-3.1-8B-Instruct",
                data_spec={"type": "graph_template", "template": {}},
                model_spec={},
                inference_spec={},
            )
        },
        node_order=["a1"],
        output_node_map={},
        dsl_to_runtime={},
    )


def test_halo_optimizer_places_api_node_on_cpu_worker() -> None:
    optimizer = HaloOptimizer()
    graph = _api_graph()
    schedule = optimizer.generate_schedule(
        graph=graph,
        worker_names=["gpu-0", "cpu-0"],
        worker_profiles={"gpu-0": {"has_gpu": True}, "cpu-0": {"has_gpu": False}},
    )
    scheduled = {
        node_id
        for worker_nodes in schedule.worker_assignment.values()
        for node_id in worker_nodes
    }
    assert scheduled == {"a1"}
    assert "a1" in schedule.worker_assignment["cpu-0"]


def _python_graph() -> RuntimeGraph:
    return RuntimeGraph(
        nodes={
            "p1": RuntimeOp(
                node_id="p1",
                task_type="python",
                backend="python",
                model="",
                data_spec={"code": "lambda a: a[0]", "plan": [], "timeout_s": 600},
                model_spec={},
                inference_spec={},
            )
        },
        node_order=["p1"],
        output_node_map={},
        dsl_to_runtime={},
    )


def test_halo_optimizer_places_python_node_on_a_python_worker() -> None:
    schedule = HaloOptimizer().generate_schedule(
        graph=_python_graph(),
        worker_names=["cpu-plain", "cpu-python"],
        worker_profiles={
            "cpu-plain": {"has_gpu": False, "supported_task_types": ["echo"]},
            "cpu-python": {
                "has_gpu": False,
                "supported_task_types": ["echo", "python"],
            },
        },
    )
    assert schedule.worker_assignment["cpu-python"] == ["p1"]
    assert schedule.worker_assignment["cpu-plain"] == []


def test_halo_optimizer_prefers_a_cpu_python_worker_over_a_gpu_one() -> None:
    schedule = HaloOptimizer().generate_schedule(
        graph=_python_graph(),
        worker_names=["gpu-python", "cpu-python"],
        worker_profiles={
            "gpu-python": {"has_gpu": True, "supported_task_types": ["python"]},
            "cpu-python": {"has_gpu": False, "supported_task_types": ["python"]},
        },
    )
    assert schedule.worker_assignment["cpu-python"] == ["p1"]


def test_halo_optimizer_never_places_python_on_a_gpu_worker() -> None:
    with pytest.raises(ValueError, match="selected CPU workers"):
        HaloOptimizer().generate_schedule(
            graph=_python_graph(),
            worker_names=["gpu-python", "cpu-plain"],
            worker_profiles={
                "gpu-python": {"has_gpu": True, "supported_task_types": ["python"]},
                "cpu-plain": {"has_gpu": False, "supported_task_types": []},
            },
        )


@pytest.mark.parametrize(
    "profile",
    [
        {"has_gpu": False, "supported_task_types": ["echo", "ssh"]},
        {"has_gpu": False},
    ],
)
def test_halo_optimizer_fails_before_scheduling_without_a_python_worker(
    profile: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="advertises the 'python' task type"):
        HaloOptimizer().generate_schedule(
            graph=_python_graph(),
            worker_names=["cpu-plain"],
            worker_profiles={"cpu-plain": profile},
        )


def _python_node(raw: dict[str, object]) -> Node:
    return Node(
        id="p1",
        type="python",
        engine="python",
        model="",
        inputs=(),
        outputs=(),
        db_queries=(),
        raw=raw,
    )


def test_python_exec_cost_uses_expected_runtime_when_present() -> None:
    optimizer = HaloOptimizer()
    assert (
        optimizer._python_exec_cost(
            _python_node({"timeout_s": 600, "expected_runtime_s": 5})
        )
        == 5
    )
    optimizer._input_query_count = 10
    assert (
        optimizer._python_exec_cost(
            _python_node({"timeout_s": 600, "expected_runtime_s": 5})
        )
        == 50
    )


def test_python_exec_cost_scales_with_literal_plan_rows() -> None:
    optimizer = HaloOptimizer()
    node = _python_node(
        {
            "timeout_s": 600,
            "expected_runtime_s": 2,
            "plan": [
                {"kind": "literal", "values": list(range(10))},
            ],
        }
    )
    assert optimizer._python_exec_cost(node) == 20


def test_python_exec_cost_counts_stage_rows_alongside_a_broadcast_literal() -> None:
    optimizer = HaloOptimizer()
    optimizer._input_query_count = 10
    node = _python_node(
        {
            "timeout_s": 600,
            "expected_runtime_s": 2,
            "plan": [
                {"kind": "stage", "node": "upstream"},
                {"kind": "literal", "values": ["prefix"]},
            ],
        }
    )
    assert optimizer._python_exec_cost(node) == 20


def test_python_exec_cost_prefers_the_recorded_row_count() -> None:
    optimizer = HaloOptimizer()
    optimizer._input_query_count = 10
    node = _python_node(
        {
            "timeout_s": 600,
            "expected_runtime_s": 2,
            "row_count": 7,
            "plan": [{"kind": "stage", "node": "upstream"}],
        }
    )
    assert optimizer._python_exec_cost(node) == 14


def test_python_exec_cost_falls_back_to_timeout_fraction() -> None:
    optimizer = HaloOptimizer()
    assert optimizer._python_exec_cost(_python_node({"timeout_s": 600})) == 30
    assert optimizer._python_exec_cost(_python_node({})) == 30
    optimizer._input_query_count = 10
    assert optimizer._python_exec_cost(_python_node({"timeout_s": 600})) == 30


def test_halo_dp_places_generated_python_steps_on_the_advertising_worker() -> None:
    """Runtime graphs the builder generates: several standalone python steps and
    plain CPU workers ahead of the one that advertises python, so the DP's CPU
    assignment would spread them over the plain workers if it ignored each
    node's eligible workers."""
    stock = input_placeholder("Stock")
    steps = [
        as_output(f"out{i}", LambdaOp([stock], fn=lambda inputs: str(inputs[0])))
        for i in range(4)
    ]
    compiled = Graph.from_ops(steps).compile(Stock=["NVDA"])
    graph = RuntimeGraphBuilder().build(compiled)
    python_nodes = {n for n, op in graph.nodes.items() if op.task_type == "python"}
    assert len(python_nodes) == 4

    schedule = HaloOptimizer().generate_schedule(
        graph=graph,
        worker_names=["cpu-a", "cpu-b", "cpu-python"],
        worker_profiles={
            "cpu-a": {"has_gpu": False, "supported_task_types": []},
            "cpu-b": {"has_gpu": False, "supported_task_types": ["echo"]},
            "cpu-python": {"has_gpu": False, "supported_task_types": ["python"]},
        },
    )

    assert schedule.worker_assignment["cpu-a"] == []
    assert schedule.worker_assignment["cpu-b"] == []
    assert set(schedule.worker_assignment["cpu-python"]) == python_nodes


def test_halo_optimizer_does_not_filter_other_task_types_by_capability() -> None:
    schedule = HaloOptimizer().generate_schedule(
        graph=_api_graph(),
        worker_names=["cpu-plain"],
        worker_profiles={"cpu-plain": {"has_gpu": False}},
    )
    assert schedule.worker_assignment["cpu-plain"] == ["a1"]


def test_model_size_resolution_parses_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = HaloOptimizer()
    metadata_calls: list[str] = []

    def resolve(model: str) -> int:
        metadata_calls.append(model)
        return 1

    monkeypatch.setattr(
        optimizer,
        "_safetensors_parameter_count_for_model",
        resolve,
    )
    size = optimizer._model_size_b(
        Node(id="n1", type="inference", engine="vllm", model="foo-1.5B", raw={})
    )
    assert size == pytest.approx(1.5)
    assert metadata_calls == []


def test_model_size_resolution_uses_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = HaloOptimizer()
    metadata_calls: list[str] = []

    def resolve(model: str) -> int:
        metadata_calls.append(model)
        return 1

    monkeypatch.setattr(
        optimizer,
        "_safetensors_parameter_count_for_model",
        resolve,
    )
    size = optimizer._model_size_b(
        Node(
            id="n1",
            type="diffusion",
            engine="vllm",
            model="stabilityai/stable-diffusion-xl-base-1.0",
            raw={},
        )
    )
    assert size == pytest.approx(3.5)
    assert metadata_calls == []


def test_model_size_resolution_covers_common_cross_encoders() -> None:
    optimizer = HaloOptimizer()
    for model, expected in (
        ("BAAI/bge-reranker-v2-m3", 0.568),
        ("cross-encoder/ms-marco-MiniLM-L6-v2", 0.0227),
    ):
        size = optimizer._model_size_b(
            Node(id="n1", type="classification", engine="vllm", model=model, raw={})
        )
        assert size == pytest.approx(expected)


def test_model_size_resolution_uses_safetensors_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = HaloOptimizer()
    monkeypatch.setattr(
        optimizer,
        "_safetensors_parameter_count_for_model",
        lambda model: 1_250_000_000,
    )
    size = optimizer._model_size_b(
        Node(
            id="n1",
            type="inference",
            engine="vllm",
            model="org/model-without-a-size-suffix",
            raw={},
        )
    )
    assert size == pytest.approx(1.25)


def test_model_size_resolution_caches_safetensors_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = HaloOptimizer()
    calls: list[str] = []

    def resolve(model: str) -> int:
        calls.append(model)
        return 750_000_000

    monkeypatch.setattr(
        optimizer,
        "_safetensors_parameter_count_for_model",
        resolve,
    )
    node = Node(
        id="n1",
        type="inference",
        engine="vllm",
        model="org/cache-test-model",
        raw={},
    )
    assert optimizer._model_size_b(node) == pytest.approx(0.75)
    assert optimizer._model_size_b(node) == pytest.approx(0.75)
    assert calls == ["org/cache-test-model"]


def test_safetensors_resolution_uses_default_hf_cache_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = HaloOptimizer()
    cache_calls: list[tuple[str, str]] = []

    def cached(model: str, filename: str) -> str:
        cache_calls.append((model, filename))
        return "/cache/model.safetensors"

    monkeypatch.setattr(
        "lumilake_server.runtime.optimizer.halo.try_to_load_from_cache",
        cached,
    )
    monkeypatch.setattr(
        optimizer,
        "_safetensors_parameter_count",
        lambda path: 570_000_000,
    )
    monkeypatch.setattr(
        optimizer,
        "_remote_parameter_count",
        lambda model: pytest.fail("remote metadata must not be consulted"),
    )
    count = optimizer._safetensors_parameter_count_for_model("org/cached-model")
    assert count == 570_000_000
    assert cache_calls == [("org/cached-model", "model.safetensors")]


def test_safetensors_parameter_count_reads_real_header(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.safetensors"
    header = {
        "__metadata__": {"format": "pt"},
        "scalar": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]},
        "matrix": {"dtype": "F32", "shape": [2, 3], "data_offsets": [4, 28]},
    }
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\0" * 28)

    assert HaloOptimizer._safetensors_parameter_count(path) == 7


def test_safetensors_parameter_count_rejects_truncated_header(
    tmp_path: Path,
) -> None:
    path = tmp_path / "truncated.safetensors"
    path.write_bytes(struct.pack("<Q", 100) + b"{}")

    with pytest.raises(ValueError, match="truncated safetensors header"):
        HaloOptimizer._safetensors_parameter_count(path)


def test_safetensors_parameter_count_rejects_invalid_shape(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid-shape.safetensors"
    header = {"tensor": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 0]}}
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes)

    with pytest.raises(ValueError, match="invalid shape"):
        HaloOptimizer._safetensors_parameter_count(path)


def test_cached_safetensors_sums_parameter_counts_across_shards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    optimizer = HaloOptimizer()
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {"weight_map": {"a": "shard-a.safetensors", "b": "shard-b.safetensors"}}
        ),
        encoding="utf-8",
    )
    shard_a = tmp_path / "shard-a.safetensors"
    shard_b = tmp_path / "shard-b.safetensors"
    for path, shape, data_size in ((shard_a, [2, 3], 24), (shard_b, [4], 16)):
        header = {
            "tensor": {"dtype": "F32", "shape": shape, "data_offsets": [0, data_size]}
        }
        header_bytes = json.dumps(header).encode("utf-8")
        path.write_bytes(
            struct.pack("<Q", len(header_bytes)) + header_bytes + b"\0" * data_size
        )

    files = {
        "model.safetensors": None,
        "model.safetensors.index.json": str(index),
        "shard-a.safetensors": str(shard_a),
        "shard-b.safetensors": str(shard_b),
    }
    monkeypatch.setattr(
        "lumilake_server.runtime.optimizer.halo.try_to_load_from_cache",
        lambda model, filename: files[filename],
    )
    assert optimizer._cached_parameter_count("org/sharded-model") == 10


def test_cached_safetensors_rejects_missing_shard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    optimizer = HaloOptimizer()
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {"weight_map": {"a": "shard-a.safetensors", "b": "shard-b.safetensors"}}
        ),
        encoding="utf-8",
    )
    shard_a = tmp_path / "shard-a.safetensors"
    header_bytes = json.dumps(
        {"tensor": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    ).encode("utf-8")
    shard_a.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\0" * 8)

    files = {
        "model.safetensors": None,
        "model.safetensors.index.json": str(index),
        "shard-a.safetensors": str(shard_a),
        "shard-b.safetensors": None,
    }
    monkeypatch.setattr(
        "lumilake_server.runtime.optimizer.halo.try_to_load_from_cache",
        lambda model, filename: files[filename],
    )

    with pytest.raises(ValueError, match="shard-b.safetensors"):
        optimizer._cached_parameter_count("org/sharded-model")


def test_safetensors_resolution_falls_back_to_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = HaloOptimizer()

    class Metadata:
        parameter_count = {"F32": 123_456_789}

    monkeypatch.setattr(
        "lumilake_server.runtime.optimizer.halo.try_to_load_from_cache",
        lambda model, filename: None,
    )
    monkeypatch.setattr(
        "lumilake_server.runtime.optimizer.halo.get_safetensors_metadata",
        lambda model: Metadata(),
    )

    assert (
        optimizer._safetensors_parameter_count_for_model("org/hub-model") == 123_456_789
    )


def test_zero_count_cache_falls_through_to_hub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    optimizer = HaloOptimizer()
    empty = tmp_path / "empty.safetensors"
    header_bytes = json.dumps({"__metadata__": {"format": "pt"}}).encode("utf-8")
    empty.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes)

    class Metadata:
        parameter_count = {"F32": 20_000_000}

    monkeypatch.setattr(
        "lumilake_server.runtime.optimizer.halo.try_to_load_from_cache",
        lambda model, filename: str(empty),
    )
    monkeypatch.setattr(
        "lumilake_server.runtime.optimizer.halo.get_safetensors_metadata",
        lambda model: Metadata(),
    )

    assert (
        optimizer._safetensors_parameter_count_for_model("org/empty-model")
        == 20_000_000
    )


def test_safetensors_resolution_reports_both_failed_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = HaloOptimizer()
    monkeypatch.setattr(
        "lumilake_server.runtime.optimizer.halo.try_to_load_from_cache",
        lambda model, filename: None,
    )
    monkeypatch.setattr(
        "lumilake_server.runtime.optimizer.halo.get_safetensors_metadata",
        lambda model: (_ for _ in ()).throw(
            NotASafetensorsRepoError("not a safetensors repository")
        ),
    )

    with pytest.raises(ValueError) as exc_info:
        optimizer._safetensors_parameter_count_for_model("org/missing-model")
    message = str(exc_info.value)
    assert "org/missing-model" in message
    assert "local HF cache" in message
    assert "Hugging Face safetensors metadata" in message


def test_safetensors_resolution_translates_hub_transport_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host with neither a cache nor network reaches the Hub and gets a
    transport error, which is not an ``OSError``; it must still fold into the
    model-size error that names every failed source instead of escaping."""
    optimizer = HaloOptimizer()
    monkeypatch.setattr(
        "lumilake_server.runtime.optimizer.halo.try_to_load_from_cache",
        lambda model, filename: None,
    )
    monkeypatch.setattr(
        "lumilake_server.runtime.optimizer.halo.get_safetensors_metadata",
        lambda model: (_ for _ in ()).throw(httpx2.ConnectError("connection refused")),
    )

    with pytest.raises(ValueError) as exc_info:
        optimizer._safetensors_parameter_count_for_model("org/unreachable-model")
    message = str(exc_info.value)
    assert "org/unreachable-model" in message
    assert "local HF cache" in message
    assert "Hugging Face safetensors metadata: connection refused" in message


def test_model_size_resolution_rejects_unknown_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = HaloOptimizer()

    def fail(_model: str) -> int:
        raise ValueError(
            "safetensors metadata failed: "
            "local HF cache: weights missing; "
            "Hugging Face safetensors metadata: HTTP 404"
        )

    monkeypatch.setattr(
        optimizer,
        "_safetensors_parameter_count_for_model",
        fail,
    )
    with pytest.raises(ValueError) as exc_info:
        optimizer._model_size_b(
            Node(
                id="n1",
                type="inference",
                engine="vllm",
                model="unknown/model-without-size",
                raw={},
            )
        )
    message = str(exc_info.value)
    assert "MODEL_SIZE catalog" in message
    assert "name suffix" in message
    assert "safetensors metadata" in message
    assert "local HF cache" in message
    assert "Hugging Face safetensors metadata" in message
    assert "unknown/model-without-size" in message


def test_model_size_resolution_rejects_missing_model() -> None:
    optimizer = HaloOptimizer()
    with pytest.raises(ValueError, match="requires a non-empty model name"):
        optimizer._model_size_b(
            Node(id="n1", type="inference", engine="vllm", model=None, raw={})
        )


def test_disable_data_profile_drops_supplied_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lumilake_server.runtime.optimizer.halo.envs.LUMILAKE_DISABLE_DATA_PROFILE",
        True,
    )

    optimizer = HaloOptimizer()
    parse_calls: list[object] = []
    build_calls: list[dict[str, tuple[DataProfileResultRow, ...]]] = []
    original_build = optimizer._build_data_profile_plan_choices

    def parse_spy(value: object) -> dict[str, tuple[DataProfileResultRow, ...]]:
        parse_calls.append(value)
        return {}

    def build_spy(graph, parsed):  # type: ignore[no-untyped-def]
        build_calls.append(dict(parsed))
        return original_build(graph, parsed)

    optimizer._parse_data_profile_results = parse_spy  # type: ignore[assignment]
    optimizer._build_data_profile_plan_choices = build_spy  # type: ignore[assignment]

    supplied = {
        "data_profile::r1::r1_query": [
            DataProfileResultRow(
                node_id="r1",
                raw_node_id="r1",
                query_name="r1_query",
                table="public.t",
                cost_estimates=[
                    DataProfileCostEstimate(
                        plan_id="pg_estimate",
                        description="pg projection",
                        raw_cost=1.0,
                        estimated_files=1,
                        total_size_bytes=100,
                        avg_file_size_bytes=100,
                        estimated_rows=37,
                        footprints={},
                    )
                ],
            )
        ]
    }

    with contextlib.suppress(RuntimeError):
        optimizer.generate_schedule(
            graph=_retrieval_graph(
                task_type="data_retrieval", backend="data_retrieval"
            ),
            worker_names=["cpu-0"],
            worker_profiles={"cpu-0": {"has_gpu": False}},
            data_profile_results=supplied,
        )

    assert parse_calls == []
    assert build_calls == [{}]
