import contextlib

import pytest

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


def test_model_size_resolution_parses_suffix() -> None:
    optimizer = HaloOptimizer()
    size = optimizer._model_size_b(
        Node(id="n1", type="inference", engine="vllm", model="foo-1.5B", raw={})
    )
    assert size == pytest.approx(1.5)


def test_model_size_resolution_uses_lookup() -> None:
    optimizer = HaloOptimizer()
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


def test_model_size_resolution_rejects_unknown_model() -> None:
    optimizer = HaloOptimizer()
    with pytest.raises(ValueError, match="model size is unknown"):
        optimizer._model_size_b(
            Node(
                id="n1",
                type="inference",
                engine="vllm",
                model="unknown/model-without-size",
                raw={},
            )
        )


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
