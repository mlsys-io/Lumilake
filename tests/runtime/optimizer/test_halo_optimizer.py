import pytest

from lumilake.runtime.optimizer.halo import HaloOptimizer
from lumilake.runtime.optimizer.schedule.models import Node
from lumilake.runtime.runtime_graph import RuntimeGraph
from lumilake.runtime.runtime_ops import RuntimeOp


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
