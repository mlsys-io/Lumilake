"""TopologicalSortOptimizer places a capability-gated node only on a worker that
advertises its task type; every other node is placed by engine alone."""

import pytest

from lumilake_server.runtime.optimizer.topological_sort import (
    TopologicalSortOptimizer,
)
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _op(node_id: str, task_type: str, backend: str) -> RuntimeOp:
    return RuntimeOp(
        node_id=node_id,
        task_type=task_type,
        backend=backend,
        model="",
        data_spec={},
        model_spec={},
        inference_spec={},
    )


def _graph(*ops: RuntimeOp) -> RuntimeGraph:
    return RuntimeGraph(
        nodes={op.node_id: op for op in ops},
        node_order=[op.node_id for op in ops],
        output_node_map={},
        dsl_to_runtime={},
    )


def test_python_node_goes_to_the_worker_that_advertises_python() -> None:
    schedule = TopologicalSortOptimizer().generate_schedule(
        graph=_graph(_op("p1", "python", "python")),
        worker_names=["cpu-plain", "cpu-python"],
        worker_profiles={
            "cpu-plain": {"has_gpu": False, "supported_task_types": ["echo"]},
            "cpu-python": {"has_gpu": False, "supported_task_types": ["python"]},
        },
    )
    assert schedule.worker_assignment == {"cpu-plain": [], "cpu-python": ["p1"]}


def test_python_node_is_never_placed_on_a_gpu_worker() -> None:
    with pytest.raises(ValueError, match="selected CPU workers"):
        TopologicalSortOptimizer().generate_schedule(
            graph=_graph(_op("p1", "python", "python")),
            worker_names=["gpu-python", "cpu-plain"],
            worker_profiles={
                "gpu-python": {"has_gpu": True, "supported_task_types": ["python"]},
                "cpu-plain": {"has_gpu": False, "supported_task_types": []},
            },
        )


def test_python_node_on_a_gpu_only_pool_fails_rather_than_treating_it_as_cpu() -> None:
    with pytest.raises(ValueError, match="selected CPU workers"):
        TopologicalSortOptimizer().generate_schedule(
            graph=_graph(_op("p1", "python", "python")),
            worker_names=["gpu-python"],
            worker_profiles={
                "gpu-python": {"has_gpu": True, "supported_task_types": ["python"]}
            },
        )


def test_python_node_without_an_advertising_worker_fails() -> None:
    with pytest.raises(ValueError, match="advertises it"):
        TopologicalSortOptimizer().generate_schedule(
            graph=_graph(_op("p1", "python", "python")),
            worker_names=["cpu-plain"],
            worker_profiles={"cpu-plain": {"has_gpu": False}},
        )


def test_other_nodes_ignore_advertised_task_types() -> None:
    schedule = TopologicalSortOptimizer().generate_schedule(
        graph=_graph(_op("a1", "api", "api")),
        worker_names=["cpu-plain"],
        worker_profiles={"cpu-plain": {"has_gpu": False}},
    )
    assert schedule.worker_assignment == {"cpu-plain": ["a1"]}
