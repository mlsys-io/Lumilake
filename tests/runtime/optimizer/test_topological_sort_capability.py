"""TopologicalSortOptimizer places a capability-gated node only on a worker that
advertises its task type; every other node is placed by engine alone."""

import pytest
from support.worker_profiles import worker_profile

from lumilake_hook import GpuDevice
from lumilake_server.runtime.optimizer.topological_sort import (
    TopologicalSortOptimizer,
)
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp
from tests.support.data_specs import runtime_op


def _op(node_id: str, task_type: str, backend: str) -> RuntimeOp:
    return runtime_op(
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


def _gpu_python_op(node_id: str, gpu: dict[str, object] | None = None) -> RuntimeOp:
    return runtime_op(
        node_id=node_id,
        task_type="python",
        backend="python",
        model="",
        data_spec={"code": "x", "gpu": gpu or {"count": 1}},
        model_spec={},
        inference_spec={},
    )


def test_python_node_goes_to_the_worker_that_advertises_python() -> None:
    schedule = TopologicalSortOptimizer().generate_schedule(
        graph=_graph(_op("p1", "python", "python")),
        worker_names=["cpu-plain", "cpu-python"],
        worker_profiles={
            "cpu-plain": worker_profile(task_types={"echo"}),
            "cpu-python": worker_profile(task_types={"python"}),
        },
    )
    assert schedule.worker_assignment == {"cpu-plain": [], "cpu-python": ["p1"]}


def test_python_node_is_never_placed_on_a_gpu_worker() -> None:
    with pytest.raises(ValueError, match="selected CPU workers"):
        TopologicalSortOptimizer().generate_schedule(
            graph=_graph(_op("p1", "python", "python")),
            worker_names=["gpu-python", "cpu-plain"],
            worker_profiles={
                "gpu-python": worker_profile(has_gpu=True, task_types={"python"}),
                "cpu-plain": worker_profile(),
            },
        )


def test_python_node_on_a_gpu_only_pool_fails_rather_than_treating_it_as_cpu() -> None:
    with pytest.raises(ValueError, match="selected CPU workers"):
        TopologicalSortOptimizer().generate_schedule(
            graph=_graph(_op("p1", "python", "python")),
            worker_names=["gpu-python"],
            worker_profiles={
                "gpu-python": worker_profile(has_gpu=True, task_types={"python"})
            },
        )


def test_python_node_without_an_advertising_worker_fails() -> None:
    with pytest.raises(ValueError, match="advertises it"):
        TopologicalSortOptimizer().generate_schedule(
            graph=_graph(_op("p1", "python", "python")),
            worker_names=["cpu-plain"],
            worker_profiles={"cpu-plain": worker_profile()},
        )


def test_gpu_request_skips_a_python_gpu_worker_it_does_not_fit() -> None:
    h100 = GpuDevice(name="H100", memory_total_bytes=80 * 1024**3)
    profiles = {
        "gpu-rtx": worker_profile(
            has_gpu=True,
            task_types={"python"},
            gpu_devices=(GpuDevice(name="RTX 4090"),),
        ),
        "gpu-h100": worker_profile(
            has_gpu=True,
            task_types={"python"},
            gpu_devices=(h100, h100),
        ),
    }
    schedule = TopologicalSortOptimizer().generate_schedule(
        graph=_graph(_gpu_python_op("p1", {"count": 2, "type": "H100"})),
        worker_names=["gpu-rtx", "gpu-h100"],
        worker_profiles=profiles,
    )
    assert schedule.worker_assignment == {"gpu-rtx": [], "gpu-h100": ["p1"]}
    without_carrier = {
        "gpu-rtx": profiles["gpu-rtx"],
        "gpu-h100": profiles["gpu-h100"].model_copy(
            update={"supported_task_types": frozenset()}
        ),
    }
    with pytest.raises(ValueError, match=r"carries the GPUs node 'p1' requests"):
        TopologicalSortOptimizer().generate_schedule(
            graph=_graph(_gpu_python_op("p1", {"count": 2, "type": "H100"})),
            worker_names=["gpu-rtx", "gpu-h100"],
            worker_profiles=without_carrier,
        )
    with pytest.raises(ValueError, match=r"carries the GPUs node 'p1' requests"):
        TopologicalSortOptimizer().generate_schedule(
            graph=_graph(_gpu_python_op("p1", {"count": 2, "type": "H100"})),
            worker_names=["gpu-rtx", "gpu-h100"],
            worker_profiles={
                "gpu-rtx": profiles["gpu-rtx"],
                "gpu-h100": profiles["gpu-h100"].model_copy(
                    update={"gpu_devices": (h100,)}
                ),
            },
        )


def test_other_nodes_ignore_advertised_task_types() -> None:
    schedule = TopologicalSortOptimizer().generate_schedule(
        graph=_graph(_op("a1", "api", "api")),
        worker_names=["cpu-plain"],
        worker_profiles={"cpu-plain": worker_profile()},
    )
    assert schedule.worker_assignment == {"cpu-plain": ["a1"]}
