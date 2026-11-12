"""HALO must place a node only on a GPU worker whose memory share can hold it.

Two workers can share one card, each owning a fraction of it. A node whose
engine reserves ``gpu_memory_utilization`` of a card cannot run on a worker
that owns less than that, even though both workers report the same card.
"""

from typing import Any

import pytest

from lumilake_server.runtime.optimizer.halo import HaloOptimizer
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp

_GPU_PROFILES: dict[str, dict[str, Any]] = {
    "gpu-0": {"has_gpu": True, "gpu_memory_share": 0.8},
    "gpu-1": {"has_gpu": True, "gpu_memory_share": 0.1},
}


def _inference_op(node_id: str, model: str, utilization: float) -> RuntimeOp:
    return RuntimeOp(
        node_id=node_id,
        task_type="inference",
        backend="vllm",
        model=model,
        data_spec={},
        model_spec={"vllm": {"gpu_memory_utilization": utilization}},
        inference_spec={},
    )


def _shared_card_graph() -> RuntimeGraph:
    """A small model and a large one, each naming the share it reserves."""
    nodes = {
        "generator": _inference_op(
            "generator", "Qwen/Qwen3-32B-Instruct", utilization=0.8
        ),
        "scorer": _inference_op("scorer", "Qwen/Qwen3-0.6B", utilization=0.1),
    }
    return RuntimeGraph(
        nodes=nodes,
        node_order=["generator", "scorer"],
        output_node_map={},
        dsl_to_runtime={},
    )


def _placed_on(schedule: Any, node_id: str) -> str:
    (worker,) = [
        worker_id
        for worker_id, node_ids in schedule.worker_assignment.items()
        if node_id in node_ids
    ]
    return worker


def test_each_node_lands_on_the_worker_whose_share_holds_it() -> None:
    schedule = HaloOptimizer().generate_schedule(
        graph=_shared_card_graph(),
        worker_names=["gpu-0", "gpu-1"],
        worker_profiles=_GPU_PROFILES,
    )

    assert _placed_on(schedule, "generator") == "gpu-0"
    assert _placed_on(schedule, "scorer") == "gpu-1"


def test_placement_does_not_depend_on_worker_order() -> None:
    optimizer = HaloOptimizer()

    first = optimizer.generate_schedule(
        graph=_shared_card_graph(),
        worker_names=["gpu-0", "gpu-1"],
        worker_profiles=_GPU_PROFILES,
    )
    second = optimizer.generate_schedule(
        graph=_shared_card_graph(),
        worker_names=["gpu-1", "gpu-0"],
        worker_profiles=_GPU_PROFILES,
    )

    assert _placed_on(first, "generator") == "gpu-0"
    assert _placed_on(second, "generator") == "gpu-0"
    assert _placed_on(first, "scorer") == "gpu-1"
    assert _placed_on(second, "scorer") == "gpu-1"


def test_a_node_naming_no_utilization_may_land_on_either_shared_worker() -> None:
    nodes = {
        "generator": _inference_op(
            "generator", "Qwen/Qwen3-32B-Instruct", utilization=0.8
        ),
        "scorer": RuntimeOp(
            node_id="scorer",
            task_type="inference",
            backend="vllm",
            model="Qwen/Qwen3-0.6B",
            data_spec={},
            model_spec={},
            inference_spec={},
        ),
    }
    graph = RuntimeGraph(
        nodes=nodes,
        node_order=["generator", "scorer"],
        output_node_map={},
        dsl_to_runtime={},
    )

    schedule = HaloOptimizer().generate_schedule(
        graph=graph,
        worker_names=["gpu-0", "gpu-1"],
        worker_profiles=_GPU_PROFILES,
    )

    assert _placed_on(schedule, "generator") == "gpu-0"
    assert _placed_on(schedule, "scorer") in {"gpu-0", "gpu-1"}


def test_a_node_skips_a_cheaper_but_too_small_worker() -> None:
    """The share alone decides: the small worker comes first and costs the same."""
    graph = RuntimeGraph(
        nodes={
            "generator": _inference_op(
                "generator", "Qwen/Qwen3-32B-Instruct", utilization=0.8
            )
        },
        node_order=["generator"],
        output_node_map={},
        dsl_to_runtime={},
    )

    schedule = HaloOptimizer().generate_schedule(
        graph=graph,
        worker_names=["gpu-a", "gpu-b"],
        worker_profiles={
            "gpu-a": {"has_gpu": True, "gpu_memory_share": 0.1},
            "gpu-b": {"has_gpu": True, "gpu_memory_share": 0.8},
        },
    )

    assert _placed_on(schedule, "generator") == "gpu-b"


def test_a_share_too_small_for_every_gpu_worker_names_the_shares() -> None:
    graph = RuntimeGraph(
        nodes={
            "generator": _inference_op(
                "generator", "Qwen/Qwen3-32B-Instruct", utilization=0.8
            )
        },
        node_order=["generator"],
        output_node_map={},
        dsl_to_runtime={},
    )

    with pytest.raises(RuntimeError) as raised:
        HaloOptimizer().generate_schedule(
            graph=graph,
            worker_names=["gpu-1"],
            worker_profiles={"gpu-1": {"has_gpu": True, "gpu_memory_share": 0.1}},
        )

    cause = raised.value.__cause__
    assert isinstance(cause, ValueError)
    assert "gpu_memory_utilization=0.8" in str(cause)
    assert "gpu-1=0.1" in str(cause)


def test_identical_workers_break_ties_on_the_same_worker_every_time() -> None:
    profiles = {"gpu-a": {"has_gpu": True}, "gpu-b": {"has_gpu": True}}
    graph = _shared_card_graph()

    first = HaloOptimizer().generate_schedule(
        graph=graph,
        worker_names=["gpu-a", "gpu-b"],
        worker_profiles=profiles,
    )
    second = HaloOptimizer().generate_schedule(
        graph=graph,
        worker_names=["gpu-b", "gpu-a"],
        worker_profiles=profiles,
    )

    assert first.worker_assignment == second.worker_assignment


def test_a_whole_device_worker_holds_any_utilization() -> None:
    schedule = HaloOptimizer().generate_schedule(
        graph=_shared_card_graph(),
        worker_names=["gpu-exclusive"],
        worker_profiles={"gpu-exclusive": {"has_gpu": True}},
    )

    assert _placed_on(schedule, "generator") == "gpu-exclusive"
    assert _placed_on(schedule, "scorer") == "gpu-exclusive"
