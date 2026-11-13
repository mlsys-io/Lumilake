"""HALO places a node only on a GPU worker whose memory share can hold it."""

import pytest

from lumilake_server.runtime.optimizer.halo import HaloOptimizer
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp

_SMALL = {"has_gpu": True, "gpu_memory_share": 0.1}
_BIG = {"has_gpu": True, "gpu_memory_share": 0.8}


def _inference_op(
    node_id: str,
    model: str,
    utilization: float,
    dependencies: tuple[str, ...] = (),
) -> RuntimeOp:
    return RuntimeOp(
        node_id=node_id,
        task_type="inference",
        backend="vllm",
        model=model,
        data_spec={},
        model_spec={"vllm": {"gpu_memory_utilization": utilization}},
        inference_spec={},
        dependencies=dependencies,
    )


@pytest.mark.parametrize(
    ("profiles", "expected"),
    [
        # The smaller share is listed first; the share alone decides.
        ({"gpu-a": _SMALL, "gpu-b": _BIG}, "gpu-b"),
        ({"gpu-whole": {"has_gpu": True}}, "gpu-whole"),
        ({"gpu-a": _SMALL}, None),
    ],
)
def test_a_node_lands_only_on_a_worker_whose_share_holds_it(
    profiles: dict[str, dict[str, object]], expected: str | None
) -> None:
    op = _inference_op("node", "Qwen/Qwen3-32B-Instruct", 0.8)
    graph = RuntimeGraph(
        nodes={"node": op}, node_order=["node"], output_node_map={}, dsl_to_runtime={}
    )
    optimizer = HaloOptimizer()
    if expected is None:
        with pytest.raises(RuntimeError) as raised:
            optimizer.generate_schedule(graph, list(profiles), profiles)
        assert "gpu_memory_utilization=0.8" in str(raised.value.__cause__)
        assert "gpu-a=0.1" in str(raised.value.__cause__)
        return
    schedule = optimizer.generate_schedule(graph, list(profiles), profiles)
    assert [w for w, ids in schedule.worker_assignment.items() if ids] == [expected]


def test_dedup_keeps_workers_whose_shares_differ_apart() -> None:
    """The half card and the whole device are not interchangeable: only the whole
    device holds the 0.9 node, so both 8B nodes belong there."""
    nodes = {
        "big-a": _inference_op("big-a", "Qwen/Qwen3-8B", 0.3),
        "small": _inference_op("small", "Qwen/Qwen3-0.6B", 0.3),
        "big-b": _inference_op("big-b", "Qwen/Qwen3-8B", 0.9, ("big-a",)),
    }
    schedule = HaloOptimizer().generate_schedule(
        graph=RuntimeGraph(
            nodes=nodes,
            node_order=list(nodes),
            output_node_map={},
            dsl_to_runtime={},
        ),
        worker_names=["gpu-half", "gpu-whole"],
        worker_profiles={
            "gpu-half": {"has_gpu": True, "gpu_memory_share": 0.5},
            "gpu-whole": {"has_gpu": True},
        },
    )

    placement = {
        node_id: worker
        for worker, node_ids in schedule.worker_assignment.items()
        for node_id in node_ids
    }
    assert placement == {
        "small": "gpu-half",
        "big-a": "gpu-whole",
        "big-b": "gpu-whole",
    }
