"""HALO places a node only on a GPU worker whose memory share can hold it."""

import pytest

from lumilake_server.runtime.optimizer.halo import HaloOptimizer
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp

_SMALL = {"has_gpu": True, "gpu_memory_share": 0.1}
_BIG = {"has_gpu": True, "gpu_memory_share": 0.8}


@pytest.mark.parametrize(
    ("profiles", "expected"),
    [
        # The smaller share is listed first; the share alone decides.
        ({"gpu-small": _SMALL, "gpu-big": _BIG}, "gpu-big"),
        ({"gpu-whole": {"has_gpu": True}}, "gpu-whole"),
        ({"gpu-small": _SMALL}, None),
    ],
)
def test_a_node_lands_only_on_a_worker_whose_share_holds_it(
    profiles: dict[str, dict[str, object]], expected: str | None
) -> None:
    op = RuntimeOp(
        node_id="node",
        task_type="inference",
        backend="vllm",
        model="Qwen/Qwen3-32B-Instruct",
        data_spec={},
        model_spec={"vllm": {"gpu_memory_utilization": 0.8}},
        inference_spec={},
    )
    graph = RuntimeGraph(
        nodes={"node": op}, node_order=["node"], output_node_map={}, dsl_to_runtime={}
    )
    optimizer = HaloOptimizer()
    if expected is None:
        with pytest.raises(RuntimeError) as raised:
            optimizer.generate_schedule(graph, list(profiles), profiles)
        assert "gpu_memory_utilization=0.8" in str(raised.value.__cause__)
        assert "gpu-small=0.1" in str(raised.value.__cause__)
        return
    schedule = optimizer.generate_schedule(graph, list(profiles), profiles)
    assert [w for w, ids in schedule.worker_assignment.items() if ids] == [expected]
