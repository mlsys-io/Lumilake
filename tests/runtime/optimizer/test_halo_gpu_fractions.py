from dataclasses import replace

import pytest
from support.runtime_server import make_runtime_op

from lumilake_server.runtime.optimizer.halo import HaloOptimizer
from lumilake_server.runtime.runtime_graph import RuntimeGraph

_SMALL = {"has_gpu": True, "gpu_memory_fraction": 0.1}
_BIG = {"has_gpu": True, "gpu_memory_fraction": 0.8}
_OP = replace(
    make_runtime_op("n"),
    backend="vllm",
    model="Qwen/Qwen3-8B",
    model_spec={"vllm": {"gpu_memory_utilization": 0.8}},
)
_GRAPH = RuntimeGraph(nodes={"n": _OP}, node_order=["n"], output_node_map={})


def test_a_node_lands_only_on_a_worker_whose_fraction_holds_it() -> None:
    optimizer = HaloOptimizer()
    shared = {"gpu-small": _SMALL, "gpu-big": _BIG}
    whole = {"gpu-whole": {"has_gpu": True}}

    for profiles, expected in ((shared, "gpu-big"), (whole, "gpu-whole")):
        schedule = optimizer.generate_schedule(_GRAPH, list(profiles), profiles)
        placed = [w for w, ids in schedule.worker_assignment.items() if ids]
        assert placed == [expected]
    with pytest.raises(RuntimeError) as raised:
        optimizer.generate_schedule(_GRAPH, ["gpu-small"], {"gpu-small": _SMALL})
    assert "gpu_memory_utilization=0.8" in str(raised.value.__cause__)
