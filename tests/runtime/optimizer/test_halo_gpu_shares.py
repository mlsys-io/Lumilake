"""HALO places a node only on a GPU worker whose memory share can hold it.

Two workers can share one card, each owning a fraction of it. A node whose
engine reserves ``gpu_memory_utilization`` of a card cannot run on a worker
that owns less than that, even though both workers report the same card.
"""

from typing import Any

import pytest

from lumilake_server.runtime.optimizer.halo import HaloOptimizer
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp

_SMALL = {"has_gpu": True, "gpu_memory_share": 0.1}
_BIG = {"has_gpu": True, "gpu_memory_share": 0.8}


def _op(
    node_id: str, utilization: float | None, dependencies: tuple[str, ...] = ()
) -> RuntimeOp:
    return RuntimeOp(
        node_id=node_id,
        task_type="inference",
        backend="vllm",
        model="Qwen/Qwen3-32B-Instruct",
        data_spec={},
        model_spec=(
            {}
            if utilization is None
            else {"vllm": {"gpu_memory_utilization": utilization}}
        ),
        inference_spec={},
        dependencies=dependencies,
    )


def _placed(profiles: dict[str, dict[str, Any]], *ops: RuntimeOp) -> dict[str, str]:
    graph = RuntimeGraph(
        nodes={op.node_id: op for op in ops},
        node_order=[op.node_id for op in ops],
        output_node_map={},
        dsl_to_runtime={},
    )
    schedule = HaloOptimizer().generate_schedule(
        graph=graph, worker_names=list(profiles), worker_profiles=profiles
    )
    return {
        node_id: worker
        for worker, node_ids in schedule.worker_assignment.items()
        for node_id in node_ids
    }


def test_a_node_lands_on_a_worker_whose_share_holds_it() -> None:
    """The smaller share is listed first; the share alone decides."""
    placed = _placed({"gpu-small": _SMALL, "gpu-big": _BIG}, _op("node", 0.8))
    assert placed == {"node": "gpu-big"}


def test_a_share_too_small_for_every_worker_names_the_shares() -> None:
    with pytest.raises(RuntimeError) as raised:
        _placed({"gpu-small": _SMALL}, _op("node", 0.8))

    cause = raised.value.__cause__
    assert isinstance(cause, ValueError)
    assert "gpu_memory_utilization=0.8" in str(cause)
    assert "gpu-small=0.1" in str(cause)
