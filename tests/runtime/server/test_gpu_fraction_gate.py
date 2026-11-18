from dataclasses import replace
from types import SimpleNamespace

import pytest
from support.runtime_server import make_runtime_op

from lumilake_server.runtime.capacity import FreeCapacity
from lumilake_server.runtime.runtime_graph import RuntimeGraph

_OP = replace(
    make_runtime_op("n"), model_spec={"vllm": {"gpu_memory_utilization": 0.8}}
)
_FRACS = {"gpu-small": 0.1, "half-a": 0.5, "half-b": 0.5, "gpu-big": 0.8}


def _free(*names: str) -> FreeCapacity:
    profiles = {n: {"has_gpu": True, "gpu_memory_fraction": _FRACS[n]} for n in names}
    return FreeCapacity(cpu_worker_ids=(), gpu_worker_ids=names, profiles=profiles)


def _util(node_id: str, utilization: float):
    spec = {"vllm": {"gpu_memory_utilization": utilization}}
    return replace(_OP, node_id=node_id, model_spec=spec)


@pytest.mark.asyncio
async def test_try_claim_workers_takes_only_a_worker_whose_fraction_fits(
    server_factory,
) -> None:
    server = server_factory()
    server.config.gpu_worker_group_size = 1
    server.config.cpu_worker_group_size = 0
    batch = SimpleNamespace(
        config=SimpleNamespace(hardware_requirements=None),
        workflows=[SimpleNamespace(request_id="req", id="wf")],
        runtime_graphs={
            "g": RuntimeGraph(nodes={"n": _OP}, node_order=["n"], output_node_map={})
        },
        clustering_seconds=0.0,
    )
    for cluster, idle, expected in (
        (("gpu-small", "gpu-big"), ("gpu-small", "gpu-big"), ["gpu-big"]),
        (("gpu-small", "gpu-big"), ("gpu-small",), None),
        (("gpu-small",), ("gpu-small",), ["gpu-small"]),
    ):

        async def total() -> FreeCapacity:
            return _free(*cluster)

        server._snapshot_total_capacity = total
        assert await server._try_claim_workers(batch, _free(*idle)) == expected


@pytest.mark.asyncio
async def test_try_claim_workers_takes_the_tightest_fitting_worker_per_fraction(
    server_factory,
) -> None:
    small, big = _util("s", 0.1), _util("b", 0.8)
    halves = (_util("a", 0.2), _util("c", 0.3))
    for ops, cluster, group_size, expected in (
        ((big, small), ("gpu-big", "gpu-small"), 1, ["gpu-small", "gpu-big"]),
        (halves, ("half-a", "half-b"), 1, ["half-a", "half-b"]),
        ((big, small), ("gpu-big",), 1, ["gpu-big"]),
        ((small,), ("gpu-big", "gpu-small"), 2, ["gpu-small", "gpu-big"]),
    ):
        server = server_factory()
        server.config.gpu_worker_group_size = group_size
        server.config.cpu_worker_group_size = 0
        names = [op.node_id for op in ops]
        batch = SimpleNamespace(
            config=SimpleNamespace(hardware_requirements=None),
            workflows=[SimpleNamespace(request_id="req", id="wf")],
            runtime_graphs={
                "g": RuntimeGraph(
                    nodes={op.node_id: op for op in ops},
                    node_order=names,
                    output_node_map={},
                )
            },
            clustering_seconds=0.0,
        )
        claimed = await server._try_claim_workers(batch, _free(*cluster))
        assert claimed == expected
