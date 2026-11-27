from dataclasses import replace
from types import SimpleNamespace

import pytest
from support.runtime_server import make_runtime_op
from support.worker_profiles import worker_profile

from lumilake_hook import GpuDevice
from lumilake_server.runtime.capacity import FreeCapacity
from lumilake_server.runtime.runtime_graph import RuntimeGraph

_OP = replace(
    make_runtime_op("n"), model_spec={"vllm": {"gpu_memory_utilization": 0.8}}
)
_FRACS = {"gpu-small": 0.1, "gpu-big": 0.8}


def _free(*names: str) -> FreeCapacity:
    profiles = {
        n: worker_profile(devices=(GpuDevice(),), gpu_memory_fraction=_FRACS[n])
        for n in names
    }
    return FreeCapacity(cpu_worker_ids=(), gpu_worker_ids=names, profiles=profiles)


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
