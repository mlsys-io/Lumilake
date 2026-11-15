"""A worker's GPU memory fraction gates a batch's worker claim."""

from types import SimpleNamespace

import pytest

from lumilake_server.runtime.capacity import FreeCapacity
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp

_FRACTIONS = {"gpu-small": 0.1, "gpu-big": 0.8}


def _capacity(*names: str) -> FreeCapacity:
    profiles = {
        n: {"has_gpu": True, "gpu_memory_fraction": _FRACTIONS[n]} for n in names
    }
    return FreeCapacity(cpu_worker_ids=(), gpu_worker_ids=names, profiles=profiles)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cluster", "idle", "expected"),
    [
        (("gpu-small", "gpu-big"), ("gpu-small", "gpu-big"), ["gpu-big"]),
        (("gpu-small", "gpu-big"), ("gpu-small",), None),
    ],
)
async def test_try_claim_workers_takes_only_a_worker_whose_fraction_fits(
    server_factory,
    cluster: tuple[str, ...],
    idle: tuple[str, ...],
    expected: list[str] | None,
) -> None:
    server = server_factory()
    server.config.gpu_worker_group_size = 1
    server.config.cpu_worker_group_size = 0
    op = RuntimeOp(
        "n",
        "inference",
        "vllm",
        "Qwen/Qwen3-8B",
        {},
        {"vllm": {"gpu_memory_utilization": 0.8}},
        {},
    )
    batch = SimpleNamespace(
        config=SimpleNamespace(hardware_requirements=None),
        workflows=[SimpleNamespace(request_id="req", id="wf")],
        runtime_graphs={
            "g": RuntimeGraph(nodes={"n": op}, node_order=["n"], output_node_map={})
        },
        clustering_seconds=0.0,
    )

    async def total() -> FreeCapacity:
        return _capacity(*cluster)

    server._snapshot_total_capacity = total

    assert await server._try_claim_workers(batch, _capacity(*idle)) == expected
