from dataclasses import replace
from types import SimpleNamespace

import pytest
from support.runtime_server import make_runtime_op

from lumilake_server.runtime.capacity import FreeCapacity
from lumilake_server.runtime.runtime_graph import RuntimeGraph

_OP = replace(
    make_runtime_op("n"),
    backend="vllm",
    model="Qwen/Qwen3-8B",
    model_spec={"vllm": {"gpu_memory_utilization": 0.8}},
)


def _capacity(*names: str) -> FreeCapacity:
    profiles = {n: {"has_gpu": True, "gpu_memory_fraction": 0.8} for n in names}
    profiles["gpu-small"]["gpu_memory_fraction"] = 0.1
    return FreeCapacity(cpu_worker_ids=(), gpu_worker_ids=names, profiles=profiles)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("idle", "cluster", "expected"),
    [
        (("gpu-small", "gpu-big"), ("gpu-small", "gpu-big"), ["gpu-big"]),
        (("gpu-small",), ("gpu-small", "gpu-big"), None),
        (("gpu-small",), ("gpu-small",), ["gpu-small"]),
    ],
)
async def test_try_claim_workers_takes_only_a_worker_whose_fraction_fits(
    server_factory,
    idle: tuple[str, ...],
    cluster: tuple[str, ...],
    expected: list[str] | None,
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

    async def total() -> FreeCapacity:
        return _capacity(*cluster)

    server._snapshot_total_capacity = total

    assert await server._try_claim_workers(batch, _capacity(*idle)) == expected
