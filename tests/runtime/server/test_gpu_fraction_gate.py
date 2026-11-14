"""A worker's GPU memory fraction gates a batch's worker claim."""

from types import SimpleNamespace

import pytest

from lumilake_server.runtime.capacity import FreeCapacity
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp

_FRACTIONS = {"gpu-small": 0.1, "gpu-big": 0.8}


def _free(*names: str) -> FreeCapacity:
    profiles = {
        n: {"has_gpu": True, "gpu_memory_fraction": _FRACTIONS[n]} for n in names
    }
    return FreeCapacity(cpu_worker_ids=(), gpu_worker_ids=names, profiles=profiles)


def _graph() -> RuntimeGraph:
    op = RuntimeOp(
        node_id="n",
        task_type="inference",
        backend="vllm",
        model="Qwen/Qwen3-8B",
        data_spec={},
        model_spec={"vllm": {"gpu_memory_utilization": 0.8}},
        inference_spec={},
    )
    return RuntimeGraph(nodes={"n": op}, node_order=["n"], output_node_map={})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cluster", "idle", "group_size", "expected"),
    [
        (("gpu-small", "gpu-big"), ("gpu-small", "gpu-big"), 1, ["gpu-big"]),
        (("gpu-small", "gpu-big"), ("gpu-small",), 1, None),
        (("gpu-small", "gpu-big"), ("gpu-small", "gpu-big"), 0, None),
        (("gpu-small",), ("gpu-small",), 1, ["gpu-small"]),
    ],
)
async def test_try_claim_workers_takes_only_a_worker_whose_fraction_fits(
    server_factory,
    cluster: tuple[str, ...],
    idle: tuple[str, ...],
    group_size: int,
    expected: list[str] | None,
) -> None:
    """It takes the fitting worker and waits while that one is busy; it proceeds
    when the cluster holds none, and a batch claiming no GPU worker is not gated."""
    server = server_factory()
    server.config.gpu_worker_group_size = group_size
    server.config.cpu_worker_group_size = 0
    batch = SimpleNamespace(
        config=SimpleNamespace(hardware_requirements=None),
        workflows=[SimpleNamespace(request_id="req", id="wf")],
        runtime_graphs={"g": _graph()},
        clustering_seconds=0.0,
    )

    async def total() -> FreeCapacity:
        return _free(*cluster)

    server._snapshot_total_capacity = total

    assert await server._try_claim_workers(batch, _free(*idle)) == expected
