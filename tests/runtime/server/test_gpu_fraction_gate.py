"""A worker's GPU memory fraction gates the claim and the schedule preview."""

from types import SimpleNamespace
from typing import Any

import pytest

from lumilake_server.runtime.capacity import FreeCapacity
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp

_FRACTIONS = {"gpu-small": 0.1, "gpu-big": 0.8, "gpu-whole": None}


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
        # The claim passes over a worker whose fraction is too small.
        (("gpu-small", "gpu-big"), ("gpu-small", "gpu-big"), 1, ["gpu-big"]),
        # The fitting worker is busy: the claim waits rather than take the small one.
        (("gpu-small", "gpu-big"), ("gpu-small",), 1, None),
        # A whole device holds any utilization.
        (("gpu-whole",), ("gpu-whole",), 1, ["gpu-whole"]),
        # A batch claiming no GPU worker is not gated on the fractions.
        (("gpu-small", "gpu-big"), ("gpu-small", "gpu-big"), 0, None),
        # No worker in the cluster ever fits: the claim proceeds so HALO reports it.
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


class _RuntimeManager:
    def __init__(self, names: tuple[str, ...]) -> None:
        self._names = names

    async def get_workers(self) -> list[str]:
        return list(self._names)

    async def get_worker_profile(self, worker_id: str) -> dict[str, Any]:
        return {
            "cpu": {"logical_cores": 16},
            "gpu": {"devices": [{"memory_fraction": _FRACTIONS[worker_id]}]},
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("names", "expected"),
    [
        # The preview passes over a worker whose fraction is too small, and names
        # the failure when no worker holds the node's utilization.
        (("gpu-small", "gpu-big"), ["gpu-big"]),
        (("gpu-small",), "memory fraction holds"),
    ],
)
async def test_preview_takes_a_worker_whose_fraction_holds_the_node(
    server_factory, names: tuple[str, ...], expected: list[str] | str
) -> None:
    server = server_factory()
    server.runtime_manager = _RuntimeManager(names)
    if isinstance(expected, str):
        with pytest.raises(RuntimeError, match=expected):
            await server._select_preview_workers_and_profiles(_graph())
        return
    assert (await server._select_preview_workers_and_profiles(_graph()))[0] == expected
