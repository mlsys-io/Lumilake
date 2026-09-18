"""``_select_preview_workers_and_profiles`` must classify API/HTTP nodes as
CPU-required, mirroring HaloOptimizer._map_engine (engine "http" is only
ever assigned to CPU workers, same as "db"/data_retrieval). Otherwise an
API-only or mixed preview can select a GPU-only worker (or no CPU worker at
all), and HALO then rejects the node the preview approved."""

from typing import Any, cast

import pytest

from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _profile(gpu_count: int) -> dict[str, Any]:
    devices = [{"memory_total_bytes": None} for _ in range(gpu_count)]
    return {
        "cpu": {"logical_cores": 16},
        "memory": {"total_bytes": 64 * (1024**3)},
        "gpu": {"devices": devices},
    }


def _api_op(node_id: str) -> RuntimeOp:
    return RuntimeOp(
        node_id=node_id,
        task_type="api",
        backend="api",
        model="model-a",
        data_spec={},
        model_spec={},
        inference_spec={},
        api_spec={"method": "POST", "url": "https://lum.id/llm/v1/chat/completions"},
    )


def _inference_op(node_id: str) -> RuntimeOp:
    return RuntimeOp(
        node_id=node_id,
        task_type="inference",
        backend="vllm",
        model="meta-llama/Llama-3.1-8B-Instruct",
        data_spec={},
        model_spec={},
        inference_spec={},
    )


class _MultiWorkerRuntimeManager:
    def __init__(self, profiles: dict[str, dict[str, Any]]) -> None:
        self._profiles = profiles

    async def get_workers(self) -> list[str]:
        return list(self._profiles)

    async def get_worker_profile(self, worker_id: str) -> dict[str, Any]:
        return self._profiles[worker_id]


@pytest.mark.asyncio
async def test_api_only_preview_selects_a_cpu_worker(server_factory) -> None:
    server = server_factory()
    server.runtime_manager = cast(
        Any,
        _MultiWorkerRuntimeManager(
            {"gpu-worker": _profile(gpu_count=1), "cpu-worker": _profile(gpu_count=0)}
        ),
    )
    runtime_graph = RuntimeGraph(
        nodes={"api-node": _api_op("api-node")},
        node_order=["api-node"],
        output_node_map={"api-node": "output"},
    )

    selected_workers, profiles = await server._select_preview_workers_and_profiles(
        runtime_graph
    )

    assert selected_workers == ["cpu-worker"]
    assert profiles["cpu-worker"]["has_gpu"] is False


@pytest.mark.asyncio
async def test_api_only_preview_raises_when_only_gpu_workers_available(
    server_factory,
) -> None:
    server = server_factory()
    server.runtime_manager = cast(
        Any, _MultiWorkerRuntimeManager({"gpu-worker": _profile(gpu_count=1)})
    )
    runtime_graph = RuntimeGraph(
        nodes={"api-node": _api_op("api-node")},
        node_order=["api-node"],
        output_node_map={"api-node": "output"},
    )

    with pytest.raises(RuntimeError, match="No CPU worker"):
        await server._select_preview_workers_and_profiles(runtime_graph)


@pytest.mark.asyncio
async def test_mixed_gpu_and_api_preview_selects_both_worker_kinds(
    server_factory,
) -> None:
    server = server_factory()
    server.runtime_manager = cast(
        Any,
        _MultiWorkerRuntimeManager(
            {"gpu-worker": _profile(gpu_count=1), "cpu-worker": _profile(gpu_count=0)}
        ),
    )
    runtime_graph = RuntimeGraph(
        nodes={
            "api-node": _api_op("api-node"),
            "llm-node": _inference_op("llm-node"),
        },
        node_order=["api-node", "llm-node"],
        output_node_map={"llm-node": "output"},
    )

    selected_workers, profiles = await server._select_preview_workers_and_profiles(
        runtime_graph
    )

    assert set(selected_workers) == {"gpu-worker", "cpu-worker"}
    assert profiles["gpu-worker"]["has_gpu"] is True
    assert profiles["cpu-worker"]["has_gpu"] is False


@pytest.mark.asyncio
async def test_gpu_only_preview_does_not_require_a_cpu_worker(server_factory) -> None:
    server = server_factory()
    server.runtime_manager = cast(
        Any, _MultiWorkerRuntimeManager({"gpu-worker": _profile(gpu_count=1)})
    )
    runtime_graph = RuntimeGraph(
        nodes={"llm-node": _inference_op("llm-node")},
        node_order=["llm-node"],
        output_node_map={"llm-node": "output"},
    )

    selected_workers, profiles = await server._select_preview_workers_and_profiles(
        runtime_graph
    )

    assert selected_workers == ["gpu-worker"]
    assert profiles["gpu-worker"]["has_gpu"] is True
