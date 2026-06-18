"""``_build_task_spec`` must read hardware from ``RequestInfo`` and only fall
back to ``HARDWARE_*`` env defaults when the override field is unset.

Mixed overrides (e.g. only ``cpu`` set) must merge field-by-field so a
partial override doesn't silently drop the other knobs back to env."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.protocol import HardwareRequirements
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _gpu_graph() -> RuntimeGraph:
    return RuntimeGraph(
        nodes={
            "n1": RuntimeOp(
                node_id="n1",
                task_type="inference",
                backend="vllm",
                model="meta-llama/Llama-3.1-8B-Instruct",
                data_spec={"type": "list", "items": ["a"]},
                model_spec={},
                inference_spec={"max_tokens": 32},
            )
        },
        node_order=["n1"],
        output_node_map={},
        dsl_to_runtime={},
    )


@pytest.mark.asyncio
async def test_task_spec_uses_full_hardware_override(
    flowmesh_manager: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Env defaults that would be visible if the override leaked.
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.envs.HARDWARE_CPU_REQUIREMENT",
        8,
    )
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.envs.HARDWARE_MEMORY_REQUIREMENT",
        "16Gi",
    )
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.envs.HARDWARE_GPU_REQUIREMENT",
        1,
    )
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.envs.HARDWARE_GPU_MEMORY_REQUIREMENT",
        "8Gi",
    )

    hardware = HardwareRequirements(cpu=32, memory="128Gi", gpu=2, gpu_memory="48Gi")
    request_info = cast(
        Any,
        SimpleNamespace(request_id="req-1", hardware_requirements=hardware),
    )

    spec, _, _ = await flowmesh_manager._build_task_spec(
        request_info=request_info,
        runtime_graph=_gpu_graph(),
        schedule=Schedule(worker_assignment={"gpu-0": ["n1"]}),
    )

    top_hardware = spec["spec"]["resources"]["hardware"]
    assert top_hardware == {"cpu": 32, "memory": "128Gi"}

    # Per-node GPU hint must reflect the override, not the env defaults.
    nodes = spec["spec"]["graph"]["nodes"]
    assert len(nodes) == 1
    node_hardware = nodes[0]["spec"]["resources"]["hardware"]
    assert node_hardware["cpu"] == 32
    assert node_hardware["memory"] == "128Gi"
    assert node_hardware["gpu"] == {"type": "any", "count": 2, "memory": "48Gi"}


@pytest.mark.asyncio
async def test_task_spec_partial_override_falls_back_to_env_per_field(
    flowmesh_manager: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only ``memory`` set on the override: cpu/gpu/gpu_memory must come from env."""
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.envs.HARDWARE_CPU_REQUIREMENT",
        4,
    )
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.envs.HARDWARE_MEMORY_REQUIREMENT",
        "16Gi",
    )
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.envs.HARDWARE_GPU_REQUIREMENT",
        1,
    )
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.envs.HARDWARE_GPU_MEMORY_REQUIREMENT",
        "8Gi",
    )

    hardware = HardwareRequirements(memory="64Gi")
    request_info = cast(
        Any,
        SimpleNamespace(request_id="req-2", hardware_requirements=hardware),
    )

    spec, _, _ = await flowmesh_manager._build_task_spec(
        request_info=request_info,
        runtime_graph=_gpu_graph(),
        schedule=Schedule(worker_assignment={"gpu-0": ["n1"]}),
    )

    top_hardware = spec["spec"]["resources"]["hardware"]
    assert top_hardware == {"cpu": 4, "memory": "64Gi"}

    node_hardware = spec["spec"]["graph"]["nodes"][0]["spec"]["resources"]["hardware"]
    assert node_hardware["cpu"] == 4
    assert node_hardware["memory"] == "64Gi"
    assert node_hardware["gpu"] == {"type": "any", "count": 1, "memory": "8Gi"}


@pytest.mark.asyncio
async def test_task_spec_no_override_uses_env_defaults(
    flowmesh_manager: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.envs.HARDWARE_CPU_REQUIREMENT",
        12,
    )
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.envs.HARDWARE_MEMORY_REQUIREMENT",
        "24Gi",
    )

    request_info = cast(
        Any,
        SimpleNamespace(request_id="req-3", hardware_requirements=None),
    )

    spec, _, _ = await flowmesh_manager._build_task_spec(
        request_info=request_info,
        runtime_graph=_gpu_graph(),
        schedule=Schedule(worker_assignment={"gpu-0": ["n1"]}),
    )
    assert spec["spec"]["resources"]["hardware"] == {"cpu": 12, "memory": "24Gi"}


@pytest.mark.asyncio
async def test_task_spec_rejects_gpu_zero_against_gpu_op(
    flowmesh_manager: Any,
) -> None:
    """Defense-in-depth: even when an internal caller bypasses the HTTP route
    guard and passes ``HardwareRequirements(gpu=0)`` with a GPU workflow,
    the runtime layer must refuse to silently upgrade gpu_count to 1."""
    request_info = cast(
        Any,
        SimpleNamespace(
            request_id="req-no-gpu",
            hardware_requirements=HardwareRequirements(gpu=0),
        ),
    )

    with pytest.raises(ValueError, match="requires a GPU worker"):
        await flowmesh_manager._build_task_spec(
            request_info=request_info,
            runtime_graph=_gpu_graph(),
            schedule=Schedule(worker_assignment={"gpu-0": ["n1"]}),
        )
