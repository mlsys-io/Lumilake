from types import SimpleNamespace
from typing import Any, cast

import pytest

from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _op(node_id: str, task_type: str = "api") -> RuntimeOp:
    return RuntimeOp(node_id, task_type, task_type, "", {}, {}, {})


@pytest.mark.asyncio
async def test_build_task_spec_pins_planned_gpu_nodes_and_offers_cpu_pool(
    flowmesh_manager,
) -> None:
    graph = RuntimeGraph(
        nodes={
            "n1": RuntimeOp(
                node_id="n1",
                task_type="inference",
                backend="vllm",
                model="meta-llama/Llama-3.1-8B-Instruct",
                data_spec={"type": "list", "items": ["a", "b"]},
                model_spec={},
                inference_spec={"max_tokens": 32},
            ),
            "c0": _op("c0"),
            "c1": _op("c1"),
            "omni": _op("omni", "omni_text2image"),
        },
        node_order=["c0", "c1", "n1", "omni"],
        output_node_map={},
        dsl_to_runtime={},
    )
    schedule = Schedule(
        worker_assignment={
            "cpu-0": ["c0"],
            "cpu-1": ["c1"],
            "gpu-0": ["n1", "omni"],
            "gpu-1": [],
        }
    )
    request_info = cast(
        Any,
        SimpleNamespace(
            request_id="req-1",
            hardware_requirements=None,
            gpu_worker_ids={"gpu-0", "gpu-1"},
        ),
    )

    spec, _, _ = await flowmesh_manager._build_task_spec(
        request_info=request_info,
        runtime_graph=graph,
        schedule=schedule,
    )
    annotations = (spec.get("metadata") or {}).get("annotations") or {}
    assert isinstance(annotations, dict)
    assert "source" not in annotations
    assert "request_id" not in annotations
    assert "trace_id" not in annotations
    assert "user_id" not in annotations
    schedule_hint = annotations.get("schedule_hint")
    assert isinstance(schedule_hint, dict)
    assert "node_schedule_in_epoch_order" not in schedule_hint
    assert schedule_hint["node_execution_order"] == ["c0", "c1", "n1", "omni"]
    assert schedule_hint["selected_worker"]["selected"] == {
        **dict.fromkeys(("c0", "c1"), ["cpu-0", "cpu-1"]),
        "n1": ["gpu-0"],
        "omni": ["gpu-0"],
    }
