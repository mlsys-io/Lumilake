from types import SimpleNamespace
from typing import Any

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
    nodes = {name: _op(name) for name in ("c0", "c1", "c2")}
    nodes["n1"] = _op("n1", "inference")
    nodes["omni"] = _op("omni", "omni_text2image")
    graph = RuntimeGraph(
        nodes=nodes, node_order=list(nodes), output_node_map={}, dsl_to_runtime={}
    )
    schedule = Schedule(
        worker_assignment={
            "cpu-0": ["c0", "c2"],
            "cpu-1": ["c1"],
            "gpu-0": ["n1", "omni"],
            "gpu-1": [],
        }
    )
    gpu_ids = {"gpu-0", "gpu-1"}
    request_info: Any = SimpleNamespace(
        request_id="req-1", hardware_requirements=None, gpu_worker_ids=gpu_ids
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
    assert schedule_hint["node_execution_order"] == ["c0", "c1", "n1", "c2", "omni"]
    assert schedule_hint["selected_worker"]["selected"] == {
        **dict.fromkeys(("c0", "c1", "c2"), ["cpu-0", "cpu-1"]),
        "n1": ["gpu-0"],
        "omni": ["gpu-0"],
    }
