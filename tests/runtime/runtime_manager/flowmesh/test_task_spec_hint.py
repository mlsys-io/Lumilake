from types import SimpleNamespace
from typing import Any, cast

import pytest

from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp


@pytest.mark.asyncio
async def test_build_task_spec_emits_flat_schedule_hint_without_epoch_flag(
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
            )
        },
        node_order=["n1"],
        output_node_map={},
        dsl_to_runtime={},
    )
    schedule = Schedule(worker_assignment={"gpu-0": ["n1"]})
    request_info = cast(Any, SimpleNamespace(request_id="req-1"))

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
    assert schedule_hint.get("node_execution_order") == ["n1"]
    assert "node_schedule_in_epoch_order" not in schedule_hint
