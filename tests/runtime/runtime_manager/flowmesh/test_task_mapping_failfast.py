import types

import pytest

from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager
from tests.runtime.runtime_manager.flowmesh.conftest import serve


@pytest.mark.asyncio
async def test_resolve_task_node_maps_fails_when_task_list_fails(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _list(_self: FlowmeshRuntimeManager, workflow_id: str) -> list:
        raise RuntimeError("network error")

    monkeypatch.setattr(
        flowmesh_manager,
        "fetch_workflow_task_statuses",
        types.MethodType(_list, flowmesh_manager),
    )

    with pytest.raises(RuntimeError, match="workflow_id=wf-1"):
        await flowmesh_manager._resolve_task_node_maps(
            "wf-1", ["task-a", "task-b"], ["node-a", "node-b"]
        )


@pytest.mark.asyncio
async def test_resolve_task_node_maps_fails_when_graph_node_name_is_invalid(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serve(
        monkeypatch,
        {"task-a": "PENDING", "task-b": "PENDING"},
        {"task-a": "node-a", "task-b": "node-x"},
    )

    with pytest.raises(RuntimeError, match="missing valid graph_node_name"):
        await flowmesh_manager._resolve_task_node_maps(
            "wf-1", ["task-a", "task-b"], ["node-a", "node-b"]
        )
