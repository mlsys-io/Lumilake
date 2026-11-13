import types

import pytest

from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager
from tests.runtime.runtime_manager.flowmesh.conftest import serve, status


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
@pytest.mark.parametrize(
    ("listed", "match"),
    [
        ([("task-a", "node-a"), ("task-b", "unknown-node")], "graph_node_name"),
        ([("task-a", "node-a"), ("task-b", "node-a")], "Duplicate"),
        ([("task-a", "node-a")], "incomplete"),
    ],
)
async def test_resolve_task_node_maps_rejects_an_inconsistent_workflow(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
    listed: list[tuple[str, str]],
    match: str,
) -> None:
    serve(
        monkeypatch,
        [status(task_id, "PENDING", node) for task_id, node in listed],
    )

    with pytest.raises(RuntimeError, match=match):
        await flowmesh_manager._resolve_task_node_maps(
            "wf-1", ["task-a", "task-b"], ["node-a", "node-b"]
        )
