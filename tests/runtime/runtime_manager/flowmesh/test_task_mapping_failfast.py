import types
from typing import Any

import pytest

from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


@pytest.mark.asyncio
async def test_resolve_task_node_maps_fails_when_description_fetch_fails(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fetch(
        _self: FlowmeshRuntimeManager,
        task_id: str,
    ) -> dict[str, Any]:
        if task_id == "task-b":
            raise RuntimeError("network error")
        return {"graph_node_name": "node-a"}

    monkeypatch.setattr(
        flowmesh_manager,
        "fetch_task_description",
        types.MethodType(_fetch, flowmesh_manager),
    )

    with pytest.raises(RuntimeError, match="task_id=task-b"):
        await flowmesh_manager._resolve_task_node_maps(
            ["task-a", "task-b"],
            ["node-a", "node-b"],
        )


@pytest.mark.asyncio
async def test_resolve_task_node_maps_fails_when_graph_node_name_is_invalid(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fetch(
        _self: FlowmeshRuntimeManager,
        task_id: str,
    ) -> dict[str, Any]:
        if task_id == "task-a":
            return {"graph_node_name": "node-a"}
        return {"graph_node_name": "unknown-node"}

    monkeypatch.setattr(
        flowmesh_manager,
        "fetch_task_description",
        types.MethodType(_fetch, flowmesh_manager),
    )

    with pytest.raises(RuntimeError, match="missing valid graph_node_name"):
        await flowmesh_manager._resolve_task_node_maps(
            ["task-a", "task-b"],
            ["node-a", "node-b"],
        )
