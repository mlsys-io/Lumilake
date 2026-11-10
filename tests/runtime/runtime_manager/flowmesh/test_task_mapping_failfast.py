import types

import pytest
from flowmesh.models.tasks import TaskStatusView

from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


def _task(task_id: str, graph_node_name: str | None) -> TaskStatusView:
    return TaskStatusView(
        task_id=task_id,
        workflow_id="wf-1",
        status="PENDING",
        submitted_at="2026-01-01T00:00:00Z",
        submitted_ts=0.0,
        attempts=0,
        max_attempts=1,
        graph_node_name=graph_node_name,
        completed=False,
        failed=False,
    )


@pytest.mark.asyncio
async def test_resolve_task_node_maps_fails_when_task_list_fails(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fetch(
        _self: FlowmeshRuntimeManager, workflow_id: str
    ) -> list[TaskStatusView]:
        raise RuntimeError("network error")

    monkeypatch.setattr(
        flowmesh_manager,
        "fetch_workflow_task_statuses",
        types.MethodType(_fetch, flowmesh_manager),
    )

    with pytest.raises(RuntimeError, match="workflow_id=wf-1"):
        await flowmesh_manager._resolve_task_node_maps(
            "wf-1",
            ["task-a", "task-b"],
            ["node-a", "node-b"],
        )


@pytest.mark.asyncio
async def test_resolve_task_node_maps_fails_when_graph_node_name_is_invalid(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fetch(
        _self: FlowmeshRuntimeManager, workflow_id: str
    ) -> list[TaskStatusView]:
        assert workflow_id == "wf-1"
        return [_task("task-a", "node-a"), _task("task-b", "unknown-node")]

    monkeypatch.setattr(
        flowmesh_manager,
        "fetch_workflow_task_statuses",
        types.MethodType(_fetch, flowmesh_manager),
    )

    with pytest.raises(RuntimeError, match="missing valid graph_node_name"):
        await flowmesh_manager._resolve_task_node_maps(
            "wf-1",
            ["task-a", "task-b"],
            ["node-a", "node-b"],
        )


@pytest.mark.asyncio
async def test_resolve_task_node_maps_rejects_duplicate_graph_node_names(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fetch(
        _self: FlowmeshRuntimeManager, workflow_id: str
    ) -> list[TaskStatusView]:
        return [_task("task-a", "node-a"), _task("task-b", "node-a")]

    monkeypatch.setattr(
        flowmesh_manager,
        "fetch_workflow_task_statuses",
        types.MethodType(_fetch, flowmesh_manager),
    )

    with pytest.raises(RuntimeError, match="Duplicate FlowMesh graph_node_name"):
        await flowmesh_manager._resolve_task_node_maps(
            "wf-1", ["task-a", "task-b"], ["node-a"]
        )


@pytest.mark.asyncio
async def test_resolve_task_node_maps_reports_missing_tasks_and_nodes(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fetch(
        _self: FlowmeshRuntimeManager, workflow_id: str
    ) -> list[TaskStatusView]:
        return [_task("task-a", "node-a")]

    monkeypatch.setattr(
        flowmesh_manager,
        "fetch_workflow_task_statuses",
        types.MethodType(_fetch, flowmesh_manager),
    )

    with pytest.raises(RuntimeError) as excinfo:
        await flowmesh_manager._resolve_task_node_maps(
            "wf-1", ["task-a", "task-b"], ["node-a", "node-b"]
        )
    assert "Missing task_ids=['task-b']" in str(excinfo.value)
    assert "missing node_names=['node-b']" in str(excinfo.value)


@pytest.mark.asyncio
async def test_resolve_task_node_maps_uses_one_status_call_per_workflow(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    full_calls: list[str] = []
    tasks = [_task(f"task-{i}", f"node-{i}") for i in range(1_500)]

    async def _list_statuses(*, workflow_id: str) -> list[TaskStatusView]:
        calls.append(workflow_id)
        return tasks

    async def _list(*, workflow_id: str) -> list[TaskStatusView]:
        full_calls.append(workflow_id)
        return tasks

    class _FakeTasks:
        list = staticmethod(_list)
        list_statuses = staticmethod(_list_statuses)

    class _FakeFm:
        tasks = _FakeTasks()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

    task_to_node, node_to_task = await flowmesh_manager._resolve_task_node_maps(
        "wf-1",
        [task.task_id for task in tasks],
        [node for node in (task.graph_node_name for task in tasks) if node],
    )

    assert calls == ["wf-1"]
    assert full_calls == []
    assert len(task_to_node) == 1_500
    assert len(node_to_task) == 1_500
