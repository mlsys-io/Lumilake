import pytest
from flowmesh.models.tasks import TaskStatusView

from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


@pytest.fixture
def flowmesh_manager() -> FlowmeshRuntimeManager:
    return FlowmeshRuntimeManager()


def status(
    task_id: str, value: str = "DONE", node: str | None = None, error: str | None = None
) -> TaskStatusView:
    return TaskStatusView.model_construct(
        task_id=task_id,
        status=value,
        error=error,
        last_error=None,
        graph_node_name=node,
    )


def serve(monkeypatch: pytest.MonkeyPatch, tasks: list[TaskStatusView]) -> None:
    """Answer every workflow status listing with ``tasks``, read on each call."""

    async def _list(self: FlowmeshRuntimeManager, workflow_id: str) -> list:
        return tasks

    monkeypatch.setattr(FlowmeshRuntimeManager, "fetch_workflow_task_statuses", _list)
