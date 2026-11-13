import pytest
from flowmesh.models.tasks import TaskStatusView

from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


@pytest.fixture
def flowmesh_manager() -> FlowmeshRuntimeManager:
    return FlowmeshRuntimeManager()


def serve(
    monkeypatch: pytest.MonkeyPatch,
    tasks: dict[str, tuple[str, str | None]],
    errors: dict[str, str | None] | None = None,
) -> None:
    """Answer every workflow status listing from ``{task id: (status, node)}``,
    read on each call so a test can advance a task's status."""

    async def _list(self: FlowmeshRuntimeManager, workflow_id: str) -> list:
        return [
            TaskStatusView.model_construct(
                task_id=task_id,
                status=status,
                graph_node_name=node,
                error=(errors or {}).get(task_id),
                last_error=None,
            )
            for task_id, (status, node) in tasks.items()
        ]

    monkeypatch.setattr(FlowmeshRuntimeManager, "fetch_workflow_task_statuses", _list)
