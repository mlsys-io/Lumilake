import pytest
from flowmesh.models.tasks import TaskStatusView

from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


@pytest.fixture
def flowmesh_manager() -> FlowmeshRuntimeManager:
    return FlowmeshRuntimeManager()


def serve(
    monkeypatch: pytest.MonkeyPatch,
    statuses: dict[str, str],
    nodes: dict[str, str] | None = None,
    errors: dict[str, str | None] | None = None,
    by_workflow: dict[str, dict[str, str]] | None = None,
) -> list[str]:
    """Answer each workflow status listing from these task statuses (or from
    ``by_workflow`` when one workflow differs from another); return the
    workflow ids the manager polled."""
    polled: list[str] = []

    async def _list(self: FlowmeshRuntimeManager, workflow_id: str) -> list:
        polled.append(workflow_id)
        listed = statuses if by_workflow is None else by_workflow.get(workflow_id, {})
        return [
            TaskStatusView.model_construct(
                task_id=task_id,
                status=value,
                graph_node_name=(nodes or {}).get(task_id),
                error=(errors or {}).get(task_id),
            )
            for task_id, value in listed.items()
        ]

    monkeypatch.setattr(FlowmeshRuntimeManager, "fetch_workflow_task_statuses", _list)
    return polled
