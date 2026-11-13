import pytest
from flowmesh.models.tasks import TaskStatusPage

from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


@pytest.fixture
def flowmesh_manager() -> FlowmeshRuntimeManager:
    return FlowmeshRuntimeManager()


def serve(
    monkeypatch: pytest.MonkeyPatch,
    tasks: dict[str, tuple[str, str | None]],
    errors: dict[str, str | None] | None = None,
    cursors: list[str | None] | None = None,
    page_cursor: str = "",
) -> None:
    """Answer every workflow status listing from ``{task id: (status, node)}``,
    read on each call so a test can advance a task's status. Each call's cursor
    is appended to ``cursors`` when given, and the reply carries
    ``page_cursor`` as the feed position for the next call."""

    async def _list(
        self: FlowmeshRuntimeManager,
        workflow_id: str,
        fields: tuple[str, ...] | None = None,
        cursor: str | None = None,
    ) -> TaskStatusPage:
        if cursors is not None:
            cursors.append(cursor)
        return TaskStatusPage(
            rows=[
                {
                    "task_id": task_id,
                    "status": status,
                    "graph_node_name": node,
                    "error": (errors or {}).get(task_id),
                    "last_error": None,
                }
                for task_id, (status, node) in tasks.items()
            ],
            cursor=page_cursor,
        )

    monkeypatch.setattr(FlowmeshRuntimeManager, "fetch_workflow_task_statuses", _list)
