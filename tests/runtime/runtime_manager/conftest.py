import pytest
from flowmesh.exceptions import CursorExpiredError
from flowmesh.models.tasks import TaskStatusPage

from lumilake_server.runtime.runtime_manager import FlowmeshRuntimeManager


@pytest.fixture
def flowmesh_manager() -> FlowmeshRuntimeManager:
    return FlowmeshRuntimeManager()


def serve(
    monkeypatch: pytest.MonkeyPatch,
    statuses: dict[str, str],
    nodes: dict[str, str] | None = None,
    errors: dict[str, str | None] | None = None,
    by_workflow: dict[str, dict[str, str]] | None = None,
    cursors: list[str | None] | None = None,
    expired: set[str] | None = None,
    pages: list[TaskStatusPage] | None = None,
) -> list[str]:
    """Answer feed calls from ``statuses``; return the ids polled."""
    polled: list[str] = []
    remaining = list(pages or [])

    async def _list(
        self: FlowmeshRuntimeManager,
        workflow_id: str,
        fields: list[str] | None = None,
        cursor: str | None = None,
    ) -> TaskStatusPage:
        polled.append(workflow_id)
        if cursors is not None:
            cursors.append(cursor)
        if expired is not None and cursor in expired:
            expired.discard(cursor)
            raise CursorExpiredError("cursor cannot be resumed", 410, "GET", "/tasks")
        if remaining:
            return remaining.pop(0)
        listed = statuses if by_workflow is None else by_workflow.get(workflow_id, {})
        rows = [
            {
                "task_id": task_id,
                "status": value,
                "graph_node_name": (nodes or {}).get(task_id),
                "error": (errors or {}).get(task_id),
            }
            for task_id, value in listed.items()
        ]
        return TaskStatusPage(rows=rows, next_cursor=":".join(listed) or "0")

    monkeypatch.setattr(FlowmeshRuntimeManager, "fetch_workflow_task_statuses", _list)
    return polled
