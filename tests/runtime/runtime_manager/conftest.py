from collections.abc import Mapping

import pytest
from flowmesh.models.tasks import TaskStatusPage

from lumilake_server.runtime.runtime_manager import FlowmeshRuntimeManager


@pytest.fixture
def flowmesh_manager() -> FlowmeshRuntimeManager:
    return FlowmeshRuntimeManager()


def serve(
    monkeypatch: pytest.MonkeyPatch,
    statuses: Mapping[str, str | tuple[str, str | None]],
    nodes: dict[str, str] | None = None,
    errors: dict[str, str | None] | None = None,
    by_workflow: Mapping[str, Mapping[str, str]] | None = None,
    cursors: list[str | None] | None = None,
    page_cursor: str = "",
) -> list[str]:
    """Answer each workflow status feed call from these task statuses (or from
    ``by_workflow`` when one workflow differs from another); return the
    workflow ids the manager polled.

    A value may be a status, or a ``(status, node)`` pair that also sets the
    task's node name. Each call's cursor is appended to ``cursors`` when given,
    and the reply carries ``page_cursor`` as the feed position for the next
    call."""
    polled: list[str] = []

    async def _list(
        self: FlowmeshRuntimeManager,
        workflow_id: str,
        fields: tuple[str, ...] | None = None,
        cursor: str | None = None,
    ) -> TaskStatusPage:
        polled.append(workflow_id)
        if cursors is not None:
            cursors.append(cursor)
        listed = statuses if by_workflow is None else by_workflow.get(workflow_id, {})
        rows = []
        for task_id, value in listed.items():
            status, node = value if isinstance(value, tuple) else (value, None)
            rows.append(
                {
                    "task_id": task_id,
                    "status": status,
                    "graph_node_name": (nodes or {}).get(task_id) or node,
                    "error": (errors or {}).get(task_id),
                    "last_error": None,
                }
            )
        return TaskStatusPage(rows=rows, cursor=page_cursor)

    monkeypatch.setattr(FlowmeshRuntimeManager, "fetch_workflow_task_statuses", _list)
    return polled
