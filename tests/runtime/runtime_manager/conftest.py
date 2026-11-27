from typing import Any

import pytest
from flowmesh.exceptions import CursorExpiredError
from flowmesh.models.result import AnyExecutorResult
from flowmesh.models.tasks import TaskInfo, TaskStatusPage
from pydantic import TypeAdapter

from lumilake import envs
from lumilake_server.runtime.runtime_manager import FlowmeshRuntimeManager
from lumilake_server.utils.job_storage import InMemoryJobStorage

API_URL = "https://api.example.com/v1/chat"

RESULT_ADAPTER: TypeAdapter[AnyExecutorResult] = TypeAdapter(AnyExecutorResult)


def api_result(**payload: Any) -> AnyExecutorResult:
    """An SDK API result as a task wrote it; ``payload`` carries its fields."""
    return RESULT_ADAPTER.validate_python(
        {
            "task_type": "api",
            "executor": "api",
            "method": "POST",
            "url": API_URL,
            "status_code": 200,
            **payload,
        }
    )


def python_result(value: Any) -> AnyExecutorResult:
    """An SDK python result carrying ``value`` as a step's return value."""
    return RESULT_ADAPTER.validate_python(
        {"task_type": "python", "exit_code": 0, "value": value}
    )


def task_info(submit_id: str | None = None, *, task_id: str = "t1") -> TaskInfo:
    """A FlowMesh task document, its annotation carrying ``submit_id``."""
    annotations = {"custom": {"lumilake_submit_id": submit_id}}
    return TaskInfo.model_validate(
        {
            "task_id": task_id,
            "workflow_id": "wf-1",
            "owner_id": "owner",
            "org_id": "org",
            "supplier_id": "supplier",
            "source": "kind: Task",
            "task": {"metadata": {"annotations": annotations}},
            "status": "DONE",
            "submitted_at": "2026-11-27T00:00:00Z",
            "submitted_ts": 0.0,
            "usages": [],
            "attempts": 0,
            "max_attempts": 1,
            "load": 0,
            "depends_on": [],
            "pending_dependencies": [],
            "dependents": [],
            "completed": True,
            "failed": False,
        }
    )


@pytest.fixture
def flowmesh_manager() -> FlowmeshRuntimeManager:
    return FlowmeshRuntimeManager()


@pytest.fixture(autouse=True)
def job_storage(monkeypatch: pytest.MonkeyPatch) -> InMemoryJobStorage:
    """Keep archived artifacts and published task blobs in memory."""
    storage = InMemoryJobStorage()
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.get_job_storage", lambda: storage
    )
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")
    return storage


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
