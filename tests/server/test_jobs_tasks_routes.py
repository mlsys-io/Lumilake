"""Coverage for the per-job FlowMesh task + log proxy routes."""

import datetime as dt
import logging
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from flowmesh.exceptions import NotFoundError
from lumid_hooks import PrincipalContext, ResourceRef

import lumilake_server.utils.job_storage as job_storage_module
from lumilake_server import hooks
from lumilake_server.middleware import TraceIdMiddleware
from lumilake_server.routes import jobs as job_routes_module
from lumilake_server.routes.jobs import JobRecord
from lumilake_server.schemas.io import S3Location
from lumilake_server.utils.job_storage import InMemoryJobStorage

_DEMO_PRINCIPAL = PrincipalContext(
    principal_id="alice",
    org_id="demo",
    external_id="alice@example.com",
    principal_type="user",
    scopes=["admin"],
)


class _AllowAllIdentity:
    name = "test.identity"

    async def resolve(
        self, token: str, logger: logging.Logger
    ) -> PrincipalContext | None:
        return _DEMO_PRINCIPAL.model_copy(deep=True) if token == "token" else None


class _AllowAllPermissions:
    name = "test.permissions"

    async def accessible_ids(
        self,
        principal: PrincipalContext,
        kind: str,
        action: str,
        logger: logging.Logger,
    ) -> frozenset[str] | None:
        return None

    async def require(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        action: str,
        logger: logging.Logger,
    ) -> None:
        return None


class _FakeTaskInfo:
    def __init__(self, task_id: str, workflow_id: str) -> None:
        self.task_id = task_id
        self.workflow_id = workflow_id
        self.status = "SUCCEEDED"
        self.task_type = "compute"
        self.category = None
        self.graph_node_name = "data_prep"
        self.assigned_worker = "w-1"
        self.submitted_at = "2026-05-31T00:00:00Z"
        self.started_ts = 1.0
        self.finished_ts = 2.0
        self.attempts = 1
        self.completed = True
        self.failed = False
        self.error = None


class _FakeLogEvent:
    def __init__(self, message: str) -> None:
        self.ts = "2026-05-31T00:00:00Z"
        self.workflow_id = "wf-1"
        self.task_id = "t-a"
        self.worker_id = "w-1"
        self.node_id = None
        self.level = "INFO"
        self.stream = "stdout"
        self.source = None
        self.message = message
        self.fields = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "node_id": self.node_id,
            "level": self.level,
            "stream": self.stream,
            "source": self.source,
            "message": self.message,
            "fields": self.fields,
        }


class _FakeLogEntry:
    def __init__(self, cursor: str, message: str) -> None:
        self.cursor = cursor
        self.event = _FakeLogEvent(message)


class _FakeLogQueryResponse:
    def __init__(self, entries: list[_FakeLogEntry], next_cursor: str | None) -> None:
        self.entries = entries
        self.next_cursor = next_cursor
        self.prev_cursor = None


class _FakeTasks:
    def __init__(
        self,
        list_result: list[_FakeTaskInfo],
        logs_result: _FakeLogQueryResponse,
        retrieve_result: _FakeTaskInfo | None = None,
    ) -> None:
        self._list_result = list_result
        self._logs_result = logs_result
        self._retrieve_result = retrieve_result
        self.list_calls: list[str] = []
        self.log_calls: list[tuple[str, int, str | None, str | None]] = []
        self.retrieve_calls: list[str] = []

    async def retrieve(self, task_id: str) -> _FakeTaskInfo:
        self.retrieve_calls.append(task_id)
        for t in self._list_result:
            if t.task_id == task_id:
                return t
        if self._retrieve_result is not None:
            return self._retrieve_result
        raise NotFoundError(f"task {task_id} not found")

    async def list(self, *, workflow_id: str) -> list[_FakeTaskInfo]:
        self.list_calls.append(workflow_id)
        return [t for t in self._list_result if t.workflow_id == workflow_id]

    async def get_logs(
        self,
        task_id: str,
        limit: int = 200,
        before: str | None = None,
        after: str | None = None,
    ) -> _FakeLogQueryResponse:
        self.log_calls.append((task_id, limit, before, after))
        return self._logs_result


class _FakeFlowMesh:
    def __init__(self, tasks: _FakeTasks) -> None:
        self.tasks = tasks


@pytest.fixture(autouse=True)
def _reset_hook_state() -> Iterator[None]:
    hooks.IDENTITY_PROVIDERS.clear()
    hooks.SUBMISSION_GUARDS.clear()
    hooks.USAGE_SINKS.clear()
    hooks.PERMISSION_CHECKERS.clear()
    hooks.RESOURCE_REGISTRARS.clear()
    hooks.IDENTITY_PROVIDERS.append(_AllowAllIdentity())
    hooks.PERMISSION_CHECKERS.append(_AllowAllPermissions())
    yield
    hooks.IDENTITY_PROVIDERS.clear()
    hooks.SUBMISSION_GUARDS.clear()
    hooks.USAGE_SINKS.clear()
    hooks.PERMISSION_CHECKERS.clear()
    hooks.RESOURCE_REGISTRARS.clear()


@pytest.fixture
def job_routes() -> Any:
    storage = InMemoryJobStorage()
    job_storage_module._job_storage = storage
    job_routes_module.jobs.clear()
    job_routes_module._job_storage = storage
    return job_routes_module


@pytest.fixture
def app(job_routes: Any) -> FastAPI:
    application = FastAPI()
    application.state.logger = logging.getLogger("test.jobs_tasks_routes")
    application.state.compute_db_pool = None
    application.state.background_tasks = set()
    application.add_middleware(TraceIdMiddleware)
    application.include_router(job_routes.router)
    return application


def _seed_job(job_routes: Any, job_id: str, trace_ids: list[str]) -> None:
    record = JobRecord(
        job_id=job_id,
        status="completed",
        submitted_at=dt.datetime.now(dt.UTC).isoformat(),
        inputs={},
        output_location={"out": S3Location(type="s3", prefix="x/y")},
        org_id="demo",
        user_id="alice@example.com",
    )
    record.trace_ids = trace_ids
    job_routes.jobs[job_id] = record
    job_routes._job_storage.save(record)


@pytest.mark.anyio
async def test_list_tasks_fans_in_over_each_workflow_id(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_job(job_routes, "j-1", ["wf-1", "wf-2"])
    fake_tasks = _FakeTasks(
        list_result=[
            _FakeTaskInfo("t-a", "wf-1"),
            _FakeTaskInfo("t-b", "wf-2"),
        ],
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    monkeypatch.setattr(
        job_routes_module, "flowmesh_for", lambda _request: _FakeFlowMesh(fake_tasks)
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/tasks", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 200
    body = resp.json()["data"]
    assert body["job_id"] == "j-1"
    assert sorted(t["task_id"] for t in body["tasks"]) == ["t-a", "t-b"]
    assert sorted(fake_tasks.list_calls) == ["wf-1", "wf-2"]


@pytest.mark.anyio
async def test_list_tasks_empty_when_no_trace_ids(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_job(job_routes, "j-empty", [])

    def _no_fm(_request: Any) -> Any:
        raise AssertionError("flowmesh_for should not be called when no trace ids")

    monkeypatch.setattr(job_routes_module, "flowmesh_for", _no_fm)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-empty/tasks", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 200
    assert resp.json()["data"]["tasks"] == []


@pytest.mark.anyio
async def test_get_logs_forwards_cursor_params(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_job(job_routes, "j-1", ["wf-1"])
    fake_tasks = _FakeTasks(
        list_result=[_FakeTaskInfo("t-a", "wf-1")],
        logs_result=_FakeLogQueryResponse(
            entries=[_FakeLogEntry("c1", "hello"), _FakeLogEntry("c2", "world")],
            next_cursor="c2",
        ),
    )
    monkeypatch.setattr(
        job_routes_module, "flowmesh_for", lambda _request: _FakeFlowMesh(fake_tasks)
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/tasks/t-a/logs",
            params={"limit": 50, "after": "c0"},
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 200
    body = resp.json()["data"]
    assert body["job_id"] == "j-1"
    assert body["task_id"] == "t-a"
    assert body["next_cursor"] == "c2"
    assert [e["event"]["message"] for e in body["entries"]] == ["hello", "world"]
    assert fake_tasks.retrieve_calls == ["t-a"]
    assert fake_tasks.log_calls == [("t-a", 50, None, "c0")]


@pytest.mark.anyio
async def test_list_tasks_unknown_job_returns_404(app: FastAPI) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/nope/tasks", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_get_logs_task_from_different_job_returns_404(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task whose workflow_id is not in the job's trace_ids must yield 404.

    Crucially, tasks.get_logs must never be called — this prevents cross-job
    log exfiltration when the caller's FlowMesh token permits the foreign task.
    """
    _seed_job(job_routes, "j-1", ["wf-1"])
    # t-foreign belongs to wf-other, which is NOT in j-1's trace_ids.
    fake_tasks = _FakeTasks(
        list_result=[_FakeTaskInfo("t-foreign", "wf-other")],
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
        retrieve_result=_FakeTaskInfo("t-foreign", "wf-other"),
    )
    monkeypatch.setattr(
        job_routes_module, "flowmesh_for", lambda _request: _FakeFlowMesh(fake_tasks)
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/tasks/t-foreign/logs",
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 404
    assert (
        fake_tasks.log_calls == []
    ), "tasks.get_logs must not be called for a foreign task"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
