"""Coverage for the per-job FlowMesh workflow + log proxy routes."""

import datetime as dt
import io
import json
import logging
import tarfile
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from flowmesh.exceptions import APIError, FlowMeshConnectionError, NotFoundError
from flowmesh.models.result import ResultEnvelope
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


class _FakeWorkflow:
    def __init__(
        self,
        workflow_id: str,
        status: str = "COMPLETED",
        completed_tasks: list[str] | None = None,
    ) -> None:
        self.workflow_id = workflow_id
        self.status = status
        self.submitted_at = "2026-05-31T00:00:00Z"
        self.task_ids = [
            "tsk-00000001",
            "tsk-00000002",
            "tsk-00000003",
            "tsk-00000004",
            "tsk-00000005",
        ]
        self.completed_tasks = completed_tasks if completed_tasks is not None else []
        self.failed_tasks = ["tsk-00000005"]


class _FakeLogEvent:
    def __init__(self, message: str) -> None:
        self.ts = "2026-05-31T00:00:00Z"
        self.workflow_id = "wf-1"
        self.task_id = "tsk-0000000a"
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

    def model_dump(self) -> dict[str, Any]:
        return {"cursor": self.cursor, "event": self.event.model_dump()}


class _FakeLogQueryResponse:
    def __init__(self, entries: list[_FakeLogEntry], next_cursor: str | None) -> None:
        self.entries = entries
        self.next_cursor = next_cursor
        self.prev_cursor = None


class _FakeWorkflows:
    def __init__(
        self,
        workflows: dict[str, _FakeWorkflow],
        logs_result: _FakeLogQueryResponse,
        stream_entries: list[_FakeLogEntry] | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self._workflows = workflows
        self._logs_result = logs_result
        self._stream_entries = stream_entries or []
        self._stream_error = stream_error
        self.retrieve_calls: list[str] = []
        self.log_calls: list[tuple[str, int, str | None, str | None]] = []
        self.stream_calls: list[tuple[str, str | None]] = []

    async def retrieve(self, workflow_id: str) -> _FakeWorkflow:
        self.retrieve_calls.append(workflow_id)
        if workflow_id in self._workflows:
            return self._workflows[workflow_id]
        raise NotFoundError(
            f"workflow {workflow_id} not found",
            status_code=404,
            method="GET",
            url=f"/workflows/{workflow_id}",
        )

    async def get_logs(
        self,
        workflow_id: str,
        limit: int = 200,
        before: str | None = None,
        after: str | None = None,
    ) -> _FakeLogQueryResponse:
        self.log_calls.append((workflow_id, limit, before, after))
        return self._logs_result

    async def stream_logs(
        self, workflow_id: str, cursor: str | None = None
    ) -> AsyncIterator[_FakeLogEntry]:
        self.stream_calls.append((workflow_id, cursor))
        for entry in self._stream_entries:
            yield entry
        if self._stream_error is not None:
            raise self._stream_error


class _FakeHttpxResponse:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def json(self) -> Any:
        return json.loads(self.content)


class _FakeResults:
    def __init__(
        self,
        results: dict[str, Any],
        errors: dict[str, Exception] | None = None,
    ) -> None:
        self._results = results
        self._errors = errors or {}
        self.retrieve_calls: list[str] = []

    async def retrieve(self, task_id: str) -> Any:
        self.retrieve_calls.append(task_id)
        if task_id in self._errors:
            raise self._errors[task_id]
        if task_id in self._results:
            return ResultEnvelope.model_validate(
                {"task_id": task_id, "result": self._results[task_id]}
            ).result
        raise NotFoundError(
            f"task {task_id} not found",
            status_code=404,
            method="GET",
            url=f"/results/{task_id}",
        )


class _FakeFlowMesh:
    def __init__(
        self,
        workflows: _FakeWorkflows,
        task_log_bytes: dict[str, bytes] | None = None,
        task_log_errors: dict[str, Exception] | None = None,
        workflow_errors: dict[str, Exception] | None = None,
        task_results: dict[str, Any] | None = None,
        task_result_errors: dict[str, Exception] | None = None,
    ) -> None:
        self.workflows = workflows
        self._task_log_bytes: dict[str, bytes] = task_log_bytes or {}
        self._task_log_errors: dict[str, Exception] = task_log_errors or {}
        self._workflow_errors: dict[str, Exception] = workflow_errors or {}
        self.results = _FakeResults(task_results or {}, task_result_errors)
        self.raw_calls: list[tuple[str, str]] = []

    async def _request_raw(self, method: str, path: str) -> _FakeHttpxResponse:
        self.raw_calls.append((method, path))
        if path.startswith("/workflows/"):
            workflow_id = path.split("/workflows/", 1)[-1]
            if workflow_id in self._workflow_errors:
                raise self._workflow_errors[workflow_id]
            wf = self.workflows._workflows.get(workflow_id)
            if wf is None:
                raise NotFoundError(
                    f"workflow {workflow_id} not found",
                    status_code=404,
                    method=method,
                    url=path,
                )
            payload = {
                "workflow_id": wf.workflow_id,
                "status": wf.status,
                "submitted_at": wf.submitted_at,
                "task_ids": wf.task_ids,
                "completed_tasks": wf.completed_tasks,
                "failed_tasks": wf.failed_tasks,
            }
            return _FakeHttpxResponse(json.dumps(payload).encode())
        task_id = path.split("/results/", 1)[-1].split("/logs")[0]
        if task_id in self._task_log_errors:
            raise self._task_log_errors[task_id]
        if task_id in self._task_log_bytes:
            return _FakeHttpxResponse(self._task_log_bytes[task_id])
        raise NotFoundError(
            f"task {task_id} not found",
            status_code=404,
            method=method,
            url=path,
        )


def _api_result(
    task_id: str,
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "task_type": "api",
        "executor": "x",
        "method": "GET",
        "url": "http://x",
        "status_code": 200,
        "usage_summary": usage,
    }


def _inference_result(
    task_id: str,
    usage: dict[str, Any] | None,
    children: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"task_type": "inference", "usage": usage}
    if children:
        payload["children"] = children
    return payload


def _echo_result(task_id: str) -> dict[str, Any]:
    return {"task_type": "echo"}


def _skip_result(task_id: str) -> dict[str, Any]:
    return {}


def _unknown_result(task_id: str) -> dict[str, Any]:
    return {"task_type": "quantum_thing"}


def _api_usage(
    prompt: int,
    completion: int,
    reasoning: int = 0,
    calls: int = 1,
    retries: int = 0,
    truncated: int = 0,
    wall: float = 1.0,
) -> dict[str, Any]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "calls": calls,
        "retries": retries,
        "truncated_calls": truncated,
        "wall_sec": wall,
    }


def _gen_usage(
    prompt: int,
    completion: int,
    num_requests: int,
    latency: float,
) -> dict[str, Any]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "num_requests": num_requests,
        "latency_sec": latency,
    }


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
    application.state.logger = logging.getLogger("test.jobs_workflows_routes")
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
async def test_list_workflows_fans_over_trace_ids(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_job(job_routes, "j-1", ["wf-1", "wf-2"])
    fake_workflows = _FakeWorkflows(
        workflows={
            "wf-1": _FakeWorkflow("wf-1", "COMPLETED"),
            "wf-2": _FakeWorkflow("wf-2", "RUNNING"),
        },
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: _FakeFlowMesh(fake_workflows),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/workflows", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 200
    body = resp.json()["data"]
    assert body["job_id"] == "j-1"
    assert sorted(w["workflow_id"] for w in body["workflows"]) == ["wf-1", "wf-2"]
    # Each workflow is retrieved exactly once: once for status, and the
    # already-retrieved workflow is reused for the usage path.
    assert fake_workflows.retrieve_calls == ["wf-1", "wf-2"]


@pytest.mark.anyio
async def test_list_workflows_empty_when_no_trace_ids(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_job(job_routes, "j-empty", [])

    def _no_fm(_request: Any) -> Any:
        raise AssertionError("flowmesh_for must not be called when no trace ids")

    monkeypatch.setattr(job_routes_module, "flowmesh_for", _no_fm)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-empty/workflows", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 200
    assert resp.json()["data"]["workflows"] == []


@pytest.mark.anyio
async def test_list_workflows_passes_usage_through(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-workflow usage is passed through; null when no task reports usage."""
    _seed_job(job_routes, "j-1", ["wf-1", "wf-2"])
    wf1 = _FakeWorkflow(
        "wf-1", "COMPLETED", completed_tasks=["tsk-00000001", "tsk-00000002"]
    )
    wf2 = _FakeWorkflow("wf-2", "COMPLETED", completed_tasks=["tsk-00000003"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": wf1, "wf-2": wf2},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(
        fake_workflows,
        task_results={
            "tsk-00000001": _api_result(
                "tsk-00000001",
                _api_usage(100, 50, reasoning=10, calls=4, retries=2, wall=3.5),
            ),
            "tsk-00000002": _echo_result("tsk-00000002"),
            "tsk-00000003": _unknown_result("tsk-00000003"),
        },
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: fake_fm,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/workflows", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 200
    workflows = {w["workflow_id"]: w for w in resp.json()["data"]["workflows"]}
    assert workflows["wf-1"]["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "reasoning_tokens": 10,
        "calls": 4,
        "retries": 2,
        "truncated_calls": 0,
        "wall_sec": 3.5,
    }
    assert workflows["wf-2"]["usage"] is None


@pytest.mark.anyio
async def test_list_workflows_keeps_workflow_when_task_result_vanished(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task result FlowMesh no longer has must not drop the workflow from
    the listing; it is still listed with status/counts and null usage."""
    _seed_job(job_routes, "j-1", ["wf-1"])
    wf1 = _FakeWorkflow(
        "wf-1", "COMPLETED", completed_tasks=["tsk-00000001", "tsk-00000002"]
    )
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": wf1},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(
        fake_workflows,
        task_results={
            "tsk-00000001": _api_result(
                "tsk-00000001",
                _api_usage(100, 50, reasoning=10, calls=4, retries=2, wall=3.5),
            ),
        },
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: fake_fm,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/workflows", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 200
    workflows = resp.json()["data"]["workflows"]
    assert len(workflows) == 1
    assert workflows[0]["workflow_id"] == "wf-1"
    assert workflows[0]["status"] == "COMPLETED"
    assert workflows[0]["succeeded_count"] == 2
    assert workflows[0]["usage"] is None


@pytest.mark.anyio
async def test_list_workflows_skips_workflow_when_retrieve_not_found(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A NotFoundError from the workflow retrieve itself skips the workflow."""
    _seed_job(job_routes, "j-1", ["wf-1", "wf-2"])
    wf1 = _FakeWorkflow("wf-1", "COMPLETED", completed_tasks=["tsk-00000001"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": wf1},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(
        fake_workflows,
        task_results={
            "tsk-00000001": _api_result(
                "tsk-00000001",
                _api_usage(100, 50, reasoning=10, calls=4, retries=2, wall=3.5),
            ),
        },
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: fake_fm,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/workflows", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 200
    workflows = resp.json()["data"]["workflows"]
    assert [w["workflow_id"] for w in workflows] == ["wf-1"]


class _FakeProgressServer:
    is_started = True

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def get_request_status(self, job_id: str) -> dict[str, Any]:
        return self._payload


@pytest.mark.anyio
async def test_progress_sums_usage_over_workflows(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Job progress sums usage over the workflows' task results."""
    _seed_job(job_routes, "j-1", ["wf-1", "wf-2"])
    job_routes.jobs["j-1"].status = "running"
    wf1 = _FakeWorkflow("wf-1", "COMPLETED", completed_tasks=["tsk-00000001"])
    wf2 = _FakeWorkflow("wf-2", "COMPLETED", completed_tasks=["tsk-00000002"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": wf1, "wf-2": wf2},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(
        fake_workflows,
        task_results={
            "tsk-00000001": _api_result(
                "tsk-00000001",
                _api_usage(100, 50, reasoning=10, calls=4, retries=2, wall=3.5),
            ),
            "tsk-00000002": _inference_result(
                "tsk-00000002",
                _gen_usage(200, 25, num_requests=2, latency=1.5),
            ),
        },
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: fake_fm,
    )
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_started_instance",
        classmethod(
            lambda cls: _FakeProgressServer({"execution": {"completed": True}})
        ),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/progress", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 200
    usage = resp.json()["data"]["progress"]["usage"]
    assert usage == {
        "prompt_tokens": 300,
        "completion_tokens": 75,
        "reasoning_tokens": 10,
        "calls": 6,
        "retries": 2,
        "truncated_calls": 0,
        "wall_sec": 5.0,
    }


@pytest.mark.anyio
async def test_progress_merged_inference_parent_not_double_counted(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A merged inference parent subtracts its children's shares."""
    _seed_job(job_routes, "j-1", ["wf-1"])
    job_routes.jobs["j-1"].status = "running"
    wf1 = _FakeWorkflow(
        "wf-1", "COMPLETED", completed_tasks=["tsk-parent", "tsk-child"]
    )
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": wf1},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(
        fake_workflows,
        task_results={
            "tsk-parent": _inference_result(
                "tsk-parent",
                _gen_usage(100, 200, num_requests=4, latency=10.0),
                children={
                    "tsk-child": {
                        "task_type": "inference",
                        "usage": _gen_usage(30, 50, num_requests=1, latency=2.0),
                    }
                },
            ),
            "tsk-child": _inference_result(
                "tsk-child",
                _gen_usage(30, 50, num_requests=1, latency=2.0),
            ),
        },
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: fake_fm,
    )
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_started_instance",
        classmethod(
            lambda cls: _FakeProgressServer({"execution": {"completed": True}})
        ),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/progress", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 200
    usage = resp.json()["data"]["progress"]["usage"]
    # parent own share (100-30, 200-50, 4-1, 10.0-2.0) + child (30, 50, 1, 2.0)
    assert usage == {
        "prompt_tokens": 100,
        "completion_tokens": 200,
        "reasoning_tokens": 0,
        "calls": 4,
        "retries": 0,
        "truncated_calls": 0,
        "wall_sec": 10.0,
    }


@pytest.mark.anyio
async def test_progress_skipped_task_contributes_nothing(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A condition-skipped task contributes nothing, not null."""
    _seed_job(job_routes, "j-1", ["wf-1"])
    job_routes.jobs["j-1"].status = "running"
    wf1 = _FakeWorkflow(
        "wf-1", "COMPLETED", completed_tasks=["tsk-00000001", "tsk-skip"]
    )
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": wf1},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(
        fake_workflows,
        task_results={
            "tsk-00000001": _api_result(
                "tsk-00000001",
                _api_usage(100, 50, reasoning=10, calls=4, retries=2, wall=3.5),
            ),
            "tsk-skip": _skip_result("tsk-skip"),
        },
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: fake_fm,
    )
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_started_instance",
        classmethod(
            lambda cls: _FakeProgressServer({"execution": {"completed": True}})
        ),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/progress", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 200
    usage = resp.json()["data"]["progress"]["usage"]
    assert usage == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "reasoning_tokens": 10,
        "calls": 4,
        "retries": 2,
        "truncated_calls": 0,
        "wall_sec": 3.5,
    }


@pytest.mark.anyio
async def test_progress_no_model_task_contributes_nothing(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A no-model task (echo) contributes nothing, not null."""
    _seed_job(job_routes, "j-1", ["wf-1"])
    job_routes.jobs["j-1"].status = "running"
    wf1 = _FakeWorkflow(
        "wf-1", "COMPLETED", completed_tasks=["tsk-00000001", "tsk-echo"]
    )
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": wf1},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(
        fake_workflows,
        task_results={
            "tsk-00000001": _api_result(
                "tsk-00000001",
                _api_usage(100, 50, reasoning=10, calls=4, retries=2, wall=3.5),
            ),
            "tsk-echo": _echo_result("tsk-echo"),
        },
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: fake_fm,
    )
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_started_instance",
        classmethod(
            lambda cls: _FakeProgressServer({"execution": {"completed": True}})
        ),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/progress", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 200
    usage = resp.json()["data"]["progress"]["usage"]
    assert usage == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "reasoning_tokens": 10,
        "calls": 4,
        "retries": 2,
        "truncated_calls": 0,
        "wall_sec": 3.5,
    }


@pytest.mark.anyio
async def test_progress_usage_null_when_task_unmappable(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unmappable model-calling task makes the job usage null."""
    _seed_job(job_routes, "j-1", ["wf-1", "wf-2"])
    job_routes.jobs["j-1"].status = "running"
    wf1 = _FakeWorkflow("wf-1", "COMPLETED", completed_tasks=["tsk-00000001"])
    wf2 = _FakeWorkflow("wf-2", "COMPLETED", completed_tasks=["tsk-00000002"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": wf1, "wf-2": wf2},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(
        fake_workflows,
        task_results={
            "tsk-00000001": _api_result(
                "tsk-00000001",
                _api_usage(100, 50, reasoning=10, calls=4, retries=2, wall=3.5),
            ),
            "tsk-00000002": _unknown_result("tsk-00000002"),
        },
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: fake_fm,
    )
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_started_instance",
        classmethod(
            lambda cls: _FakeProgressServer({"execution": {"completed": True}})
        ),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/progress", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 200
    assert resp.json()["data"]["progress"]["usage"] is None


@pytest.mark.anyio
async def test_progress_terminal_usage_survives_persistence(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal job's usage is read from the persisted record, not FlowMesh."""
    _seed_job(job_routes, "j-1", ["wf-1"])
    record = job_routes.jobs["j-1"]
    record.status = "completed"
    record.progress.usage = job_routes_module.WorkflowUsage(
        prompt_tokens=100,
        completion_tokens=50,
        reasoning_tokens=10,
        calls=4,
        retries=2,
        truncated_calls=0,
        wall_sec=3.5,
    )
    job_routes._job_storage.save(record)

    def _no_fm(_request: Any) -> Any:
        raise AssertionError("flowmesh_for must not be called for a terminal job")

    monkeypatch.setattr(job_routes_module, "flowmesh_for", _no_fm)
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_started_instance",
        classmethod(lambda cls: _FakeProgressServer({"error": "Request ID not found"})),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/progress", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 200
    usage = resp.json()["data"]["progress"]["usage"]
    assert usage == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "reasoning_tokens": 10,
        "calls": 4,
        "retries": 2,
        "truncated_calls": 0,
        "wall_sec": 3.5,
    }


@pytest.mark.anyio
async def test_persist_terminal_usage_null_on_fetch_error(
    job_routes: Any,
) -> None:
    """An upstream error at terminal time persists null, not a partial sum."""
    _seed_job(job_routes, "j-1", ["wf-1", "wf-2"])
    record = job_routes.jobs["j-1"]
    record.status = "completed"
    wf1 = _FakeWorkflow("wf-1", "COMPLETED", completed_tasks=["tsk-00000001"])
    wf2 = _FakeWorkflow("wf-2", "COMPLETED", completed_tasks=["tsk-00000002"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": wf1, "wf-2": wf2},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(
        workflows=fake_workflows,
        task_results={
            "tsk-00000001": _api_result(
                "tsk-00000001",
                _api_usage(100, 50, reasoning=10, calls=4, retries=2, wall=3.5),
            ),
        },
        task_result_errors={
            "tsk-00000002": APIError(
                "upstream failure",
                status_code=500,
                method="GET",
                url="/results/tsk-00000002",
            )
        },
    )
    await job_routes_module._persist_terminal_usage(
        record, fake_fm, logging.getLogger("test")
    )
    assert record.progress.usage is None


@pytest.mark.anyio
async def test_persist_terminal_usage_null_on_not_found(
    job_routes: Any,
) -> None:
    """A task result FlowMesh no longer knows at terminal time persists null."""
    _seed_job(job_routes, "j-1", ["wf-1", "wf-2"])
    record = job_routes.jobs["j-1"]
    record.status = "completed"
    wf1 = _FakeWorkflow("wf-1", "COMPLETED", completed_tasks=["tsk-00000001"])
    wf2 = _FakeWorkflow("wf-2", "COMPLETED", completed_tasks=["tsk-00000002"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": wf1, "wf-2": wf2},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(
        workflows=fake_workflows,
        task_results={
            "tsk-00000001": _api_result(
                "tsk-00000001",
                _api_usage(100, 50, reasoning=10, calls=4, retries=2, wall=3.5),
            ),
        },
    )
    await job_routes_module._persist_terminal_usage(
        record, fake_fm, logging.getLogger("test")
    )
    assert record.progress.usage is None


@pytest.mark.anyio
async def test_progress_live_null_when_task_result_vanished(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A vanished task result makes the live job usage null, not a partial sum."""
    _seed_job(job_routes, "j-1", ["wf-1", "wf-2"])
    job_routes.jobs["j-1"].status = "running"
    wf1 = _FakeWorkflow("wf-1", "COMPLETED", completed_tasks=["tsk-00000001"])
    wf2 = _FakeWorkflow("wf-2", "COMPLETED", completed_tasks=["tsk-00000002"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": wf1, "wf-2": wf2},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(
        fake_workflows,
        task_results={
            "tsk-00000001": _api_result(
                "tsk-00000001",
                _api_usage(100, 50, reasoning=10, calls=4, retries=2, wall=3.5),
            ),
        },
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: fake_fm,
    )
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_started_instance",
        classmethod(
            lambda cls: _FakeProgressServer({"execution": {"completed": True}})
        ),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/progress", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 200
    assert resp.json()["data"]["progress"]["usage"] is None


@pytest.mark.anyio
async def test_progress_live_null_when_api_task_usage_missing(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An API task with no usage makes the live job usage null."""
    _seed_job(job_routes, "j-1", ["wf-1", "wf-2"])
    job_routes.jobs["j-1"].status = "running"
    wf1 = _FakeWorkflow("wf-1", "COMPLETED", completed_tasks=["tsk-00000001"])
    wf2 = _FakeWorkflow("wf-2", "COMPLETED", completed_tasks=["tsk-00000002"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": wf1, "wf-2": wf2},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(
        fake_workflows,
        task_results={
            "tsk-00000001": _api_result(
                "tsk-00000001",
                _api_usage(100, 50, reasoning=10, calls=4, retries=2, wall=3.5),
            ),
            "tsk-00000002": _api_result("tsk-00000002", None),
        },
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: fake_fm,
    )
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_started_instance",
        classmethod(
            lambda cls: _FakeProgressServer({"execution": {"completed": True}})
        ),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/progress", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 200
    assert resp.json()["data"]["progress"]["usage"] is None


@pytest.mark.anyio
async def test_persist_terminal_usage_null_when_api_task_usage_missing(
    job_routes: Any,
) -> None:
    """An API task with no usage at terminal time persists null."""
    _seed_job(job_routes, "j-1", ["wf-1", "wf-2"])
    record = job_routes.jobs["j-1"]
    record.status = "completed"
    wf1 = _FakeWorkflow("wf-1", "COMPLETED", completed_tasks=["tsk-00000001"])
    wf2 = _FakeWorkflow("wf-2", "COMPLETED", completed_tasks=["tsk-00000002"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": wf1, "wf-2": wf2},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(
        workflows=fake_workflows,
        task_results={
            "tsk-00000001": _api_result(
                "tsk-00000001",
                _api_usage(100, 50, reasoning=10, calls=4, retries=2, wall=3.5),
            ),
            "tsk-00000002": _api_result("tsk-00000002", None),
        },
    )
    await job_routes_module._persist_terminal_usage(
        record, fake_fm, logging.getLogger("test")
    )
    assert record.progress.usage is None


@pytest.mark.anyio
async def test_progress_live_no_stale_value_when_usage_now_null(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job whose task now reports unmappable usage ends null, not stale total."""
    _seed_job(job_routes, "j-1", ["wf-1"])
    record = job_routes.jobs["j-1"]
    record.status = "running"
    record.progress.usage = job_routes_module.WorkflowUsage(
        prompt_tokens=100,
        completion_tokens=50,
        reasoning_tokens=10,
        calls=4,
        retries=2,
        truncated_calls=0,
        wall_sec=3.5,
    )
    job_routes._job_storage.save(record)
    wf1 = _FakeWorkflow("wf-1", "COMPLETED", completed_tasks=["tsk-00000001"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": wf1},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(
        fake_workflows,
        task_results={"tsk-00000001": _unknown_result("tsk-00000001")},
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: fake_fm,
    )
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_started_instance",
        classmethod(
            lambda cls: _FakeProgressServer({"execution": {"completed": True}})
        ),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/progress", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 200
    assert resp.json()["data"]["progress"]["usage"] is None


@pytest.mark.anyio
async def test_progress_all_zero_tasks_give_zero_total(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All-zero task usage gives an all-zero job total, not null."""
    _seed_job(job_routes, "j-1", ["wf-1", "wf-2"])
    job_routes.jobs["j-1"].status = "running"
    zero = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "calls": 0,
        "retries": 0,
        "truncated_calls": 0,
        "wall_sec": 0.0,
    }
    wf1 = _FakeWorkflow("wf-1", "COMPLETED", completed_tasks=["tsk-00000001"])
    wf2 = _FakeWorkflow("wf-2", "COMPLETED", completed_tasks=["tsk-00000002"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": wf1, "wf-2": wf2},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(
        fake_workflows,
        task_results={
            "tsk-00000001": _api_result("tsk-00000001", zero),
            "tsk-00000002": _api_result("tsk-00000002", zero),
        },
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: fake_fm,
    )
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_started_instance",
        classmethod(
            lambda cls: _FakeProgressServer({"execution": {"completed": True}})
        ),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/progress", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 200
    usage = resp.json()["data"]["progress"]["usage"]
    assert usage == zero


@pytest.mark.anyio
async def test_progress_malformed_result_returns_502(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed task result from FlowMesh returns 502."""
    _seed_job(job_routes, "j-1", ["wf-1"])
    job_routes.jobs["j-1"].status = "running"
    wf1 = _FakeWorkflow("wf-1", "COMPLETED", completed_tasks=["tsk-00000001"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": wf1},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(
        fake_workflows,
        task_results={
            "tsk-00000001": {
                "task_type": "api",
                "executor": "x",
                "method": "GET",
                "url": "http://x",
                "status_code": 200,
                "usage_summary": {"prompt_tokens": "not-an-int"},
            }
        },
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: fake_fm,
    )
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_started_instance",
        classmethod(
            lambda cls: _FakeProgressServer({"execution": {"completed": True}})
        ),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/progress", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 502


@pytest.mark.anyio
async def test_persist_terminal_usage_null_on_connection_error(
    job_routes: Any,
) -> None:
    """A FlowMeshConnectionError at terminal time persists null."""
    _seed_job(job_routes, "j-1", ["wf-1"])
    record = job_routes.jobs["j-1"]
    record.status = "completed"
    wf1 = _FakeWorkflow("wf-1", "COMPLETED", completed_tasks=["tsk-00000001"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": wf1},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(
        workflows=fake_workflows,
        task_result_errors={"tsk-00000001": FlowMeshConnectionError("down")},
    )
    await job_routes_module._persist_terminal_usage(
        record, fake_fm, logging.getLogger("test")
    )
    assert record.progress.usage is None


@pytest.mark.anyio
async def test_progress_live_connection_error_returns_502(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A FlowMeshConnectionError on the live path returns 502."""
    _seed_job(job_routes, "j-1", ["wf-1"])
    job_routes.jobs["j-1"].status = "running"
    wf1 = _FakeWorkflow("wf-1", "COMPLETED", completed_tasks=["tsk-00000001"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": wf1},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(
        fake_workflows,
        task_result_errors={"tsk-00000001": FlowMeshConnectionError("down")},
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: fake_fm,
    )
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_started_instance",
        classmethod(
            lambda cls: _FakeProgressServer({"execution": {"completed": True}})
        ),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/progress", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 502


@pytest.mark.anyio
async def test_persist_terminal_usage_null_on_read_timeout(
    job_routes: Any,
) -> None:
    """A httpx.ReadTimeout at terminal time persists null."""
    _seed_job(job_routes, "j-1", ["wf-1"])
    record = job_routes.jobs["j-1"]
    record.status = "completed"
    wf1 = _FakeWorkflow("wf-1", "COMPLETED", completed_tasks=["tsk-00000001"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": wf1},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(
        workflows=fake_workflows,
        task_result_errors={"tsk-00000001": httpx.ReadTimeout("timed out")},
    )
    await job_routes_module._persist_terminal_usage(
        record, fake_fm, logging.getLogger("test")
    )
    assert record.progress.usage is None


@pytest.mark.anyio
async def test_progress_live_read_timeout_returns_502(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A httpx.ReadTimeout on the live path returns 502."""
    _seed_job(job_routes, "j-1", ["wf-1"])
    job_routes.jobs["j-1"].status = "running"
    wf1 = _FakeWorkflow("wf-1", "COMPLETED", completed_tasks=["tsk-00000001"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": wf1},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(
        fake_workflows,
        task_result_errors={"tsk-00000001": httpx.ReadTimeout("timed out")},
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: fake_fm,
    )
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_started_instance",
        classmethod(
            lambda cls: _FakeProgressServer({"execution": {"completed": True}})
        ),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/progress", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 502


@pytest.mark.anyio
async def test_progress_live_empty_job_gives_zero_total(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job with no workflows gets an all-zero usage, not null."""
    _seed_job(job_routes, "j-1", [])
    job_routes.jobs["j-1"].status = "running"
    fake_workflows = _FakeWorkflows(
        workflows={},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: _FakeFlowMesh(fake_workflows),
    )
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_started_instance",
        classmethod(
            lambda cls: _FakeProgressServer({"execution": {"completed": True}})
        ),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/progress", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 200
    usage = resp.json()["data"]["progress"]["usage"]
    assert usage == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "calls": 0,
        "retries": 0,
        "truncated_calls": 0,
        "wall_sec": 0.0,
    }


@pytest.mark.anyio
async def test_progress_cancelled_job_keeps_persisted_usage(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled job's progress keeps the terminal usage the job runner
    persisted before cancellation, rather than returning a blank record."""
    _seed_job(job_routes, "j-1", ["wf-1"])
    record = job_routes.jobs["j-1"]
    record.status = "cancelled"
    record.progress.usage = job_routes_module.WorkflowUsage(
        prompt_tokens=44,
        completion_tokens=105,
        reasoning_tokens=97,
        calls=2,
        retries=0,
        truncated_calls=0,
        wall_sec=1.05,
    )
    job_routes._job_storage.save(record)

    def _no_fm(_request: Any) -> Any:
        raise AssertionError("flowmesh_for must not be called for a cancelled job")

    monkeypatch.setattr(job_routes_module, "flowmesh_for", _no_fm)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/progress", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 200
    usage = resp.json()["data"]["progress"]["usage"]
    assert usage == {
        "prompt_tokens": 44,
        "completion_tokens": 105,
        "reasoning_tokens": 97,
        "calls": 2,
        "retries": 0,
        "truncated_calls": 0,
        "wall_sec": 1.05,
    }


@pytest.mark.anyio
async def test_get_logs_forwards_cursor_params(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_job(job_routes, "j-1", ["wf-1"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": _FakeWorkflow("wf-1")},
        logs_result=_FakeLogQueryResponse(
            entries=[_FakeLogEntry("c1", "hello"), _FakeLogEntry("c2", "world")],
            next_cursor="c2",
        ),
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: _FakeFlowMesh(fake_workflows),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/workflows/wf-1/logs",
            params={"limit": 50, "after": "c0"},
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 200
    body = resp.json()["data"]
    assert body["job_id"] == "j-1"
    assert body["workflow_id"] == "wf-1"
    assert body["next_cursor"] == "c2"
    assert [e["event"]["message"] for e in body["entries"]] == ["hello", "world"]
    assert fake_workflows.log_calls == [("wf-1", 50, None, "c0")]


@pytest.mark.anyio
async def test_get_logs_workflow_not_in_job_returns_404(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """workflow_id not in job's trace_ids must yield 404 with the canonical message."""
    _seed_job(job_routes, "j-1", ["wf-1"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-other": _FakeWorkflow("wf-other")},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: _FakeFlowMesh(fake_workflows),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/workflows/wf-other/logs",
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 404
    assert "wf-other" in resp.json()["detail"]
    assert (
        fake_workflows.log_calls == []
    ), "get_logs must not be called for foreign workflow"


@pytest.mark.anyio
async def test_stream_logs_emits_sse_events(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_job(job_routes, "j-1", ["wf-1"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": _FakeWorkflow("wf-1")},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
        stream_entries=[
            _FakeLogEntry("c1", "first"),
            _FakeLogEntry("c2", "second"),
        ],
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: _FakeFlowMesh(fake_workflows),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/workflows/wf-1/logs/stream",
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    raw = resp.text
    data_lines = [
        line[len("data:") :].strip()
        for line in raw.splitlines()
        if line.startswith("data:")
    ]
    messages = [json.loads(d)["event"]["message"] for d in data_lines if d]
    assert messages == ["first", "second"]


@pytest.mark.anyio
async def test_stream_logs_ownership_check_before_stream(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ownership check must fire before opening the upstream stream."""
    _seed_job(job_routes, "j-1", ["wf-1"])
    fake_workflows = _FakeWorkflows(
        workflows={},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: _FakeFlowMesh(fake_workflows),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/workflows/wf-foreign/logs/stream",
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 404
    assert fake_workflows.stream_calls == []


@pytest.mark.anyio
async def test_list_workflows_unknown_job_returns_404(app: FastAPI) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/nope/workflows", headers={"Authorization": "Bearer token"}
        )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_download_logs_returns_tar_with_task_files(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_job(job_routes, "j-1", ["wf-1"])
    content_t1 = b'{"message": "task1-log"}\n'
    content_t2 = b'{"message": "task2-log"}\n'
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": _FakeWorkflow("wf-1")},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_workflows._workflows["wf-1"].task_ids = ["tsk-00000001", "tsk-00000002"]
    fake_fm = _FakeFlowMesh(
        workflows=fake_workflows,
        task_log_bytes={"tsk-00000001": content_t1, "tsk-00000002": content_t2},
    )
    monkeypatch.setattr(job_routes_module, "flowmesh_for", lambda _request: fake_fm)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/workflows/wf-1/logs/download",
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/x-tar"
    assert "wf-1-logs.tar" in resp.headers.get("content-disposition", "")

    buf = io.BytesIO(resp.content)
    with tarfile.open(fileobj=buf, mode="r") as tf:
        names = tf.getnames()
        assert sorted(names) == ["tsk-00000001-logs.jsonl", "tsk-00000002-logs.jsonl"]
        assert tf.extractfile("tsk-00000001-logs.jsonl").read() == content_t1  # type: ignore[union-attr]
        assert tf.extractfile("tsk-00000002-logs.jsonl").read() == content_t2  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_download_logs_skips_missing_task_archives(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_job(job_routes, "j-1", ["wf-1"])
    content_t2 = b'{"message": "only-task2"}\n'
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": _FakeWorkflow("wf-1")},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_workflows._workflows["wf-1"].task_ids = ["tsk-00000000", "tsk-00000002"]
    fake_fm = _FakeFlowMesh(
        workflows=fake_workflows,
        task_log_bytes={"tsk-00000002": content_t2},
    )
    monkeypatch.setattr(job_routes_module, "flowmesh_for", lambda _request: fake_fm)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/workflows/wf-1/logs/download",
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 200
    buf = io.BytesIO(resp.content)
    with tarfile.open(fileobj=buf, mode="r") as tf:
        names = tf.getnames()
        assert names == ["tsk-00000002-logs.jsonl"]


@pytest.mark.anyio
async def test_download_logs_all_archives_missing_returns_empty_tar(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_job(job_routes, "j-1", ["wf-1"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": _FakeWorkflow("wf-1")},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_workflows._workflows["wf-1"].task_ids = ["tsk-00000011", "tsk-00000012"]
    fake_fm = _FakeFlowMesh(workflows=fake_workflows, task_log_bytes={})
    monkeypatch.setattr(job_routes_module, "flowmesh_for", lambda _request: fake_fm)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/workflows/wf-1/logs/download",
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 200
    buf = io.BytesIO(resp.content)
    with tarfile.open(fileobj=buf, mode="r") as tf:
        assert tf.getmembers() == []


@pytest.mark.anyio
async def test_download_logs_ownership_check_returns_404(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_job(job_routes, "j-1", ["wf-1"])
    fake_workflows = _FakeWorkflows(
        workflows={},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_fm = _FakeFlowMesh(workflows=fake_workflows)
    monkeypatch.setattr(job_routes_module, "flowmesh_for", lambda _request: fake_fm)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/workflows/wf-foreign/logs/download",
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 404
    assert fake_fm.raw_calls == []


@pytest.mark.anyio
async def test_download_logs_api_error_returns_502(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_job(job_routes, "j-1", ["wf-1"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": _FakeWorkflow("wf-1")},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_workflows._workflows["wf-1"].task_ids = ["tsk-badbad01"]
    fake_fm = _FakeFlowMesh(
        workflows=fake_workflows,
        task_log_errors={
            "tsk-badbad01": APIError(
                "upstream failure",
                status_code=500,
                method="GET",
                url="/results/tsk-badbad01/logs",
            )
        },
    )
    monkeypatch.setattr(job_routes_module, "flowmesh_for", lambda _request: fake_fm)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/workflows/wf-1/logs/download",
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 502


@pytest.mark.anyio
async def test_stream_logs_not_found_error_emits_structured_sse_error(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NotFoundError mid-stream must emit a structured JSON error frame."""
    _seed_job(job_routes, "j-1", ["wf-1"])
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": _FakeWorkflow("wf-1")},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
        stream_error=NotFoundError(
            "workflow wf-1 not found", status_code=404, method="GET", url="/logs/stream"
        ),
    )
    monkeypatch.setattr(
        job_routes_module,
        "flowmesh_for",
        lambda _request: _FakeFlowMesh(fake_workflows),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/workflows/wf-1/logs/stream",
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 200
    raw = resp.text
    # Locate the error event block
    error_data: dict[str, Any] | None = None
    for block in raw.split("\n\n"):
        lines = block.splitlines()
        event_type = next(
            (ln[len("event:") :].strip() for ln in lines if ln.startswith("event:")),
            None,
        )
        if event_type == "error":
            data_line = next(
                (ln[len("data:") :].strip() for ln in lines if ln.startswith("data:")),
                None,
            )
            if data_line:
                error_data = json.loads(data_line)
    assert error_data is not None, "no error event frame found in SSE body"
    assert error_data["kind"] == "stream_error"
    assert error_data["code"] == "NotFoundError"
    # message must not contain raw exception repr / internal details
    assert "not found or expired" in error_data["message"]


def test_safe_tar_name_accepts_normal_ids() -> None:
    """_safe_tar_name accepts standard and uppercase task IDs."""
    assert job_routes_module._safe_tar_name("tsk-00000abc") == "tsk-00000abc-logs.jsonl"
    assert (
        job_routes_module._safe_tar_name("tsk-uppercaseXYZ")
        == "tsk-uppercaseXYZ-logs.jsonl"
    )


def test_safe_tar_name_escapes_path_separators() -> None:
    """_safe_tar_name URL-encodes path separators so distinct IDs stay distinct."""
    slash = job_routes_module._safe_tar_name("tsk-with/slash")
    assert slash == "tsk-with%2Fslash-logs.jsonl"
    # Distinct inputs must produce distinct member names (collision-free).
    underscore = job_routes_module._safe_tar_name("tsk-with_slash")
    backslash = job_routes_module._safe_tar_name("tsk-with\\slash")
    assert len({slash, underscore, backslash}) == 3


def test_safe_tar_name_preserves_whitespace_so_no_collision() -> None:
    """IDs differing only by surrounding whitespace must produce distinct names."""
    bare = job_routes_module._safe_tar_name("tsk-abc")
    padded = job_routes_module._safe_tar_name(" tsk-abc ")
    assert bare == "tsk-abc-logs.jsonl"
    assert padded == "%20tsk-abc%20-logs.jsonl"
    assert bare != padded


def test_safe_tar_name_rejects_degenerate_ids() -> None:
    """_safe_tar_name raises ValueError for IDs that resolve to parent/current/empty."""
    with pytest.raises(ValueError):
        job_routes_module._safe_tar_name("..")
    with pytest.raises(ValueError):
        job_routes_module._safe_tar_name(".")
    with pytest.raises(ValueError):
        job_routes_module._safe_tar_name("")


@pytest.mark.anyio
async def test_download_logs_invalid_task_id_skipped(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task IDs unsafe for tar members are skipped; valid uppercase IDs appear."""
    _seed_job(job_routes, "j-1", ["wf-1"])
    content_good = b'{"message": "good-task"}\n'
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": _FakeWorkflow("wf-1")},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    # Mix: one path-traversal id (reduces to ".."), one valid id with uppercase chars
    fake_workflows._workflows["wf-1"].task_ids = [
        "..",
        "tsk-upperXYZ",
    ]
    fake_fm = _FakeFlowMesh(
        workflows=fake_workflows,
        task_log_bytes={"tsk-upperXYZ": content_good},
    )
    monkeypatch.setattr(job_routes_module, "flowmesh_for", lambda _request: fake_fm)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/workflows/wf-1/logs/download",
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 200
    buf = io.BytesIO(resp.content)
    with tarfile.open(fileobj=buf, mode="r") as tf:
        names = tf.getnames()
    # Unsafe id was skipped; uppercase id is included
    assert names == ["tsk-upperXYZ-logs.jsonl"]
    # The fake was never called with the unsafe path
    called_paths = [path for _, path in fake_fm.raw_calls]
    assert not any("%2E%2E" in p or ".." in p for p in called_paths)


@pytest.mark.anyio
async def test_download_logs_spool_spills_to_disk(
    app: FastAPI, job_routes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Archive exceeding LUMILAKE_LOG_DOWNLOAD_SPOOL_MAX_MB must spill spool to disk."""
    _seed_job(job_routes, "j-1", ["wf-1"])
    # Two 1 MiB members — total tar will exceed a 1 MiB spool threshold
    member_bytes = b"x" * (1024 * 1024)
    fake_workflows = _FakeWorkflows(
        workflows={"wf-1": _FakeWorkflow("wf-1")},
        logs_result=_FakeLogQueryResponse(entries=[], next_cursor=None),
    )
    fake_workflows._workflows["wf-1"].task_ids = ["tsk-aa000001", "tsk-aa000002"]
    fake_fm = _FakeFlowMesh(
        workflows=fake_workflows,
        task_log_bytes={
            "tsk-aa000001": member_bytes,
            "tsk-aa000002": member_bytes,
        },
    )
    monkeypatch.setattr(job_routes_module, "flowmesh_for", lambda _request: fake_fm)
    # Set spool threshold to 1 MiB so the two-member tar spills to disk
    monkeypatch.setattr(job_routes_module.envs, "LUMILAKE_LOG_DOWNLOAD_SPOOL_MAX_MB", 1)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/j-1/workflows/wf-1/logs/download",
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 200
    buf = io.BytesIO(resp.content)
    with tarfile.open(fileobj=buf, mode="r") as tf:
        names = sorted(tf.getnames())
    assert names == ["tsk-aa000001-logs.jsonl", "tsk-aa000002-logs.jsonl"]
    # Verify the content round-trips correctly despite spilling
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r") as tf:
        assert tf.extractfile("tsk-aa000001-logs.jsonl").read() == member_bytes  # type: ignore[union-attr]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
