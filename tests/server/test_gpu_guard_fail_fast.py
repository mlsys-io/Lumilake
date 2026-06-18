"""Synchronous fail-fast for hardware.gpu=0 against a GPU workflow.

The guard must reject at submit/preview time with HTTP 422, before the
optimizer runs and before a job_id slot is allocated. A queued job that
fails seconds later is the bug the guard exists to prevent.
"""

import logging
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from lumid_hooks import PrincipalContext, ResourceRef

import lumilake_server.utils.job_storage as job_storage_module
from lumilake_server import hooks
from lumilake_server.middleware import TraceIdMiddleware
from lumilake_server.routes import jobs as job_routes_module
from lumilake_server.runtime.protocol import LumilakeResponse
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


class _PreviewResult:
    def __init__(self) -> None:
        self.request_id = "preview-x"
        self.selected_workers: list[Any] = []
        self.schedule = type("_S", (), {"worker_assignment": {}})()
        self.runtime_graph_node_counts: dict[str, int] = {}
        self.merged_runtime_node_count = 0
        self.selection_seconds = 0.0
        self.clustering_seconds = 0.0
        self.optimization_seconds = 0.0


class _FakeRuntimeServer:
    is_started = True

    def parse_query(self, graph_specs: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return graph_specs

    async def execute(
        self,
        graphs: dict[str, Any],
        request_id: str | None = None,
        config: Any | None = None,
        workflow_slices: dict[str, Any] | None = None,
    ) -> LumilakeResponse:
        return LumilakeResponse(outputs={})

    def trace_ids_for_request(self, job_id: str) -> list[str]:
        return []

    def optimization_seconds_for_request(self, job_id: str) -> float:
        return 0.0

    def release_request_workflows(self, job_id: str) -> None:
        return None

    async def cancel_request(self, job_id: str) -> None:
        pass

    def selection_seconds_for_request(self, job_id: str) -> float:
        return 0.0

    def clustering_seconds_for_request(self, job_id: str) -> float:
        return 0.0

    async def get_request_status(self, job_id: str) -> dict[str, Any]:
        return {}

    async def preview_schedule(self, **kwargs: Any) -> Any:
        return _PreviewResult()


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
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    storage = InMemoryJobStorage()
    job_storage_module._job_storage = storage
    job_routes_module.jobs.clear()
    job_routes_module._job_storage = storage
    fake_server = _FakeRuntimeServer()
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_started_instance",
        classmethod(lambda cls: fake_server),
    )
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_instance",
        classmethod(lambda cls: fake_server),
    )
    monkeypatch.setattr(
        job_routes_module, "build_request_data_profile_tasks", lambda **_: []
    )
    application = FastAPI()
    application.state.logger = logging.getLogger("test.gpu_guard")
    application.state.background_tasks = set()
    application.add_middleware(TraceIdMiddleware)
    application.include_router(job_routes_module.router)
    return application


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_submit_gpu_zero_against_gpu_workflow_fails_fast_with_422(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /jobs with hardware.gpu=0 against a GPU workflow must return 422
    synchronously — not a 200 with a job that later fails."""
    monkeypatch.setattr(
        job_routes_module,
        "_any_graph_requires_gpu",
        lambda server, graphs: True,
    )
    body = {
        "data": [
            {
                "name": "g",
                "workflow": "{}",
                "inputs": {"x": ["hello"]},
                "output_location": {"type": "s3", "prefix": "out/"},
            }
        ],
        "hardware": {"gpu": 0},
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=body,
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 422
    assert "hardware.gpu=0" in resp.json()["detail"]
    # No job slot allocated.
    assert job_routes_module.jobs == {}


@pytest.mark.anyio
async def test_preview_gpu_zero_against_gpu_workflow_fails_fast_with_422(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /jobs/preview with hardware.gpu=0 against a GPU workflow must
    return 422 synchronously before the optimizer runs."""
    monkeypatch.setattr(
        job_routes_module,
        "_any_graph_requires_gpu",
        lambda server, graphs: True,
    )
    body = {
        "data": [
            {
                "name": "g",
                "workflow": "{}",
                "inputs": {"x": ["hello"]},
            }
        ],
        "hardware": {"gpu": 0},
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs/preview",
            json=body,
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 422
    assert "hardware.gpu=0" in resp.json()["detail"]


@pytest.mark.anyio
async def test_submit_gpu_zero_against_cpu_only_workflow_is_accepted(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """gpu=0 must not fail-fast when the workflow is CPU-only."""
    monkeypatch.setattr(
        job_routes_module,
        "_any_graph_requires_gpu",
        lambda server, graphs: False,
    )

    # Stub _run_job so the submit handler does not spawn a real background
    # task that would hit the (un-stubbed) optimizer + FlowMesh stack and
    # hang the test process. Other tests in this file go through /preview,
    # which has no background task.
    async def _fake_run_job(*_a: Any, **_kw: Any) -> None:
        return None

    monkeypatch.setattr(job_routes_module, "_run_job", _fake_run_job)

    body = {
        "data": [
            {
                "name": "g",
                "workflow": "{}",
                "inputs": {"x": ["hello"]},
                "output_location": {"type": "s3", "prefix": "out/"},
            }
        ],
        "hardware": {"gpu": 0},
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=body,
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 200
    assert resp.json()["data"]["status"] == "pending"

    # Drain background tasks the submit handler may have scheduled so the
    # test doesn't leak a Task that races with pytest teardown.
    for task in list(app.state.background_tasks):
        await task
