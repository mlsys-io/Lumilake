import logging
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from flowmesh.exceptions import NotFoundError
from flowmesh.models.workers import WorkerInfo

from lumilake_server.routes import workers as workers_routes
from lumilake_server.runtime import flowmesh_client
from lumilake_server.runtime.server import LumilakeServer


class _FakeWorkersResource:
    def __init__(self, workers: list[Any] | None = None) -> None:
        self._workers = workers or []

    async def list(self, **_kwargs: Any) -> list[Any]:
        return self._workers

    async def retrieve(self, worker_id: str) -> Any:
        for worker in self._workers:
            if worker.id == worker_id:
                return worker
        raise NotFoundError("not found")


class _RecordingFlowMesh:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        http_client: Any,
    ) -> None:
        self.api_key = api_key
        self.workers = _FakeWorkersResource()


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    captured: list[str | None] = []

    def factory(*, base_url: str, api_key: str | None, http_client: Any) -> Any:
        captured.append(api_key)
        return _RecordingFlowMesh(
            base_url=base_url, api_key=api_key, http_client=http_client
        )

    monkeypatch.setattr(flowmesh_client, "AsyncFlowMesh", factory)
    return captured


@pytest.fixture
def app() -> FastAPI:
    application = FastAPI()
    application.state.logger = logging.getLogger("test.runtime_token_routing")
    application.include_router(workers_routes.router)
    return application


@pytest.mark.asyncio
async def test_bearer_is_forwarded_to_flowmesh(
    app: FastAPI, recorder: list[str | None]
) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/workers",
            headers={"Authorization": "Bearer abc123"},
        )
    assert response.status_code == 200
    assert recorder == ["abc123"]


@pytest.mark.asyncio
async def test_missing_bearer_forwards_no_api_key(
    app: FastAPI, recorder: list[str | None]
) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/workers")
    assert response.status_code == 200
    assert recorder == [None]


@pytest.mark.asyncio
async def test_bearer_parsing_normalizes_case_and_whitespace(
    app: FastAPI, recorder: list[str | None]
) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for header in ("Bearer  abc123", "bearer abc123", "BEARER\tabc123"):
            recorder.clear()
            response = await client.get("/workers", headers={"Authorization": header})
            assert response.status_code == 200, header
            assert recorder == ["abc123"], header


@pytest.mark.asyncio
async def test_non_bearer_scheme_yields_no_api_key(
    app: FastAPI, recorder: list[str | None]
) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/workers", headers={"Authorization": "Basic dXNlcjpwYXNz"}
        )
    assert response.status_code == 200
    assert recorder == [None]


def _make_worker(worker_id: str) -> WorkerInfo:
    return WorkerInfo(
        id=worker_id,
        namespace="default",
        cluster="local",
        node_id=worker_id,
        node_alias=worker_id,
        status="online",
    )


@pytest.fixture
def busy_server(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """Patch LumilakeServer.get_instance to a stub tracking busy workers."""
    busy: set[str] = set()
    stub = SimpleNamespace(_busy_workers=busy)
    monkeypatch.setattr(LumilakeServer, "get_instance", lambda: stub)
    return busy


@pytest.fixture
def worker_app(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FastAPI, _FakeWorkersResource]:
    """App whose flowmesh returns one worker, plus the fake resource handle."""
    application = FastAPI()
    application.state.logger = logging.getLogger("test.worker_busy")
    application.include_router(workers_routes.router)
    resource = _FakeWorkersResource([_make_worker("worker-1")])

    def factory(*, base_url: str, api_key: str | None, http_client: Any) -> Any:
        return _RecordingFlowMesh(
            base_url=base_url, api_key=api_key, http_client=http_client
        )

    monkeypatch.setattr(flowmesh_client, "AsyncFlowMesh", factory)
    # Point the recording flowmesh's workers resource at our fake.
    original = _RecordingFlowMesh.__init__

    def _init(self, *, base_url: str, api_key: str | None, http_client: Any) -> None:
        original(self, base_url=base_url, api_key=api_key, http_client=http_client)
        self.workers = resource

    monkeypatch.setattr(_RecordingFlowMesh, "__init__", _init)
    return application, resource


@pytest.mark.asyncio
async def test_worker_busy_state_reflects_claimed_workers(
    worker_app: tuple[FastAPI, _FakeWorkersResource], busy_server: set[str]
) -> None:
    """A worker claimed by a dispatched batch reports busy: true; released, it
    reports busy: false."""
    app, _resource = worker_app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Idle: not busy.
        resp = await client.get("/workers", headers={"Authorization": "Bearer abc"})
        assert resp.status_code == 200, resp.text
        assert resp.json()[0]["busy"] is False

        # Claimed: busy.
        busy_server.add("worker-1")
        resp = await client.get("/workers", headers={"Authorization": "Bearer abc"})
        assert resp.status_code == 200, resp.text
        assert resp.json()[0]["busy"] is True

        # Released: not busy again.
        busy_server.discard("worker-1")
        resp = await client.get("/workers", headers={"Authorization": "Bearer abc"})
        assert resp.status_code == 200, resp.text
        assert resp.json()[0]["busy"] is False

        # Single-worker fetch reflects busy state too.
        busy_server.add("worker-1")
        resp = await client.get(
            "/workers/worker-1", headers={"Authorization": "Bearer abc"}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["busy"] is True
