import logging
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from lumilake_server.routes import workers as workers_routes
from lumilake_server.runtime import flowmesh_client


class _FakeWorkersResource:
    async def list(self, **_kwargs: Any) -> list[Any]:
        return []


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
