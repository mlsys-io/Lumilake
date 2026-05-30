"""SDK Authorization-header handling on the base client."""

from collections.abc import Iterator

import httpx
import pytest
import respx
from lumilake import AsyncLumilakeClient, LumilakeClient
from lumilake._base_client import BaseClient


@pytest.fixture(autouse=True)
def _scrub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LUMILAKE_API_KEY", raising=False)
    monkeypatch.delenv("LUMILAKE_BASE_URL", raising=False)


@pytest.fixture
def base_url() -> str:
    return "http://lumilake.test"


@pytest.fixture
def http_with_key(base_url: str) -> Iterator[BaseClient]:
    client = BaseClient(base_url=base_url, api_key="secret-key")
    try:
        yield client
    finally:
        client.close()


def test_base_client_attaches_bearer_header(
    http_with_key: BaseClient, base_url: str
) -> None:
    with respx.mock(base_url=base_url) as mocked:
        route = mocked.get("/api/v1/healthz").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        http_with_key.get("/healthz")
        assert route.calls.last.request.headers["Authorization"] == "Bearer secret-key"


def test_base_client_omits_header_without_key(base_url: str) -> None:
    client = BaseClient(base_url=base_url)
    try:
        with respx.mock(base_url=base_url) as mocked:
            route = mocked.get("/api/v1/healthz").mock(
                return_value=httpx.Response(200, json={"ok": True})
            )
            client.get("/healthz")
            assert "Authorization" not in route.calls.last.request.headers
    finally:
        client.close()


def test_sync_client_reads_env_api_key(
    monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    monkeypatch.setenv("LUMILAKE_API_KEY", "env-bearer")
    client = LumilakeClient(base_url=base_url)
    try:
        with respx.mock(base_url=base_url) as mocked:
            route = mocked.get("/healthz").mock(
                return_value=httpx.Response(
                    200, json={"ok": True, "service": "lumilake-server"}
                )
            )
            client.health()
            assert (
                route.calls.last.request.headers["Authorization"] == "Bearer env-bearer"
            )
    finally:
        client.close()


def test_sync_client_kwarg_wins_over_env(
    monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    monkeypatch.setenv("LUMILAKE_API_KEY", "env-bearer")
    client = LumilakeClient(base_url=base_url, api_key="explicit-bearer")
    try:
        with respx.mock(base_url=base_url) as mocked:
            route = mocked.get("/healthz").mock(
                return_value=httpx.Response(
                    200, json={"ok": True, "service": "lumilake-server"}
                )
            )
            client.health()
            assert (
                route.calls.last.request.headers["Authorization"]
                == "Bearer explicit-bearer"
            )
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_client_attaches_bearer(
    monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    monkeypatch.setenv("LUMILAKE_API_KEY", "async-bearer")
    client = AsyncLumilakeClient(base_url=base_url)
    try:
        with respx.mock(base_url=base_url) as mocked:
            route = mocked.get("/healthz").mock(
                return_value=httpx.Response(
                    200, json={"ok": True, "service": "lumilake-server"}
                )
            )
            await client.health()
            assert (
                route.calls.last.request.headers["Authorization"]
                == "Bearer async-bearer"
            )
    finally:
        await client.close()
