"""Tests for BaseClient + BaseAsyncClient — request shaping + error mapping.

Uses respx to mock httpx without a real server.
"""

import httpx
import pytest
import respx
from lumilake import (
    BaseAsyncClient,
    BaseClient,
    HttpError,
    NotFoundError,
    unwrap,
)


def test_get_uses_versioned_path(http: BaseClient) -> None:
    with respx.mock(base_url=http.base_url) as mocked:
        route = mocked.get("/api/v1/jobs/123").mock(
            return_value=httpx.Response(200, json={"data": {"id": "123"}})
        )
        resp = http.get("/jobs/123")
        assert route.called
        assert resp.json() == {"data": {"id": "123"}}


def test_no_version_prefix_for_healthz(http: BaseClient) -> None:
    with respx.mock(base_url=http.base_url) as mocked:
        mocked.get("/healthz").mock(return_value=httpx.Response(200, json={"ok": True}))
        resp = http.get("/healthz", version_prefix=False)
        assert resp.json() == {"ok": True}


def test_404_raises_not_found(http: BaseClient) -> None:
    with respx.mock(base_url=http.base_url) as mocked:
        mocked.get("/api/v1/jobs/missing").mock(
            return_value=httpx.Response(404, text="nope")
        )
        with pytest.raises(NotFoundError) as exc:
            http.get("/jobs/missing")
        assert exc.value.status == 404


def test_500_raises_generic_http_error(http: BaseClient) -> None:
    with respx.mock(base_url=http.base_url) as mocked:
        mocked.get("/api/v1/jobs").mock(return_value=httpx.Response(503, text="down"))
        with pytest.raises(HttpError) as exc:
            http.get("/jobs")
        assert exc.value.status == 503


def test_unwrap_handles_envelope() -> None:
    assert unwrap(httpx.Response(200, json={"data": {"id": "x"}})) == {"id": "x"}
    # also handles already-unwrapped responses
    assert unwrap(httpx.Response(200, json={"id": "x"})) == {"id": "x"}


@pytest.mark.asyncio
async def test_async_get_versioned_path(async_http: BaseAsyncClient) -> None:
    with respx.mock(base_url=async_http.base_url) as mocked:
        route = mocked.get("/api/v1/jobs/123").mock(
            return_value=httpx.Response(200, json={"data": {"id": "123"}})
        )
        resp = await async_http.get("/jobs/123")
        assert route.called
        assert resp.json() == {"data": {"id": "123"}}
        await async_http.close()


@pytest.mark.asyncio
async def test_async_404_raises_not_found(async_http: BaseAsyncClient) -> None:
    with respx.mock(base_url=async_http.base_url) as mocked:
        mocked.get("/api/v1/jobs/missing").mock(
            return_value=httpx.Response(404, text="nope")
        )
        with pytest.raises(NotFoundError):
            await async_http.get("/jobs/missing")
        await async_http.close()
