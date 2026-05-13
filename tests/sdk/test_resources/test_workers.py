"""Tests for Workers (sync) + AsyncWorkers."""

import httpx
import pytest
import respx

from lumilake.sdk import AsyncWorkers, BaseAsyncClient, BaseClient, Workers


@pytest.fixture
def workers(http: BaseClient) -> Workers:
    return Workers(http)


@pytest.fixture
def async_workers(async_http: BaseAsyncClient) -> AsyncWorkers:
    return AsyncWorkers(async_http)


def test_list_with_filters(workers: Workers, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        route = mocked.get("/api/v1/workers").mock(
            return_value=httpx.Response(200, json={"data": {"items": [{"id": "w-0"}]}})
        )
        assert workers.list(status="IDLE", tag="cpu") == [{"id": "w-0"}]
        url = str(route.calls.last.request.url)
        assert "status=IDLE" in url and "tag=cpu" in url


def test_get(workers: Workers, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/workers/w-1").mock(
            return_value=httpx.Response(200, json={"data": {"id": "w-1"}})
        )
        assert workers.get("w-1") == {"id": "w-1"}


@pytest.mark.asyncio
async def test_async_list(async_workers: AsyncWorkers, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/workers").mock(
            return_value=httpx.Response(200, json={"data": {"items": [{"id": "w-0"}]}})
        )
        assert await async_workers.list() == [{"id": "w-0"}]
        await async_workers._client.close()


@pytest.mark.asyncio
async def test_async_get(async_workers: AsyncWorkers, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/workers/w-1").mock(
            return_value=httpx.Response(200, json={"data": {"id": "w-1"}})
        )
        assert await async_workers.get("w-1") == {"id": "w-1"}
        await async_workers._client.close()
