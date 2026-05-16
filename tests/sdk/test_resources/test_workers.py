"""Tests for Workers (sync) + AsyncWorkers."""

import httpx
import pytest
import respx
from lumilake import AsyncWorkers, BaseAsyncClient, BaseClient, Workers


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


def test_list_all_traverses_cursor(workers: Workers, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/workers").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "data": {
                            "items": [{"id": "w-0"}, {"id": "w-1"}],
                            "next_cursor": "next",
                        }
                    },
                ),
                httpx.Response(
                    200,
                    json={"data": {"items": [{"id": "w-2"}], "next_cursor": None}},
                ),
            ]
        )
        results = list(workers.list_all(page_size=2))
        assert [r["id"] for r in results] == ["w-0", "w-1", "w-2"]


def test_list_all_raises_on_replayed_cursor(workers: Workers, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/workers").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "data": {
                            "items": [{"id": "w-0"}],
                            "next_cursor": "stuck",
                        }
                    },
                ),
                httpx.Response(
                    200,
                    json={
                        "data": {
                            "items": [{"id": "w-1"}],
                            "next_cursor": "stuck",
                        }
                    },
                ),
            ]
        )
        with pytest.raises(RuntimeError, match="replayed cursor"):
            list(workers.list_all(page_size=1))


@pytest.mark.asyncio
async def test_async_list_all_raises_on_replayed_cursor(
    async_workers: AsyncWorkers, base_url: str
) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/workers").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "data": {
                            "items": [{"id": "w-0"}],
                            "next_cursor": "stuck",
                        }
                    },
                ),
                httpx.Response(
                    200,
                    json={
                        "data": {
                            "items": [{"id": "w-1"}],
                            "next_cursor": "stuck",
                        }
                    },
                ),
            ]
        )
        with pytest.raises(RuntimeError, match="replayed cursor"):
            async for _ in async_workers.list_all(page_size=1):
                pass
        await async_workers._client.close()


@pytest.mark.asyncio
async def test_async_list_all(async_workers: AsyncWorkers, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/workers").mock(
            return_value=httpx.Response(200, json={"data": {"items": [{"id": "w-0"}]}})
        )
        collected: list[dict[str, str]] = []
        async for item in async_workers.list_all():
            collected.append(item)
        assert collected == [{"id": "w-0"}]
        await async_workers._client.close()
