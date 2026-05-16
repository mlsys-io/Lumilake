"""Tests for Traces (sync) + AsyncTraces."""

import httpx
import pytest
import respx
from lumilake import AsyncTraces, BaseAsyncClient, BaseClient, Traces


@pytest.fixture
def traces(http: BaseClient) -> Traces:
    return Traces(http)


@pytest.fixture
def async_traces(async_http: BaseAsyncClient) -> AsyncTraces:
    return AsyncTraces(async_http)


def test_list_all_raises_on_replayed_cursor(traces: Traces, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/traces").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "data": {
                            "items": [{"id": "t-0"}],
                            "next_cursor": "stuck",
                        }
                    },
                ),
                httpx.Response(
                    200,
                    json={
                        "data": {
                            "items": [{"id": "t-1"}],
                            "next_cursor": "stuck",
                        }
                    },
                ),
            ]
        )
        with pytest.raises(RuntimeError, match="replayed cursor"):
            list(traces.list_all(page_size=1))


@pytest.mark.asyncio
async def test_async_list_all_raises_on_replayed_cursor(
    async_traces: AsyncTraces, base_url: str
) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/traces").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "data": {
                            "items": [{"id": "t-0"}],
                            "next_cursor": "stuck",
                        }
                    },
                ),
                httpx.Response(
                    200,
                    json={
                        "data": {
                            "items": [{"id": "t-1"}],
                            "next_cursor": "stuck",
                        }
                    },
                ),
            ]
        )
        with pytest.raises(RuntimeError, match="replayed cursor"):
            async for _ in async_traces.list_all(page_size=1):
                pass
        await async_traces._client.close()
