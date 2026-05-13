"""Tests for Jobs (sync) + AsyncJobs."""

import httpx
import pytest
import respx

from lumilake.sdk import AsyncJobs, BaseAsyncClient, BaseClient, Jobs


@pytest.fixture
def jobs(http: BaseClient) -> Jobs:
    return Jobs(http)


@pytest.fixture
def async_jobs(async_http: BaseAsyncClient) -> AsyncJobs:
    return AsyncJobs(async_http)


def test_submit(jobs: Jobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        route = mocked.post("/api/v1/jobs").mock(
            return_value=httpx.Response(
                200, json={"data": {"id": "j-1", "status": "pending"}}
            )
        )
        result = jobs.submit({"workflow": {"x": 1}})
        assert route.called
        assert result == {"id": "j-1", "status": "pending"}
        body = route.calls.last.request.read()
        assert b'"workflow"' in body


def test_list_extracts_items(jobs: Jobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs").mock(
            return_value=httpx.Response(
                200, json={"data": {"items": [{"id": "a"}, {"id": "b"}]}}
            )
        )
        assert jobs.list() == [{"id": "a"}, {"id": "b"}]


def test_list_with_filters(jobs: Jobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        route = mocked.get("/api/v1/jobs").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        jobs.list(status="completed", limit=10)
        url = str(route.calls.last.request.url)
        assert "status=completed" in url and "limit=10" in url


def test_get(jobs: Jobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j-1").mock(
            return_value=httpx.Response(200, json={"data": {"id": "j-1"}})
        )
        assert jobs.get("j-1") == {"id": "j-1"}


def test_cancel(jobs: Jobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        route = mocked.post("/api/v1/jobs/j-1/cancel").mock(
            return_value=httpx.Response(200, json={"data": {"status": "cancelled"}})
        )
        assert jobs.cancel("j-1") == {"status": "cancelled"}
        assert route.called


def test_wait_terminal(jobs: Jobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j").mock(
            side_effect=[
                httpx.Response(200, json={"data": {"id": "j", "status": "pending"}}),
                httpx.Response(200, json={"data": {"id": "j", "status": "completed"}}),
            ]
        )
        result = jobs.wait("j", poll_interval=0.0, timeout=2.0)
        assert result["status"] == "completed"


def test_wait_timeout(jobs: Jobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j").mock(
            return_value=httpx.Response(
                200, json={"data": {"id": "j", "status": "pending"}}
            )
        )
        with pytest.raises(TimeoutError, match="still in status"):
            jobs.wait("j", poll_interval=0.0, timeout=0.05)


@pytest.mark.asyncio
async def test_async_submit(async_jobs: AsyncJobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.post("/api/v1/jobs").mock(
            return_value=httpx.Response(
                200, json={"data": {"id": "j-1", "status": "pending"}}
            )
        )
        result = await async_jobs.submit({"workflow": {"x": 1}})
        assert result == {"id": "j-1", "status": "pending"}
        await async_jobs._client.close()


@pytest.mark.asyncio
async def test_async_list_extracts_items(async_jobs: AsyncJobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs").mock(
            return_value=httpx.Response(
                200, json={"data": {"items": [{"id": "a"}, {"id": "b"}]}}
            )
        )
        assert await async_jobs.list() == [{"id": "a"}, {"id": "b"}]
        await async_jobs._client.close()


@pytest.mark.asyncio
async def test_async_wait_terminal(async_jobs: AsyncJobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j").mock(
            side_effect=[
                httpx.Response(200, json={"data": {"id": "j", "status": "pending"}}),
                httpx.Response(200, json={"data": {"id": "j", "status": "completed"}}),
            ]
        )
        result = await async_jobs.wait("j", poll_interval=0.0, timeout=2.0)
        assert result["status"] == "completed"
        await async_jobs._client.close()
