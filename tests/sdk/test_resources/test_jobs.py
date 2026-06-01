"""Tests for Jobs (sync) + AsyncJobs."""

import io
import json
import tarfile
from pathlib import Path

import httpx
import pytest
import respx
from lumilake import (
    AsyncJobs,
    BaseAsyncClient,
    BaseClient,
    Jobs,
    JobWorkflowInfo,
    LogEntry,
    LogQueryResponse,
)
from lumilake.errors import NotFoundError


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


# ---- Parity helpers ----


def test_preview_posts_with_format_header(jobs: Jobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        route = mocked.post("/api/v1/jobs/preview").mock(
            return_value=httpx.Response(200, json={"data": {"request_id": "p-1"}})
        )
        result = jobs.preview({"data": []}, workflow_format="yaml")
        assert result == {"request_id": "p-1"}
        assert route.called
        assert route.calls.last.request.headers["Workflow-Format"] == "yaml"


def test_preview_omits_format_header_by_default(jobs: Jobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        route = mocked.post("/api/v1/jobs/preview").mock(
            return_value=httpx.Response(200, json={"data": {"request_id": "p-1"}})
        )
        jobs.preview({"data": []})
        assert "Workflow-Format" not in route.calls.last.request.headers


def test_progress(jobs: Jobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j-1/progress").mock(
            return_value=httpx.Response(
                200,
                json={"data": {"job_id": "j-1", "progress": {"completed": 2}}},
            )
        )
        assert jobs.progress("j-1") == {
            "job_id": "j-1",
            "progress": {"completed": 2},
        }


def test_result(jobs: Jobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j-1/result").mock(
            return_value=httpx.Response(
                200, json={"data": {"job_id": "j-1", "result": {"ok": True}}}
            )
        )
        result = jobs.result("j-1")
        assert result["result"] == {"ok": True}


def test_inputs(jobs: Jobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j-1/inputs").mock(
            return_value=httpx.Response(
                200, json={"data": {"job_id": "j-1", "inputs": {"Stock": ["NVDA"]}}}
            )
        )
        assert jobs.inputs("j-1")["inputs"] == {"Stock": ["NVDA"]}


def test_artifact_streams_to_disk(jobs: Jobs, base_url: str, tmp_path: Path) -> None:
    target = tmp_path / "out" / "result.bin"
    with respx.mock(base_url=base_url) as mocked:
        route = mocked.get("/api/v1/jobs/j-1/artifact").mock(
            return_value=httpx.Response(200, content=b"binary-payload")
        )
        path = jobs.artifact("j-1", path="s3://bucket/foo", output=target)
        assert path == target
        assert target.read_bytes() == b"binary-payload"
        assert "path=s3" in str(route.calls.last.request.url)


def test_artifact_maps_404_to_sdk_error(
    jobs: Jobs, base_url: str, tmp_path: Path
) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/missing/artifact").mock(
            return_value=httpx.Response(404, text="not found")
        )
        with pytest.raises(NotFoundError):
            jobs.artifact("missing", path="s3://bucket/foo", output=tmp_path / "out")


def test_watch_yields_snapshots_until_terminal(jobs: Jobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j-1").mock(
            side_effect=[
                httpx.Response(200, json={"data": {"id": "j-1", "status": "pending"}}),
                httpx.Response(
                    200, json={"data": {"id": "j-1", "status": "completed"}}
                ),
            ]
        )
        mocked.get("/api/v1/jobs/j-1/progress").mock(
            return_value=httpx.Response(
                200, json={"data": {"job_id": "j-1", "progress": {"step": 1}}}
            )
        )

        snapshots = list(jobs.watch("j-1", poll_interval=0.0, timeout=2.0))

    assert [s["status"] for s in snapshots] == ["pending", "completed"]
    assert snapshots[-1]["progress"] == {"step": 1}


def test_list_all_traverses_cursor(jobs: Jobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "data": {
                            "items": [{"id": "a"}, {"id": "b"}],
                            "next_cursor": "abc",
                        }
                    },
                ),
                httpx.Response(
                    200,
                    json={"data": {"items": [{"id": "c"}], "next_cursor": None}},
                ),
            ]
        )
        results = list(jobs.list_all(page_size=2))
        assert [r["id"] for r in results] == ["a", "b", "c"]


def test_list_all_stops_with_no_cursor(jobs: Jobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs").mock(
            return_value=httpx.Response(200, json={"data": {"items": [{"id": "a"}]}})
        )
        results = list(jobs.list_all())
        assert results == [{"id": "a"}]


def test_list_all_raises_on_replayed_cursor(jobs: Jobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "data": {
                            "items": [{"id": "a"}],
                            "next_cursor": "stuck",
                        }
                    },
                ),
                httpx.Response(
                    200,
                    json={
                        "data": {
                            "items": [{"id": "b"}],
                            "next_cursor": "stuck",
                        }
                    },
                ),
            ]
        )
        with pytest.raises(RuntimeError, match="replayed cursor"):
            list(jobs.list_all(page_size=1))


def test_watch_propagates_non_http_progress_error(
    jobs: Jobs, base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ValueError("non-http boom")

    monkeypatch.setattr(jobs, "progress", _boom)
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j-1").mock(
            return_value=httpx.Response(
                200, json={"data": {"id": "j-1", "status": "pending"}}
            )
        )
        with pytest.raises(ValueError, match="non-http boom"):
            list(jobs.watch("j-1", poll_interval=0.0, timeout=2.0))


def test_get_respects_per_call_timeout(jobs: Jobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        route = mocked.get("/api/v1/jobs/j-1").mock(
            return_value=httpx.Response(200, json={"data": {"id": "j-1"}})
        )
        assert jobs.get("j-1", timeout=42.0) == {"id": "j-1"}
        assert route.called


# ---- Async parity ----


@pytest.mark.asyncio
async def test_async_preview(async_jobs: AsyncJobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.post("/api/v1/jobs/preview").mock(
            return_value=httpx.Response(200, json={"data": {"request_id": "p-2"}})
        )
        result = await async_jobs.preview({"data": []}, workflow_format="native")
        assert result == {"request_id": "p-2"}
        await async_jobs._client.close()


@pytest.mark.asyncio
async def test_async_progress(async_jobs: AsyncJobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j-1/progress").mock(
            return_value=httpx.Response(
                200, json={"data": {"job_id": "j-1", "progress": {"k": 1}}}
            )
        )
        result = await async_jobs.progress("j-1")
        assert result["progress"] == {"k": 1}
        await async_jobs._client.close()


@pytest.mark.asyncio
async def test_async_list_all(async_jobs: AsyncJobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "data": {
                            "items": [{"id": "a"}],
                            "next_cursor": "next",
                        }
                    },
                ),
                httpx.Response(
                    200, json={"data": {"items": [{"id": "b"}], "next_cursor": None}}
                ),
            ]
        )
        collected: list[dict[str, str]] = []
        async for item in async_jobs.list_all():
            collected.append(item)
        assert [c["id"] for c in collected] == ["a", "b"]
        await async_jobs._client.close()


@pytest.mark.asyncio
async def test_async_watch(async_jobs: AsyncJobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j-1").mock(
            side_effect=[
                httpx.Response(
                    200, json={"data": {"id": "j-1", "status": "completed"}}
                ),
            ]
        )
        mocked.get("/api/v1/jobs/j-1/progress").mock(
            return_value=httpx.Response(200, json={"data": {"progress": {"x": 1}}})
        )
        snapshots: list[dict[str, object]] = []
        async for snap in async_jobs.watch("j-1", poll_interval=0.0, timeout=2.0):
            snapshots.append(snap)
        assert [s["status"] for s in snapshots] == ["completed"]
        await async_jobs._client.close()


@pytest.mark.asyncio
async def test_async_list_all_raises_on_replayed_cursor(
    async_jobs: AsyncJobs, base_url: str
) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "data": {
                            "items": [{"id": "a"}],
                            "next_cursor": "stuck",
                        }
                    },
                ),
                httpx.Response(
                    200,
                    json={
                        "data": {
                            "items": [{"id": "b"}],
                            "next_cursor": "stuck",
                        }
                    },
                ),
            ]
        )
        with pytest.raises(RuntimeError, match="replayed cursor"):
            collected: list[dict[str, object]] = []
            async for item in async_jobs.list_all(page_size=1):
                collected.append(item)
        await async_jobs._client.close()


@pytest.mark.asyncio
async def test_async_watch_propagates_non_http_progress_error(
    async_jobs: AsyncJobs, base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _boom(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ValueError("non-http boom")

    monkeypatch.setattr(async_jobs, "progress", _boom)
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j-1").mock(
            return_value=httpx.Response(
                200, json={"data": {"id": "j-1", "status": "pending"}}
            )
        )
        with pytest.raises(ValueError, match="non-http boom"):
            async for _ in async_jobs.watch("j-1", poll_interval=0.0, timeout=2.0):
                pass
        await async_jobs._client.close()


@pytest.mark.asyncio
async def test_async_artifact(
    async_jobs: AsyncJobs, base_url: str, tmp_path: Path
) -> None:
    target = tmp_path / "out.bin"
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j-1/artifact").mock(
            return_value=httpx.Response(200, content=b"abc")
        )
        path = await async_jobs.artifact("j-1", path="s3://x", output=target)
        assert path == target
        assert target.read_bytes() == b"abc"
        await async_jobs._client.close()


@pytest.mark.asyncio
async def test_async_artifact_maps_404_to_sdk_error(
    async_jobs: AsyncJobs, base_url: str, tmp_path: Path
) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/missing/artifact").mock(
            return_value=httpx.Response(404, text="not found")
        )
        with pytest.raises(NotFoundError):
            await async_jobs.artifact(
                "missing", path="s3://bucket/foo", output=tmp_path / "out"
            )
        await async_jobs._client.close()


# ---- Workflows + logs ----


def test_list_workflows_returns_typed_models(jobs: Jobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j-1/workflows").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": {
                        "job_id": "j-1",
                        "workflows": [
                            {
                                "workflow_id": "wf-1",
                                "status": "COMPLETED",
                                "submitted_at": "2026-05-31T00:00:00Z",
                                "task_count": 5,
                                "succeeded_count": 4,
                                "failed_count": 1,
                            },
                            {
                                "workflow_id": "wf-2",
                                "status": "RUNNING",
                                "submitted_at": None,
                                "task_count": None,
                                "succeeded_count": None,
                                "failed_count": None,
                            },
                        ],
                    }
                },
            )
        )
        result = jobs.list_workflows("j-1")
    assert len(result) == 2
    assert all(isinstance(w, JobWorkflowInfo) for w in result)
    assert result[0].workflow_id == "wf-1"
    assert result[0].status == "COMPLETED"
    assert result[0].task_count == 5
    assert result[1].workflow_id == "wf-2"


def test_list_workflows_empty_when_no_workflows_key(jobs: Jobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j-1/workflows").mock(
            return_value=httpx.Response(200, json={"data": {"job_id": "j-1"}})
        )
        assert jobs.list_workflows("j-1") == []


def test_get_logs_returns_typed_model(jobs: Jobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        route = mocked.get("/api/v1/jobs/j-1/workflows/wf-1/logs").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": {
                        "job_id": "j-1",
                        "workflow_id": "wf-1",
                        "entries": [
                            {
                                "cursor": "c1",
                                "event": {
                                    "ts": "2026-05-31T00:00:00Z",
                                    "level": "INFO",
                                    "stream": "stdout",
                                    "message": "hello",
                                },
                            }
                        ],
                        "next_cursor": "c1",
                        "prev_cursor": None,
                    }
                },
            )
        )
        result = jobs.get_logs("j-1", "wf-1", limit=50, after="c0")
    assert isinstance(result, LogQueryResponse)
    assert result.workflow_id == "wf-1"
    assert result.entries[0].event.message == "hello"
    assert result.next_cursor == "c1"
    url = str(route.calls.last.request.url)
    assert "limit=50" in url and "after=c0" in url


def test_stream_logs_yields_entries(jobs: Jobs, base_url: str) -> None:
    def _event(msg: str) -> dict[str, object]:
        return {
            "message": msg,
            "ts": None,
            "workflow_id": None,
            "task_id": None,
            "worker_id": None,
            "node_id": None,
            "level": "INFO",
            "stream": None,
            "source": None,
            "fields": None,
        }

    entry1 = {"cursor": "c1", "event": _event("one")}
    entry2 = {"cursor": "c2", "event": _event("two")}
    sse = f"data: {json.dumps(entry1)}\n\ndata: {json.dumps(entry2)}\n\n"
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j-1/workflows/wf-1/logs/stream").mock(
            return_value=httpx.Response(
                200,
                text=sse,
                headers={"content-type": "text/event-stream"},
            )
        )
        result = list(jobs.stream_logs("j-1", "wf-1"))
    assert len(result) == 2
    assert all(isinstance(e, LogEntry) for e in result)
    assert result[0].event.message == "one"
    assert result[1].event.message == "two"


@pytest.mark.asyncio
async def test_async_list_workflows(async_jobs: AsyncJobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j-1/workflows").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": {
                        "job_id": "j-1",
                        "workflows": [
                            {
                                "workflow_id": "wf-1",
                                "status": "COMPLETED",
                                "submitted_at": None,
                                "task_count": None,
                                "succeeded_count": None,
                                "failed_count": None,
                            }
                        ],
                    }
                },
            )
        )
        result = await async_jobs.list_workflows("j-1")
    assert len(result) == 1
    assert isinstance(result[0], JobWorkflowInfo)
    assert result[0].workflow_id == "wf-1"
    await async_jobs._client.close()


@pytest.mark.asyncio
async def test_async_get_logs(async_jobs: AsyncJobs, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j-1/workflows/wf-1/logs").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": {
                        "job_id": "j-1",
                        "workflow_id": "wf-1",
                        "entries": [],
                        "next_cursor": None,
                        "prev_cursor": None,
                    }
                },
            )
        )
        result = await async_jobs.get_logs("j-1", "wf-1")
    assert isinstance(result, LogQueryResponse)
    assert result.workflow_id == "wf-1"
    assert result.entries == []
    await async_jobs._client.close()


# ---- download_logs ----


def _make_tar_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def test_download_logs_extracts_files(
    jobs: Jobs, base_url: str, tmp_path: Path
) -> None:
    content_t1 = b'{"message": "log1"}\n'
    content_t2 = b'{"message": "log2"}\n'
    tar_bytes = _make_tar_bytes(
        {"t-1-logs.jsonl": content_t1, "t-2-logs.jsonl": content_t2}
    )
    paths: list[Path]
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j-1/workflows/wf-1/logs/download").mock(
            return_value=httpx.Response(
                200,
                content=tar_bytes,
                headers={"content-type": "application/x-tar"},
            )
        )
        paths = jobs.download_logs("j-1", "wf-1", tmp_path)

    assert sorted(p.name for p in paths) == ["t-1-logs.jsonl", "t-2-logs.jsonl"]
    assert (tmp_path / "t-1-logs.jsonl").read_bytes() == content_t1
    assert (tmp_path / "t-2-logs.jsonl").read_bytes() == content_t2


def test_download_logs_empty_tar_returns_empty_list(
    jobs: Jobs, base_url: str, tmp_path: Path
) -> None:
    tar_bytes = _make_tar_bytes({})
    paths: list[Path]
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j-1/workflows/wf-1/logs/download").mock(
            return_value=httpx.Response(
                200,
                content=tar_bytes,
                headers={"content-type": "application/x-tar"},
            )
        )
        paths = jobs.download_logs("j-1", "wf-1", tmp_path)
    assert paths == []


def test_download_logs_404_raises_not_found(
    jobs: Jobs, base_url: str, tmp_path: Path
) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j-1/workflows/wf-missing/logs/download").mock(
            return_value=httpx.Response(404, text="not found")
        )
        with pytest.raises(NotFoundError):
            jobs.download_logs("j-1", "wf-missing", tmp_path)


@pytest.mark.asyncio
async def test_async_download_logs_extracts_files(
    async_jobs: AsyncJobs, base_url: str, tmp_path: Path
) -> None:
    content = b'{"message": "async-log"}\n'
    tar_bytes = _make_tar_bytes({"t-a-logs.jsonl": content})
    paths: list[Path]
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/api/v1/jobs/j-1/workflows/wf-1/logs/download").mock(
            return_value=httpx.Response(
                200,
                content=tar_bytes,
                headers={"content-type": "application/x-tar"},
            )
        )
        paths = await async_jobs.download_logs("j-1", "wf-1", tmp_path)
    assert len(paths) == 1
    assert paths[0].name == "t-a-logs.jsonl"
    assert paths[0].read_bytes() == content
    await async_jobs._client.close()
