"""Optimization jobs backed by the server's ``/api/v1/jobs`` routes."""

import asyncio
import json
import logging
import tarfile
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

from lumilake._base_client import _raise_for_status, unwrap
from lumilake.errors import HttpError, LogStreamError
from lumilake.resources._base import AsyncResource, SyncResource
from lumilake.resources._log_models import JobWorkflowInfo, LogEntry, LogQueryResponse

logger = logging.getLogger(__name__)

_list = list  # Alias so method names named ``list`` don't shadow the builtin.

TERMINAL_STATES: tuple[str, ...] = ("completed", "failed", "cancelled")


def _items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and "items" in data:
        return list(data["items"])
    return list(data) if isinstance(data, list) else []


def _next_cursor(data: Any) -> str | None:
    if isinstance(data, dict):
        cursor = data.get("next_cursor")
        if isinstance(cursor, str) and cursor:
            return cursor
    return None


def _list_params(
    status: str | None,
    limit: int | None,
    cursor: str | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if status is not None:
        params["status"] = status
    if limit is not None:
        params["limit"] = limit
    if cursor is not None:
        params["cursor"] = cursor
    return params


def _workflow_headers(workflow_format: str | None) -> Mapping[str, str] | None:
    return {"Workflow-Format": workflow_format} if workflow_format else None


def _request_kwargs(timeout: float | None) -> dict[str, Any]:
    if timeout is None:
        return {}
    return {"timeout": timeout}


def _log_params(
    limit: int,
    before: str | None,
    after: str | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit}
    if before is not None:
        params["before"] = before
    if after is not None:
        params["after"] = after
    return params


class Jobs(SyncResource):
    def submit(
        self,
        payload: dict[str, Any],
        *,
        workflow_format: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return unwrap(
            self._client.post(
                "/jobs",
                json_body=payload,
                headers=_workflow_headers(workflow_format),
                **_request_kwargs(timeout),
            )
        )

    def preview(
        self,
        payload: dict[str, Any],
        *,
        workflow_format: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Schedule a workflow without dispatching runtime work."""
        return unwrap(
            self._client.post(
                "/jobs/preview",
                json_body=payload,
                headers=_workflow_headers(workflow_format),
                **_request_kwargs(timeout),
            )
        )

    def list(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        return _items(
            unwrap(
                self._client.get(
                    "/jobs",
                    params=_list_params(status, limit, cursor),
                    **_request_kwargs(timeout),
                )
            )
        )

    def list_all(
        self,
        *,
        status: str | None = None,
        page_size: int | None = None,
        timeout: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Aborts if the server replays a cursor, to avoid infinite loops."""
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            payload = unwrap(
                self._client.get(
                    "/jobs",
                    params=_list_params(status, page_size, cursor),
                    **_request_kwargs(timeout),
                )
            )
            yield from _items(payload)
            cursor = _next_cursor(payload)
            if not cursor:
                return
            if cursor in seen_cursors:
                raise RuntimeError(
                    f"server replayed cursor {cursor!r}; aborting pagination"
                )
            seen_cursors.add(cursor)

    def get(self, job_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        return unwrap(self._client.get(f"/jobs/{job_id}", **_request_kwargs(timeout)))

    def progress(self, job_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        """Fetch detailed progress data for a job (available throughout its life)."""
        return unwrap(
            self._client.get(f"/jobs/{job_id}/progress", **_request_kwargs(timeout))
        )

    def result(self, job_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        return unwrap(
            self._client.get(f"/jobs/{job_id}/result", **_request_kwargs(timeout))
        )

    def inputs(self, job_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        return unwrap(
            self._client.get(f"/jobs/{job_id}/inputs", **_request_kwargs(timeout))
        )

    def list_workflows(
        self, job_id: str, *, timeout: float | None = None
    ) -> _list[JobWorkflowInfo]:
        """List FlowMesh workflows associated with a job."""
        payload = unwrap(
            self._client.get(f"/jobs/{job_id}/workflows", **_request_kwargs(timeout))
        )
        if isinstance(payload, dict):
            workflows = payload.get("workflows")
            if isinstance(workflows, list):
                return [JobWorkflowInfo.model_validate(w) for w in workflows]
        return []

    def get_logs(
        self,
        job_id: str,
        workflow_id: str,
        *,
        limit: int = 200,
        before: str | None = None,
        after: str | None = None,
        timeout: float | None = None,
    ) -> LogQueryResponse:
        """Fetch one page of logs for ``workflow_id`` under ``job_id``."""
        payload = unwrap(
            self._client.get(
                f"/jobs/{job_id}/workflows/{workflow_id}/logs",
                params=_log_params(limit, before, after),
                **_request_kwargs(timeout),
            )
        )
        return LogQueryResponse.model_validate(payload)

    def stream_logs(
        self,
        job_id: str,
        workflow_id: str,
        *,
        cursor: str | None = None,
        timeout: float | None = None,
    ) -> Iterator[LogEntry]:
        """Stream logs for ``workflow_id`` under ``job_id`` via SSE."""
        url = (
            f"{self._client.base_url.rstrip('/')}"
            f"/api/v1/jobs/{job_id}/workflows/{workflow_id}/logs/stream"
        )
        params: dict[str, str] = {}
        if cursor is not None:
            params["cursor"] = cursor
        extra: dict[str, Any] = {}
        if timeout is not None:
            extra["timeout"] = timeout
        with self._client._http.stream(
            "GET",
            url,
            params=params or None,
            headers={
                "Accept": "text/event-stream",
                "Authorization": (
                    f"Bearer {self._client.api_key}" if self._client.api_key else ""
                ),
            },
            **extra,
        ) as response:
            if response.status_code >= 400:
                response.read()
            _raise_for_status(response, url)
            buffer = ""
            for chunk in response.iter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    current_event: str | None = None
                    for line in block.splitlines():
                        if line.startswith("event:"):
                            current_event = line[len("event:") :].strip()
                        elif line.startswith("data:"):
                            data = line[len("data:") :].strip()
                            if data:
                                if current_event == "error":
                                    try:
                                        payload = json.loads(data)
                                        raise LogStreamError(
                                            code=payload.get("code", "APIError"),
                                            message=payload.get("message", ""),
                                        )
                                    except (ValueError, KeyError):
                                        raise LogStreamError(
                                            code="APIError",
                                            message="Upstream stream error.",
                                        )
                                yield LogEntry.model_validate_json(data)

    def download_logs(
        self,
        job_id: str,
        workflow_id: str,
        output_dir: Path,
        *,
        chunk_size: int = 256 * 1024,
        timeout: float | None = None,
    ) -> _list[Path]:
        """Download per-task archived logs for ``workflow_id`` under ``job_id``.

        Calls ``GET /api/v1/jobs/{job_id}/workflows/{workflow_id}/logs/download``,
        streams the tar response, extracts to ``output_dir``, and returns the
        list of extracted ``.jsonl`` file paths (one per task that had archived
        logs). Returns an empty list when the workflow had no archived task logs.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        url = (
            f"{self._client.base_url.rstrip('/')}"
            f"/api/v1/jobs/{job_id}/workflows/{workflow_id}/logs/download"
        )
        extra: dict[str, Any] = {}
        if timeout is not None:
            extra["timeout"] = timeout
        with self._client._http.stream(
            "GET",
            url,
            headers={
                "Accept": "application/x-tar",
                "Authorization": (
                    f"Bearer {self._client.api_key}" if self._client.api_key else ""
                ),
            },
            **extra,
        ) as response:
            if response.status_code >= 400:
                response.read()
            _raise_for_status(response, url)
            buf = BytesIO()
            for chunk in response.iter_bytes(chunk_size=chunk_size):
                if chunk:
                    buf.write(chunk)

        buf.seek(0)
        extracted: _list[Path] = []
        with tarfile.open(fileobj=buf, mode="r") as tf:
            for member in tf.getmembers():
                tf.extract(member, path=output_dir, filter="data")
                extracted.append(output_dir / member.name)
        return extracted

    def artifact(
        self,
        job_id: str,
        *,
        path: str,
        output: Path | str,
        chunk_size: int = 256 * 1024,
        timeout: float | None = None,
    ) -> Path:
        """Stream a job artifact to ``output`` and return the local path."""
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self._client.base_url.rstrip('/')}/api/v1/jobs/{job_id}/artifact"
        with self._client._http.stream(
            "GET",
            url,
            params={"path": path},
            headers={"Accept": "application/octet-stream"},
            **_request_kwargs(timeout),
        ) as response:
            if response.status_code >= 400:
                response.read()
            _raise_for_status(response, url)
            with target.open("wb") as fh:
                for chunk in response.iter_bytes(chunk_size=chunk_size):
                    if chunk:
                        fh.write(chunk)
        return target

    def cancel(self, job_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        return unwrap(
            self._client.post(f"/jobs/{job_id}/cancel", **_request_kwargs(timeout))
        )

    def wait(
        self,
        job_id: str,
        *,
        terminal_states: tuple[str, ...] = TERMINAL_STATES,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            job = self.get(job_id, timeout=request_timeout)
            if job.get("status", "") in terminal_states:
                return job
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"job {job_id!r} still in status {job.get('status', '')!r} "
                    f"after {timeout}s"
                )
            time.sleep(poll_interval)

    def watch(
        self,
        job_id: str,
        *,
        terminal_states: tuple[str, ...] = TERMINAL_STATES,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
        request_timeout: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yields ``{"status", "job", "progress"}`` snapshots until terminal."""
        deadline = time.monotonic() + timeout
        while True:
            job = self.get(job_id, timeout=request_timeout)
            try:
                prog = self.progress(job_id, timeout=request_timeout).get(
                    "progress", {}
                )
            except HttpError:
                prog = {}
            status = job.get("status", "")
            yield {"status": status, "job": job, "progress": prog}
            if status in terminal_states:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"job {job_id!r} still in status {status!r} after {timeout}s"
                )
            time.sleep(poll_interval)


class AsyncJobs(AsyncResource):
    async def submit(
        self,
        payload: dict[str, Any],
        *,
        workflow_format: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        response = await self._client.post(
            "/jobs",
            json_body=payload,
            headers=_workflow_headers(workflow_format),
            **_request_kwargs(timeout),
        )
        return unwrap(response)

    async def preview(
        self,
        payload: dict[str, Any],
        *,
        workflow_format: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        response = await self._client.post(
            "/jobs/preview",
            json_body=payload,
            headers=_workflow_headers(workflow_format),
            **_request_kwargs(timeout),
        )
        return unwrap(response)

    async def list(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        response = await self._client.get(
            "/jobs",
            params=_list_params(status, limit, cursor),
            **_request_kwargs(timeout),
        )
        return _items(unwrap(response))

    async def list_all(
        self,
        *,
        status: str | None = None,
        page_size: int | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            response = await self._client.get(
                "/jobs",
                params=_list_params(status, page_size, cursor),
                **_request_kwargs(timeout),
            )
            payload = unwrap(response)
            for item in _items(payload):
                yield item
            cursor = _next_cursor(payload)
            if not cursor:
                return
            if cursor in seen_cursors:
                raise RuntimeError(
                    f"server replayed cursor {cursor!r}; aborting pagination"
                )
            seen_cursors.add(cursor)

    async def get(self, job_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        response = await self._client.get(f"/jobs/{job_id}", **_request_kwargs(timeout))
        return unwrap(response)

    async def progress(
        self, job_id: str, *, timeout: float | None = None
    ) -> dict[str, Any]:
        response = await self._client.get(
            f"/jobs/{job_id}/progress", **_request_kwargs(timeout)
        )
        return unwrap(response)

    async def result(
        self, job_id: str, *, timeout: float | None = None
    ) -> dict[str, Any]:
        response = await self._client.get(
            f"/jobs/{job_id}/result", **_request_kwargs(timeout)
        )
        return unwrap(response)

    async def inputs(
        self, job_id: str, *, timeout: float | None = None
    ) -> dict[str, Any]:
        response = await self._client.get(
            f"/jobs/{job_id}/inputs", **_request_kwargs(timeout)
        )
        return unwrap(response)

    async def list_workflows(
        self, job_id: str, *, timeout: float | None = None
    ) -> _list[JobWorkflowInfo]:
        """List FlowMesh workflows associated with a job."""
        response = await self._client.get(
            f"/jobs/{job_id}/workflows", **_request_kwargs(timeout)
        )
        payload = unwrap(response)
        if isinstance(payload, dict):
            workflows = payload.get("workflows")
            if isinstance(workflows, list):
                return [JobWorkflowInfo.model_validate(w) for w in workflows]
        return []

    async def get_logs(
        self,
        job_id: str,
        workflow_id: str,
        *,
        limit: int = 200,
        before: str | None = None,
        after: str | None = None,
        timeout: float | None = None,
    ) -> LogQueryResponse:
        """Fetch one page of logs for ``workflow_id`` under ``job_id``."""
        response = await self._client.get(
            f"/jobs/{job_id}/workflows/{workflow_id}/logs",
            params=_log_params(limit, before, after),
            **_request_kwargs(timeout),
        )
        return LogQueryResponse.model_validate(unwrap(response))

    async def stream_logs(
        self,
        job_id: str,
        workflow_id: str,
        *,
        cursor: str | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[LogEntry]:
        """Stream logs for ``workflow_id`` under ``job_id`` via SSE."""
        url = (
            f"{self._client.base_url.rstrip('/')}"
            f"/api/v1/jobs/{job_id}/workflows/{workflow_id}/logs/stream"
        )
        params: dict[str, str] = {}
        if cursor is not None:
            params["cursor"] = cursor
        extra: dict[str, Any] = {}
        if timeout is not None:
            extra["timeout"] = timeout
        async with self._client._http.stream(
            "GET",
            url,
            params=params or None,
            headers={
                "Accept": "text/event-stream",
                "Authorization": (
                    f"Bearer {self._client.api_key}" if self._client.api_key else ""
                ),
            },
            **extra,
        ) as response:
            if response.status_code >= 400:
                await response.aread()
            _raise_for_status(response, url)
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    current_event: str | None = None
                    for line in block.splitlines():
                        if line.startswith("event:"):
                            current_event = line[len("event:") :].strip()
                        elif line.startswith("data:"):
                            data = line[len("data:") :].strip()
                            if data:
                                if current_event == "error":
                                    try:
                                        payload = json.loads(data)
                                        raise LogStreamError(
                                            code=payload.get("code", "APIError"),
                                            message=payload.get("message", ""),
                                        )
                                    except (ValueError, KeyError):
                                        raise LogStreamError(
                                            code="APIError",
                                            message="Upstream stream error.",
                                        )
                                yield LogEntry.model_validate_json(data)

    async def download_logs(
        self,
        job_id: str,
        workflow_id: str,
        output_dir: Path,
        *,
        chunk_size: int = 256 * 1024,
        timeout: float | None = None,
    ) -> _list[Path]:
        """Download per-task archived logs for ``workflow_id`` under ``job_id``.

        Async counterpart to ``Jobs.download_logs``. Calls
        ``GET /api/v1/jobs/{job_id}/workflows/{workflow_id}/logs/download``,
        streams the tar response, extracts to ``output_dir``, and returns the
        list of extracted ``.jsonl`` file paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        url = (
            f"{self._client.base_url.rstrip('/')}"
            f"/api/v1/jobs/{job_id}/workflows/{workflow_id}/logs/download"
        )
        extra: dict[str, Any] = {}
        if timeout is not None:
            extra["timeout"] = timeout
        async with self._client._http.stream(
            "GET",
            url,
            headers={
                "Accept": "application/x-tar",
                "Authorization": (
                    f"Bearer {self._client.api_key}" if self._client.api_key else ""
                ),
            },
            **extra,
        ) as response:
            if response.status_code >= 400:
                await response.aread()
            _raise_for_status(response, url)
            buf = BytesIO()
            async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                if chunk:
                    buf.write(chunk)

        buf.seek(0)
        extracted: _list[Path] = []
        with tarfile.open(fileobj=buf, mode="r") as tf:
            for member in tf.getmembers():
                tf.extract(member, path=output_dir, filter="data")
                extracted.append(output_dir / member.name)
        return extracted

    async def artifact(
        self,
        job_id: str,
        *,
        path: str,
        output: Path | str,
        chunk_size: int = 256 * 1024,
        timeout: float | None = None,
    ) -> Path:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self._client.base_url.rstrip('/')}/api/v1/jobs/{job_id}/artifact"
        async with self._client._http.stream(
            "GET",
            url,
            params={"path": path},
            headers={"Accept": "application/octet-stream"},
            **_request_kwargs(timeout),
        ) as response:
            if response.status_code >= 400:
                await response.aread()
            _raise_for_status(response, url)
            with target.open("wb") as fh:
                async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                    if chunk:
                        fh.write(chunk)
        return target

    async def cancel(
        self, job_id: str, *, timeout: float | None = None
    ) -> dict[str, Any]:
        response = await self._client.post(
            f"/jobs/{job_id}/cancel", **_request_kwargs(timeout)
        )
        return unwrap(response)

    async def wait(
        self,
        job_id: str,
        *,
        terminal_states: tuple[str, ...] = TERMINAL_STATES,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            job = await self.get(job_id, timeout=request_timeout)
            if job.get("status", "") in terminal_states:
                return job
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"job {job_id!r} still in status {job.get('status', '')!r} "
                    f"after {timeout}s"
                )
            await asyncio.sleep(poll_interval)

    async def watch(
        self,
        job_id: str,
        *,
        terminal_states: tuple[str, ...] = TERMINAL_STATES,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
        request_timeout: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while True:
            job = await self.get(job_id, timeout=request_timeout)
            try:
                prog_payload = await self.progress(job_id, timeout=request_timeout)
                prog = prog_payload.get("progress", {})
            except HttpError:
                prog = {}
            status = job.get("status", "")
            yield {"status": status, "job": job, "progress": prog}
            if status in terminal_states:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"job {job_id!r} still in status {status!r} after {timeout}s"
                )
            await asyncio.sleep(poll_interval)
