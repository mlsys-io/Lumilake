"""Optimization jobs backed by the server's ``/api/v1/jobs`` routes."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from pathlib import Path
from typing import Any

from lumilake._base_client import unwrap
from lumilake.resources._base import AsyncResource, SyncResource

logger = logging.getLogger(__name__)

TERMINAL_STATES: tuple[str, ...] = ("completed", "failed", "cancelled")


def _items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and "items" in data:
        return list(data["items"])
    return list(data) if isinstance(data, list) else []


def _next_cursor(data: Any) -> str | None:
    if isinstance(data, dict):
        cursor = data.get("next_cursor") or data.get("cursor")
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


def _preview_headers(workflow_format: str) -> dict[str, str]:
    return {"Workflow-Format": workflow_format}


def _request_kwargs(timeout: float | None) -> dict[str, Any]:
    """Build per-call kwargs for the underlying httpx request.

    Returns ``{}`` when ``timeout`` is ``None`` so the transport uses
    its configured default. When set, passes the value through as an
    httpx-compatible per-request ``timeout``.
    """
    if timeout is None:
        return {}
    return {"timeout": timeout}


class Jobs(SyncResource):
    def submit(
        self,
        payload: dict[str, Any],
        *,
        workflow_format: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        headers: Mapping[str, str] | None = (
            {"Workflow-Format": workflow_format} if workflow_format else None
        )
        return unwrap(
            self._client.post(
                "/jobs",
                json_body=payload,
                headers=headers,
                **_request_kwargs(timeout),
            )
        )

    def preview(
        self,
        payload: dict[str, Any],
        *,
        workflow_format: str = "n8n",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Schedule a workflow without dispatching runtime work."""
        return unwrap(
            self._client.post(
                "/jobs/preview",
                json_body=payload,
                headers=_preview_headers(workflow_format),
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
        """Iterate through every job, traversing pagination cursors.

        Yields one job dict at a time. ``page_size`` is forwarded as the
        underlying ``limit`` and bounds each request; the iterator stops
        when the server stops returning a ``next_cursor``.
        """
        cursor: str | None = None
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

    def get(self, job_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        return unwrap(self._client.get(f"/jobs/{job_id}", **_request_kwargs(timeout)))

    def progress(self, job_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        """Fetch detailed progress data for a job (available throughout its life)."""
        return unwrap(
            self._client.get(f"/jobs/{job_id}/progress", **_request_kwargs(timeout))
        )

    def result(self, job_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        """Fetch the final result of a completed job."""
        return unwrap(
            self._client.get(f"/jobs/{job_id}/result", **_request_kwargs(timeout))
        )

    def inputs(self, job_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        """Fetch the resolved inputs of a submitted job."""
        return unwrap(
            self._client.get(f"/jobs/{job_id}/inputs", **_request_kwargs(timeout))
        )

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
            response.raise_for_status()
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
        """Poll a job until it reaches a terminal state, yielding every status snapshot.

        Each yielded dict carries the current ``status``, ``job``, and
        ``progress`` fields. The iterator stops once a terminal state is
        observed (yielding the terminal snapshot first).
        """
        deadline = time.monotonic() + timeout
        while True:
            job = self.get(job_id, timeout=request_timeout)
            try:
                prog = self.progress(job_id, timeout=request_timeout).get(
                    "progress", {}
                )
            except Exception:  # noqa: BLE001 -- progress is best-effort here
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
        headers: Mapping[str, str] | None = (
            {"Workflow-Format": workflow_format} if workflow_format else None
        )
        response = await self._client.post(
            "/jobs",
            json_body=payload,
            headers=headers,
            **_request_kwargs(timeout),
        )
        return unwrap(response)

    async def preview(
        self,
        payload: dict[str, Any],
        *,
        workflow_format: str = "n8n",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        response = await self._client.post(
            "/jobs/preview",
            json_body=payload,
            headers=_preview_headers(workflow_format),
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
            response.raise_for_status()
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
            except Exception:  # noqa: BLE001 -- progress is best-effort here
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
