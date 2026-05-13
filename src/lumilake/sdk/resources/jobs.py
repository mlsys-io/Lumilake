"""Optimization jobs backed by the server's ``/api/v1/jobs`` routes."""

import asyncio
import logging
import time
from typing import Any

from lumilake.sdk._base_client import unwrap
from lumilake.sdk.resources._base import AsyncResource, SyncResource

logger = logging.getLogger(__name__)


def _items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and "items" in data:
        return list(data["items"])
    return list(data) if isinstance(data, list) else []


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


class Jobs(SyncResource):
    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        return unwrap(self._client.post("/jobs", json_body=payload))

    def list(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        return _items(
            unwrap(
                self._client.get("/jobs", params=_list_params(status, limit, cursor))
            )
        )

    def get(self, job_id: str) -> dict[str, Any]:
        return unwrap(self._client.get(f"/jobs/{job_id}"))

    def cancel(self, job_id: str) -> dict[str, Any]:
        return unwrap(self._client.post(f"/jobs/{job_id}/cancel"))

    def wait(
        self,
        job_id: str,
        *,
        terminal_states: tuple[str, ...] = ("completed", "failed", "cancelled"),
        poll_interval: float = 2.0,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            job = self.get(job_id)
            if job.get("status", "") in terminal_states:
                return job
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"job {job_id!r} still in status {job.get('status', '')!r} "
                    f"after {timeout}s"
                )
            time.sleep(poll_interval)


class AsyncJobs(AsyncResource):
    async def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._client.post("/jobs", json_body=payload)
        return unwrap(response)

    async def list(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        response = await self._client.get(
            "/jobs", params=_list_params(status, limit, cursor)
        )
        return _items(unwrap(response))

    async def get(self, job_id: str) -> dict[str, Any]:
        response = await self._client.get(f"/jobs/{job_id}")
        return unwrap(response)

    async def cancel(self, job_id: str) -> dict[str, Any]:
        response = await self._client.post(f"/jobs/{job_id}/cancel")
        return unwrap(response)

    async def wait(
        self,
        job_id: str,
        *,
        terminal_states: tuple[str, ...] = ("completed", "failed", "cancelled"),
        poll_interval: float = 2.0,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            job = await self.get(job_id)
            if job.get("status", "") in terminal_states:
                return job
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"job {job_id!r} still in status {job.get('status', '')!r} "
                    f"after {timeout}s"
                )
            await asyncio.sleep(poll_interval)
