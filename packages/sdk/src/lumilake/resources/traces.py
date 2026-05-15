"""Execution traces — list traces (optionally filtered by job_id) and
fetch a single trace's full payload. Traces capture per-op metrics +
inputs / outputs for a job's runtime DAG.
"""

from collections.abc import AsyncIterator, Iterator
from typing import Any

from lumilake._base_client import unwrap
from lumilake.resources._base import AsyncResource, SyncResource


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
    job_id: str | None,
    limit: int | None,
    cursor: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if job_id is not None:
        params["job_id"] = job_id
    if limit is not None:
        params["limit"] = limit
    if cursor is not None:
        params["cursor"] = cursor
    return params


def _request_kwargs(timeout: float | None) -> dict[str, Any]:
    if timeout is None:
        return {}
    return {"timeout": timeout}


class Traces(SyncResource):
    def list(
        self,
        *,
        job_id: str | None = None,
        limit: int | None = None,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        return _items(
            unwrap(
                self._client.get(
                    "/traces",
                    params=_list_params(job_id, limit),
                    **_request_kwargs(timeout),
                )
            )
        )

    def list_all(
        self,
        *,
        job_id: str | None = None,
        page_size: int | None = None,
        timeout: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate through every trace, traversing pagination cursors."""
        cursor: str | None = None
        while True:
            payload = unwrap(
                self._client.get(
                    "/traces",
                    params=_list_params(job_id, page_size, cursor),
                    **_request_kwargs(timeout),
                )
            )
            yield from _items(payload)
            cursor = _next_cursor(payload)
            if not cursor:
                return

    def get(self, trace_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        return unwrap(
            self._client.get(f"/traces/{trace_id}", **_request_kwargs(timeout))
        )


class AsyncTraces(AsyncResource):
    async def list(
        self,
        *,
        job_id: str | None = None,
        limit: int | None = None,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        response = await self._client.get(
            "/traces",
            params=_list_params(job_id, limit),
            **_request_kwargs(timeout),
        )
        return _items(unwrap(response))

    async def list_all(
        self,
        *,
        job_id: str | None = None,
        page_size: int | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        cursor: str | None = None
        while True:
            response = await self._client.get(
                "/traces",
                params=_list_params(job_id, page_size, cursor),
                **_request_kwargs(timeout),
            )
            payload = unwrap(response)
            for item in _items(payload):
                yield item
            cursor = _next_cursor(payload)
            if not cursor:
                return

    async def get(
        self, trace_id: str, *, timeout: float | None = None
    ) -> dict[str, Any]:
        response = await self._client.get(
            f"/traces/{trace_id}", **_request_kwargs(timeout)
        )
        return unwrap(response)
