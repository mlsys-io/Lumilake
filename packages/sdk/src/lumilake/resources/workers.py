"""Runtime workers — list registered workers and inspect a single one
by id, with optional status / tag filters. Read-only; worker lifecycle
(start / stop) lives under FlowMesh, not the lumilake server.
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
        cursor = data.get("next_cursor")
        if isinstance(cursor, str) and cursor:
            return cursor
    return None


def _list_params(
    status: str | None,
    tag: str | None,
    limit: int | None = None,
    cursor: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if status is not None:
        params["status"] = status
    if tag is not None:
        params["tag"] = tag
    if limit is not None:
        params["limit"] = limit
    if cursor is not None:
        params["cursor"] = cursor
    return params


def _request_kwargs(timeout: float | None) -> dict[str, Any]:
    if timeout is None:
        return {}
    return {"timeout": timeout}


class Workers(SyncResource):
    def list(
        self,
        *,
        status: str | None = None,
        tag: str | None = None,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        return _items(
            unwrap(
                self._client.get(
                    "/workers",
                    params=_list_params(status, tag),
                    **_request_kwargs(timeout),
                )
            )
        )

    def list_all(
        self,
        *,
        status: str | None = None,
        tag: str | None = None,
        page_size: int | None = None,
        timeout: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate through every worker, traversing pagination cursors."""
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            payload = unwrap(
                self._client.get(
                    "/workers",
                    params=_list_params(status, tag, page_size, cursor),
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

    def get(self, worker_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        return unwrap(
            self._client.get(f"/workers/{worker_id}", **_request_kwargs(timeout))
        )


class AsyncWorkers(AsyncResource):
    async def list(
        self,
        *,
        status: str | None = None,
        tag: str | None = None,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        response = await self._client.get(
            "/workers",
            params=_list_params(status, tag),
            **_request_kwargs(timeout),
        )
        return _items(unwrap(response))

    async def list_all(
        self,
        *,
        status: str | None = None,
        tag: str | None = None,
        page_size: int | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            response = await self._client.get(
                "/workers",
                params=_list_params(status, tag, page_size, cursor),
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

    async def get(
        self, worker_id: str, *, timeout: float | None = None
    ) -> dict[str, Any]:
        response = await self._client.get(
            f"/workers/{worker_id}", **_request_kwargs(timeout)
        )
        return unwrap(response)
