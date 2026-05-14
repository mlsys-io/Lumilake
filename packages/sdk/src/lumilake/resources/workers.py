"""Runtime workers — list registered workers and inspect a single one
by id, with optional status / tag filters. Read-only; worker lifecycle
(start / stop) lives under FlowMesh, not the lumilake server.
"""

from typing import Any

from lumilake._base_client import unwrap
from lumilake.resources._base import AsyncResource, SyncResource


def _items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and "items" in data:
        return list(data["items"])
    return list(data) if isinstance(data, list) else []


def _list_params(status: str | None, tag: str | None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if status is not None:
        params["status"] = status
    if tag is not None:
        params["tag"] = tag
    return params


class Workers(SyncResource):
    def list(
        self,
        *,
        status: str | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        return _items(
            unwrap(self._client.get("/workers", params=_list_params(status, tag)))
        )

    def get(self, worker_id: str) -> dict[str, Any]:
        return unwrap(self._client.get(f"/workers/{worker_id}"))


class AsyncWorkers(AsyncResource):
    async def list(
        self,
        *,
        status: str | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        response = await self._client.get("/workers", params=_list_params(status, tag))
        return _items(unwrap(response))

    async def get(self, worker_id: str) -> dict[str, Any]:
        response = await self._client.get(f"/workers/{worker_id}")
        return unwrap(response)
