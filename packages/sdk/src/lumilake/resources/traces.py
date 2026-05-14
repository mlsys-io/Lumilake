"""Execution traces — list traces (optionally filtered by job_id) and
fetch a single trace's full payload. Traces capture per-op metrics +
inputs / outputs for a job's runtime DAG.
"""

from typing import Any

from lumilake._base_client import unwrap
from lumilake.resources._base import AsyncResource, SyncResource


def _items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and "items" in data:
        return list(data["items"])
    return list(data) if isinstance(data, list) else []


def _list_params(job_id: str | None, limit: int | None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if job_id is not None:
        params["job_id"] = job_id
    if limit is not None:
        params["limit"] = limit
    return params


class Traces(SyncResource):
    def list(
        self,
        *,
        job_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _items(
            unwrap(self._client.get("/traces", params=_list_params(job_id, limit)))
        )

    def get(self, trace_id: str) -> dict[str, Any]:
        return unwrap(self._client.get(f"/traces/{trace_id}"))


class AsyncTraces(AsyncResource):
    async def list(
        self,
        *,
        job_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        response = await self._client.get("/traces", params=_list_params(job_id, limit))
        return _items(unwrap(response))

    async def get(self, trace_id: str) -> dict[str, Any]:
        response = await self._client.get(f"/traces/{trace_id}")
        return unwrap(response)
