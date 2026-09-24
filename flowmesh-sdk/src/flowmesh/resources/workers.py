"""Worker resource operations."""

import builtins

from ..models.workers import WorkerInfo
from ..params import append_param, extend_params
from ._base import AsyncResource, SyncResource


class Workers(SyncResource):
    """Synchronous worker operations."""

    def retrieve(self, worker_id: str) -> WorkerInfo:
        """Retrieve worker details by ID."""
        data = self._client._request("GET", f"/workers/{worker_id}")
        return WorkerInfo.model_validate(data)

    def list(
        self,
        worker_id: str | None = None,
        alias: str | None = None,
        namespace: str | None = None,
        cluster: str | None = None,
        status: str | builtins.list[str] | None = None,
        tags: str | builtins.list[str] | None = None,
        stale: bool | None = None,
        query_params: builtins.list[tuple[str, str]] | None = None,
    ) -> builtins.list[WorkerInfo]:
        """List workers with optional filters."""
        params: list[tuple[str, str]] = []
        append_param(params, "id", worker_id)
        append_param(params, "alias", alias)
        append_param(params, "namespace", namespace)
        append_param(params, "cluster", cluster)
        extend_params(params, "status", status)
        extend_params(params, "tags", tags)
        append_param(params, "stale", stale)
        if query_params:
            params.extend(query_params)
        data = self._client._request("GET", "/workers", params=params or None)
        return [WorkerInfo.model_validate(w) for w in data]


class AsyncWorkers(AsyncResource):
    """Asynchronous worker operations."""

    async def retrieve(self, worker_id: str) -> WorkerInfo:
        """Retrieve worker details by ID."""
        data = await self._client._request("GET", f"/workers/{worker_id}")
        return WorkerInfo.model_validate(data)

    async def list(
        self,
        worker_id: str | None = None,
        alias: str | None = None,
        namespace: str | None = None,
        cluster: str | None = None,
        status: str | builtins.list[str] | None = None,
        tags: str | builtins.list[str] | None = None,
        stale: bool | None = None,
        query_params: builtins.list[tuple[str, str]] | None = None,
    ) -> builtins.list[WorkerInfo]:
        """List workers with optional filters."""
        params: list[tuple[str, str]] = []
        append_param(params, "id", worker_id)
        append_param(params, "alias", alias)
        append_param(params, "namespace", namespace)
        append_param(params, "cluster", cluster)
        extend_params(params, "status", status)
        extend_params(params, "tags", tags)
        append_param(params, "stale", stale)
        if query_params:
            params.extend(query_params)
        data = await self._client._request("GET", "/workers", params=params or None)
        return [WorkerInfo.model_validate(w) for w in data]
