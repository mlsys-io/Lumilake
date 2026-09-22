"""Node resource operations."""

import builtins
from typing import Any

from ..models.nodes import (
    Node,
    NodeRegisterResponse,
    NodeWorkerInfo,
    WorkerRegisterResponse,
)
from ..params import append_param, extend_params
from ._base import AsyncResource, SyncResource


class Nodes(SyncResource):
    """Synchronous node operations."""

    def retrieve(self, node_id: str) -> Node:
        """Retrieve node details by ID."""
        data = self._client._request("GET", f"/nodes/{node_id}")
        return Node.model_validate(data)

    def list(
        self,
        node_id: str | None = None,
        namespace: str | None = None,
        cluster: str | None = None,
        alias: str | None = None,
        tags: str | list[str] | None = None,
        query_params: builtins.list[tuple[str, str]] | None = None,
    ) -> list[Node]:
        """List nodes with optional filters."""
        params: list[tuple[str, str]] = []
        append_param(params, "id", node_id)
        append_param(params, "namespace", namespace)
        append_param(params, "cluster", cluster)
        append_param(params, "alias", alias)
        extend_params(params, "tags", tags)
        if query_params:
            params.extend(query_params)
        data = self._client._request("GET", "/nodes", params=params or None)
        return [Node.model_validate(g) for g in data]

    def register(
        self,
        namespace: str,
        cluster: str,
        alias: str,
        started_at: str,
        version: str | None = None,
        tags: builtins.list[str] | None = None,
        last_seen: str | None = None,
    ) -> NodeRegisterResponse:
        """Register a new node."""
        payload: dict[str, Any] = {
            "namespace": namespace,
            "cluster": cluster,
            "alias": alias,
            "started_at": started_at,
            "tags": tags or [],
            "last_seen": last_seen or started_at,
        }
        if version is not None:
            payload["version"] = version
        data = self._client._request("POST", "/nodes/register", json_body=payload)
        return NodeRegisterResponse.model_validate(data)

    def list_workers(
        self,
        node_id: str,
        query_params: builtins.list[tuple[str, str]] | None = None,
    ) -> builtins.list[NodeWorkerInfo]:
        """List workers managed by a specific node."""
        data = self._client._request(
            "GET",
            f"/nodes/{node_id}/workers",
            params=query_params or None,
        )
        return [NodeWorkerInfo.model_validate(w) for w in data]

    def list_all_workers(
        self,
        query_params: builtins.list[tuple[str, str]] | None = None,
    ) -> builtins.list[NodeWorkerInfo]:
        """List all workers across all nodes."""
        data = self._client._request(
            "GET", "/nodes/workers", params=query_params or None
        )
        return [NodeWorkerInfo.model_validate(w) for w in data]

    def register_worker(
        self,
        node_id: str,
        worker_metadata: dict[str, Any],
    ) -> WorkerRegisterResponse:
        """Register a worker under a node."""
        data = self._client._request(
            "POST",
            f"/nodes/{node_id}/workers/register",
            json_body=worker_metadata,
        )
        return WorkerRegisterResponse.model_validate(data)

    def start_worker(self, node_id: str, worker_name: str) -> None:
        """Start a worker on a node."""
        self._client._request(
            "POST",
            f"/nodes/{node_id}/workers/{worker_name}/start",
        )

    def stop_worker(self, node_id: str, worker_name: str) -> None:
        """Stop a worker on a node."""
        self._client._request(
            "POST",
            f"/nodes/{node_id}/workers/{worker_name}/stop",
        )


class AsyncNodes(AsyncResource):
    """Asynchronous node operations."""

    async def retrieve(self, node_id: str) -> Node:
        """Retrieve node details by ID."""
        data = await self._client._request("GET", f"/nodes/{node_id}")
        return Node.model_validate(data)

    async def list(
        self,
        node_id: str | None = None,
        namespace: str | None = None,
        cluster: str | None = None,
        alias: str | None = None,
        tags: str | list[str] | None = None,
        query_params: builtins.list[tuple[str, str]] | None = None,
    ) -> list[Node]:
        """List nodes with optional filters."""
        params: list[tuple[str, str]] = []
        append_param(params, "id", node_id)
        append_param(params, "namespace", namespace)
        append_param(params, "cluster", cluster)
        append_param(params, "alias", alias)
        extend_params(params, "tags", tags)
        if query_params:
            params.extend(query_params)
        data = await self._client._request("GET", "/nodes", params=params or None)
        return [Node.model_validate(g) for g in data]

    async def register(
        self,
        namespace: str,
        cluster: str,
        alias: str,
        started_at: str,
        version: str | None = None,
        tags: builtins.list[str] | None = None,
        last_seen: str | None = None,
    ) -> NodeRegisterResponse:
        """Register a new node."""
        payload: dict[str, Any] = {
            "namespace": namespace,
            "cluster": cluster,
            "alias": alias,
            "started_at": started_at,
            "tags": tags or [],
            "last_seen": last_seen or started_at,
        }
        if version is not None:
            payload["version"] = version
        data = await self._client._request("POST", "/nodes/register", json_body=payload)
        return NodeRegisterResponse.model_validate(data)

    async def list_workers(
        self,
        node_id: str,
        query_params: builtins.list[tuple[str, str]] | None = None,
    ) -> builtins.list[NodeWorkerInfo]:
        """List workers managed by a specific node."""
        data = await self._client._request(
            "GET",
            f"/nodes/{node_id}/workers",
            params=query_params or None,
        )
        return [NodeWorkerInfo.model_validate(w) for w in data]

    async def list_all_workers(
        self,
        query_params: builtins.list[tuple[str, str]] | None = None,
    ) -> builtins.list[NodeWorkerInfo]:
        """List all workers across all nodes."""
        data = await self._client._request(
            "GET", "/nodes/workers", params=query_params or None
        )
        return [NodeWorkerInfo.model_validate(w) for w in data]

    async def register_worker(
        self,
        node_id: str,
        worker_metadata: dict[str, Any],
    ) -> WorkerRegisterResponse:
        """Register a worker under a node."""
        data = await self._client._request(
            "POST",
            f"/nodes/{node_id}/workers/register",
            json_body=worker_metadata,
        )
        return WorkerRegisterResponse.model_validate(data)

    async def start_worker(self, node_id: str, worker_name: str) -> None:
        """Start a worker on a node."""
        await self._client._request(
            "POST",
            f"/nodes/{node_id}/workers/{worker_name}/start",
        )

    async def stop_worker(self, node_id: str, worker_name: str) -> None:
        """Stop a worker on a node."""
        await self._client._request(
            "POST",
            f"/nodes/{node_id}/workers/{worker_name}/stop",
        )
