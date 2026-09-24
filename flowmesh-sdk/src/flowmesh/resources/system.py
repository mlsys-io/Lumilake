"""System resource operations."""

from typing import Any

import httpx

from .._base_client import _raise_for_status
from ..exceptions import FlowMeshConnectionError
from ..models.common import OkResponse, VersionResponse
from ._base import AsyncResource, SyncResource


class System(SyncResource):
    """Synchronous system operations."""

    def metrics(self) -> dict[str, Any]:
        """Get system metrics snapshot (admin only)."""
        return self._client._request("GET", "/system/metrics")

    def version(self) -> VersionResponse:
        """Get the server version."""
        return VersionResponse.model_validate(
            self._client._request("GET", "/system/version")
        )

    def health(self) -> OkResponse:
        """Check server health.

        Note: Uses ``/healthz`` which is outside the ``/api/v1`` prefix.
        """

        url = self._client.base_url.rstrip("/") + "/healthz"
        try:
            response = self._client._http.get(url)
        except httpx.ConnectError as exc:
            raise FlowMeshConnectionError(f"Failed to connect to {url}: {exc}")
        _raise_for_status(response, "GET")
        return OkResponse.model_validate(response.json())


class AsyncSystem(AsyncResource):
    """Asynchronous system operations."""

    async def metrics(self) -> dict[str, Any]:
        """Get system metrics snapshot (admin only)."""
        return await self._client._request("GET", "/system/metrics")

    async def version(self) -> VersionResponse:
        """Get the server version."""
        return VersionResponse.model_validate(
            await self._client._request("GET", "/system/version")
        )

    async def health(self) -> OkResponse:
        """Check server health."""
        url = self._client.base_url.rstrip("/") + "/healthz"
        try:
            response = await self._client._http.get(url)
        except httpx.ConnectError as exc:
            raise FlowMeshConnectionError(f"Failed to connect to {url}: {exc}")
        _raise_for_status(response, "GET")
        return OkResponse.model_validate(response.json())
