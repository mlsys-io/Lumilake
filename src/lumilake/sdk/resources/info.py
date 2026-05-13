"""Server status probes. ``health()`` hits the unversioned ``/healthz``
endpoint; ``status()`` hits ``GET /info`` for version, runtime
configuration, and feature flags.
"""

from typing import Any

from lumilake.sdk._base_client import unwrap
from lumilake.sdk.resources._base import AsyncResource, SyncResource


class Info(SyncResource):
    def health(self) -> dict[str, Any]:
        return self._client.get("/healthz", version_prefix=False).json()

    def status(self) -> dict[str, Any]:
        return unwrap(self._client.get("/info"))


class AsyncInfo(AsyncResource):
    async def health(self) -> dict[str, Any]:
        response = await self._client.get("/healthz", version_prefix=False)
        return response.json()

    async def status(self) -> dict[str, Any]:
        response = await self._client.get("/info")
        return unwrap(response)
