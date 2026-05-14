"""Server health probes. ``health()`` hits the unversioned ``/healthz`` endpoint."""

from typing import Any

from lumilake.resources._base import AsyncResource, SyncResource


class Info(SyncResource):
    def health(self) -> dict[str, Any]:
        return self._client.get("/healthz", version_prefix=False).json()

    def status(self) -> dict[str, Any]:
        return self.health()


class AsyncInfo(AsyncResource):
    async def health(self) -> dict[str, Any]:
        response = await self._client.get("/healthz", version_prefix=False)
        return response.json()

    async def status(self) -> dict[str, Any]:
        return await self.health()
