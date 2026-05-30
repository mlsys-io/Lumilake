"""AsyncLumilakeClient — asynchronous SDK entry point.

Composes every async resource — HTTP-backed ones over ``httpx.AsyncClient``,
``deploy`` over ``asyncio.to_thread`` calls into ``lumilake_deploy``. Same
construction options as ``LumilakeClient`` (explicit args, env, or
``.from_config()``).
"""

import logging
from pathlib import Path
from typing import Any, Self

import httpx
from lumilake._base_client import BaseAsyncClient, resolve_config
from lumilake.config import DEFAULT_CONFIG_PATH, LumilakeConfig
from lumilake.resources.deploy import AsyncDeploy
from lumilake.resources.info import AsyncInfo
from lumilake.resources.jobs import AsyncJobs
from lumilake.resources.traces import AsyncTraces
from lumilake.resources.workers import AsyncWorkers

logger = logging.getLogger(__name__)


class AsyncLumilakeClient(BaseAsyncClient):
    """Asynchronous Lumilake API client.

    Usage::

        from lumilake import AsyncLumilakeClient

        async with AsyncLumilakeClient(
            base_url="http://localhost:19000",
        ) as client:
            await client.jobs.list()
            await client.deploy.up()

    Or::

        client = AsyncLumilakeClient.from_config()
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float | None = None,
        verify: bool | str = True,
        http_client: httpx.AsyncClient | None = None,
        repo_root: Path | str | None = None,
    ) -> None:
        cfg = resolve_config(base_url, api_key)
        super().__init__(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=timeout,
            verify=verify,
            http_client=http_client,
        )
        self._repo_root = Path(repo_root) if repo_root else Path.cwd()

        self.info = AsyncInfo(self)
        self.deploy = AsyncDeploy(self._repo_root)
        self.jobs = AsyncJobs(self)
        self.workers = AsyncWorkers(self)
        self.traces = AsyncTraces(self)

    @classmethod
    def from_config(
        cls,
        path: Path | str | None = None,
        *,
        api_key: str | None = None,
        timeout: float | None = None,
        verify: bool | str = True,
        http_client: httpx.AsyncClient | None = None,
        repo_root: Path | str | None = None,
    ) -> Self:
        config = LumilakeConfig.from_file(Path(path) if path else DEFAULT_CONFIG_PATH)
        return cls(
            base_url=config.base_url,
            api_key=api_key if api_key is not None else config.api_key,
            timeout=timeout,
            verify=verify,
            http_client=http_client,
            repo_root=repo_root,
        )

    async def health(self) -> dict[str, Any]:
        return await self.info.health()
