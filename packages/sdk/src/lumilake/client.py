"""LumilakeClient — synchronous SDK entry point.

Composes every server-API resource (HTTP) plus the ``deploy`` resource
(subprocess) and exposes them as attributes (``client.jobs``,
``client.workers``, ``client.deploy``, …). Construct with an explicit
``base_url``, ``LUMILAKE_BASE_URL`` env var, or
``LumilakeClient.from_config()`` to read the file ``lumilake deploy
up`` writes.
"""

import logging
from pathlib import Path
from typing import Any, Self

import httpx
from lumilake._base_client import BaseClient, resolve_config
from lumilake.config import LumilakeConfig
from lumilake.resources.deploy import Deploy
from lumilake.resources.info import Info
from lumilake.resources.jobs import Jobs
from lumilake.resources.traces import Traces
from lumilake.resources.workers import Workers

logger = logging.getLogger(__name__)


class LumilakeClient(BaseClient):
    """Synchronous Lumilake API client.

    Usage::

        from lumilake import LumilakeClient

        with LumilakeClient(base_url="http://localhost:19000") as client:
            client.jobs.list()
            client.deploy.up()

    Or pull the URL from ``~/.lumilake/config.toml`` (written by
    ``lumilake deploy up``)::

        client = LumilakeClient.from_config()
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float | None = None,
        verify: bool | str = True,
        http_client: httpx.Client | None = None,
        repo_root: Path | str | None = None,
    ) -> None:
        url = resolve_config(base_url)
        super().__init__(
            base_url=url,
            timeout=timeout,
            verify=verify,
            http_client=http_client,
        )
        self._repo_root = Path(repo_root) if repo_root else Path.cwd()

        self.info = Info(self)
        self.deploy = Deploy(self._repo_root)
        self.jobs = Jobs(self)
        self.workers = Workers(self)
        self.traces = Traces(self)

    @classmethod
    def from_config(
        cls,
        path: Path | str | None = None,
        *,
        timeout: float | None = None,
        verify: bool | str = True,
        http_client: httpx.Client | None = None,
        repo_root: Path | str | None = None,
    ) -> Self:
        config = LumilakeConfig.load(path)
        return cls(
            base_url=config.base_url,
            timeout=timeout,
            verify=verify,
            http_client=http_client,
            repo_root=repo_root,
        )

    def health(self) -> dict[str, Any]:
        """Shortcut for ``client.info.health()``."""
        return self.info.health()
