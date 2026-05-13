"""``lumilake.sdk`` — Python SDK for Lumilake.

Programmatic equivalent of the ``lumilake`` CLI. Every CLI subcommand has a
matching method on ``LumilakeClient`` / ``AsyncLumilakeClient`` (or one of
their resource clients).

Sync usage::

    from lumilake.sdk import LumilakeClient

    with LumilakeClient.from_config() as client:
        client.jobs.list()
        client.deploy.up()

Async usage::

    from lumilake.sdk import AsyncLumilakeClient

    async with AsyncLumilakeClient.from_config() as client:
        await client.jobs.list()
        await client.deploy.up()
"""

from lumilake.sdk._base_client import BaseAsyncClient, BaseClient, unwrap
from lumilake.sdk.async_client import AsyncLumilakeClient
from lumilake.sdk.client import LumilakeClient
from lumilake.sdk.config import DEFAULT_CONFIG_PATH, LumilakeConfig
from lumilake.sdk.errors import (
    DeployError,
    HttpError,
    LumilakeError,
    NotFoundError,
)
from lumilake.sdk.resources.deploy import CONTAINER_NAMES, AsyncDeploy, Deploy
from lumilake.sdk.resources.info import AsyncInfo, Info
from lumilake.sdk.resources.jobs import AsyncJobs, Jobs
from lumilake.sdk.resources.traces import AsyncTraces, Traces
from lumilake.sdk.resources.workers import AsyncWorkers, Workers

__all__ = [
    "CONTAINER_NAMES",
    "DEFAULT_CONFIG_PATH",
    "AsyncDeploy",
    "AsyncInfo",
    "AsyncJobs",
    "AsyncLumilakeClient",
    "AsyncTraces",
    "AsyncWorkers",
    "BaseAsyncClient",
    "BaseClient",
    "Deploy",
    "DeployError",
    "HttpError",
    "Info",
    "Jobs",
    "LumilakeClient",
    "LumilakeConfig",
    "LumilakeError",
    "NotFoundError",
    "Traces",
    "Workers",
    "unwrap",
]
