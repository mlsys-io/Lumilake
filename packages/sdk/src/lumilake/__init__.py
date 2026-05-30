"""Lumilake Python SDK.

Programmatic equivalent of the ``lumilake`` CLI. Every CLI subcommand has a
matching method on ``LumilakeClient`` / ``AsyncLumilakeClient``.

    from lumilake import LumilakeClient, AsyncLumilakeClient
    with LumilakeClient.from_config() as client:
        client.jobs.list()
"""

import logging

from lumilake._base_client import BaseAsyncClient, BaseClient, unwrap
from lumilake.async_client import AsyncLumilakeClient
from lumilake.client import LumilakeClient
from lumilake.config import DEFAULT_CONFIG_PATH, LumilakeConfig
from lumilake.errors import (
    DeployError,
    HttpError,
    LumilakeError,
    NotFoundError,
)
from lumilake.resources.deploy import SERVICE_NAMES, AsyncDeploy, Deploy
from lumilake.resources.info import AsyncInfo, Info
from lumilake.resources.jobs import AsyncJobs, Jobs
from lumilake.resources.traces import AsyncTraces, Traces
from lumilake.resources.workers import AsyncWorkers, Workers

__version__ = "0.1.2"

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "SERVICE_NAMES",
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
