"""Asynchronous FlowMesh client."""

import httpx

from ._base_client import BaseAsyncClient, resolve_config
from ._constants import DEFAULT_TIMEOUT
from .resources.nodes import AsyncNodes
from .resources.results import AsyncResults
from .resources.ssh import AsyncSSH
from .resources.system import AsyncSystem
from .resources.tasks import AsyncTasks
from .resources.traces import AsyncTraces
from .resources.workers import AsyncWorkers
from .resources.workflows import AsyncWorkflows


class AsyncFlowMesh(BaseAsyncClient):
    """Asynchronous FlowMesh API client.

    Usage::

        from flowmesh import AsyncFlowMesh

        async with AsyncFlowMesh(
            base_url="https://kv.run:8000/flowmesh",
            api_key="flm-xxxx-...",
        ) as client:
            result = await client.workflows.submit(
                open("workflow.yaml").read()
            )
            workers = await client.workers.list()

    Configuration resolution is the same as :class:`FlowMesh`.
    """

    workflows: AsyncWorkflows
    tasks: AsyncTasks
    results: AsyncResults
    workers: AsyncWorkers
    nodes: AsyncNodes
    ssh: AsyncSSH
    system: AsyncSystem
    traces: AsyncTraces

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        cfg = resolve_config(base_url, api_key)
        super().__init__(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=timeout,
            http_client=http_client,
        )
        self.workflows = AsyncWorkflows(self)
        self.tasks = AsyncTasks(self)
        self.results = AsyncResults(self)
        self.workers = AsyncWorkers(self)
        self.nodes = AsyncNodes(self)
        self.ssh = AsyncSSH(self)
        self.system = AsyncSystem(self)
        self.traces = AsyncTraces(self)
