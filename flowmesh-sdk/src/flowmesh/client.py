"""Synchronous FlowMesh client."""

import httpx

from ._base_client import BaseClient, resolve_config
from ._constants import DEFAULT_TIMEOUT
from .resources.nodes import Nodes
from .resources.results import Results
from .resources.ssh import SSH
from .resources.system import System
from .resources.tasks import Tasks
from .resources.traces import Traces
from .resources.workers import Workers
from .resources.workflows import Workflows


class FlowMesh(BaseClient):
    """Synchronous FlowMesh API client.

    Usage::

        from flowmesh import FlowMesh

        client = FlowMesh(
            base_url="https://kv.run:8000/flowmesh",
            api_key="flm-xxxx-...",
        )

        # Submit a workflow
        result = client.workflows.submit(open("workflow.yaml").read())

        # List workers
        workers = client.workers.list()

        # Use as context manager
        with FlowMesh(base_url="...", api_key="...") as client:
            wf = client.workflows.retrieve("wf-123")

    Configuration resolution (in order of precedence):
        1. Explicit ``base_url`` / ``api_key`` parameters
        2. Environment variables: ``FLOWMESH_BASE_URL``, ``FLOWMESH_API_KEY``
        3. CLI config file: ``~/.flowmesh/config.toml``
        4. Default: ``http://localhost:8000``
    """

    workflows: Workflows
    tasks: Tasks
    results: Results
    workers: Workers
    nodes: Nodes
    ssh: SSH
    system: System
    traces: Traces

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: httpx.Client | None = None,
    ) -> None:
        cfg = resolve_config(base_url, api_key)
        super().__init__(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=timeout,
            http_client=http_client,
        )
        self.workflows = Workflows(self)
        self.tasks = Tasks(self)
        self.results = Results(self)
        self.workers = Workers(self)
        self.nodes = Nodes(self)
        self.ssh = SSH(self)
        self.system = System(self)
        self.traces = Traces(self)
