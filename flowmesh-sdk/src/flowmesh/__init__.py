"""FlowMesh Python SDK.

Provides typed sync and async clients for the FlowMesh Server REST API.

Quick start::

    from flowmesh import FlowMesh

    client = FlowMesh(base_url="https://kv.run:8000/flowmesh", api_key="flm-...")

    # Submit a workflow
    resp = client.workflows.submit(open("workflow.yaml").read())
    print(resp.workflow_id)

    # List workers
    for w in client.workers.list():
        print(w.id, w.status)

Async usage::

    from flowmesh import AsyncFlowMesh

    async with AsyncFlowMesh(base_url="...", api_key="...") as client:
        workflows = await client.workflows.list()
"""

from ._version import __version__
from .async_client import AsyncFlowMesh
from .client import FlowMesh
from .config import FlowMeshConfig
from .exceptions import (
    APIError,
    AuthenticationError,
    ConfigInvalidError,
    ConfigNotFoundError,
    FlowMeshConnectionError,
    FlowMeshError,
    NotFoundError,
    ValidationError,
)

__all__ = [
    "FlowMesh",
    "AsyncFlowMesh",
    "FlowMeshConfig",
    "FlowMeshError",
    "APIError",
    "AuthenticationError",
    "NotFoundError",
    "ValidationError",
    "FlowMeshConnectionError",
    "ConfigNotFoundError",
    "ConfigInvalidError",
    "__version__",
]
