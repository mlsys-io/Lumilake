"""Shared FlowMesh SDK client accessor.

The SDK's ``AsyncFlowMesh`` client holds an HTTP connection pool bound to the
event loop of the first coroutine that uses it. When the server is restarted
(e.g. uvicorn worker reload) or the event loop changes, the cached client
must be discarded.

This helper centralizes the per-event-loop caching so both the FastAPI routes
layer and the runtime manager use one consistent pattern. Callers only need
to call :func:`get_async_client` — the helper handles recycling.
"""

import asyncio

import httpx
from flowmesh import AsyncFlowMesh
from lumilake import envs

_client: AsyncFlowMesh | None = None
_loop_id: int | None = None


def get_async_client() -> AsyncFlowMesh:
    """Return a process-wide ``AsyncFlowMesh`` instance bound to the current loop.

    The client is recreated when the running event loop changes (which
    happens across uvicorn reloads or tests that spin up fresh loops).
    """
    global _client, _loop_id
    current_loop_id = id(asyncio.get_running_loop())
    if _client is None or _loop_id != current_loop_id:
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(envs.LUMILAKE_HTTP_TIMEOUT_SECONDS),
            limits=httpx.Limits(keepalive_expiry=0.0),
        )
        _client = AsyncFlowMesh(
            base_url=envs.RUNTIME_ORCHESTRATOR_URL,
            api_key=envs.RUNTIME_TOKEN,
            http_client=http_client,
        )
        _loop_id = current_loop_id
    return _client
