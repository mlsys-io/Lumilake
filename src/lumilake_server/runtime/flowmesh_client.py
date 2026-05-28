"""FlowMesh SDK client accessors.

One ``httpx.AsyncClient`` is pooled per event loop. Outgoing auth is staged
per-task in ``_outgoing_token_var`` by ``flowmesh_for_token`` and applied two
ways: the request event hook sets ``Authorization`` for non-streaming calls,
and the SDK's ``api_key`` argument carries it through the SSE streaming path.
"""

import asyncio
import contextvars
import threading

import httpx
from fastapi import Request
from flowmesh import AsyncFlowMesh
from lumilake import envs

from lumilake_server.hooks.security import get_runtime_token, runtime_token_var

_outgoing_token_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "lumilake_flowmesh_outgoing_token", default=None
)

# Threading lock because FastAPI and _AsyncRunner live on different threads.
_http_clients: dict[int, httpx.AsyncClient] = {}
_http_clients_lock = threading.Lock()


async def _inject_auth_header(request: httpx.Request) -> None:
    token = _outgoing_token_var.get()
    if token:
        request.headers["Authorization"] = f"Bearer {token}"


def _shared_http_client() -> httpx.AsyncClient:
    # httpx.AsyncClient is loop-bound; key the pool by running-loop id.
    loop_id = id(asyncio.get_running_loop())
    with _http_clients_lock:
        client = _http_clients.get(loop_id)
        if client is None:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(envs.LUMILAKE_HTTP_TIMEOUT_SECONDS),
                limits=httpx.Limits(keepalive_expiry=0.0),
                event_hooks={"request": [_inject_auth_header]},
            )
            _http_clients[loop_id] = client
        return client


def flowmesh_for(request: Request) -> AsyncFlowMesh:
    """Build an ``AsyncFlowMesh`` carrying the request's captured bearer."""
    return flowmesh_for_token(get_runtime_token(request))


def flowmesh_for_token(token: str | None) -> AsyncFlowMesh:
    """Build an ``AsyncFlowMesh`` carrying ``token`` as ``Authorization: Bearer``."""
    _outgoing_token_var.set(token)
    return AsyncFlowMesh(
        base_url=envs.RUNTIME_ORCHESTRATOR_URL,
        api_key=token or None,
        http_client=_shared_http_client(),
    )


def flowmesh_for_context() -> AsyncFlowMesh:
    """Build an ``AsyncFlowMesh`` carrying the current task's runtime token."""
    return flowmesh_for_token(runtime_token_var.get())


def flowmesh_for_server() -> AsyncFlowMesh:
    """Build an ``AsyncFlowMesh`` carrying the scheduler credential.

    Scheduler-internal only — route handlers must not call this.
    """
    return flowmesh_for_token(envs.RUNTIME_TOKEN)


async def close_current_loop_http_client() -> None:
    """Close the httpx client bound to the running event loop."""
    loop_id = id(asyncio.get_running_loop())
    with _http_clients_lock:
        client = _http_clients.pop(loop_id, None)
    if client is not None:
        await client.aclose()
