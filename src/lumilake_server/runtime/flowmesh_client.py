"""FlowMesh SDK client accessors."""

import asyncio

import httpx
from fastapi import Request
from flowmesh import AsyncFlowMesh
from lumilake import envs

from lumilake_server.hooks.security import get_runtime_token, runtime_token_var

_http_client: httpx.AsyncClient | None = None
_loop_id: int | None = None


def _shared_http_client() -> httpx.AsyncClient:
    # httpx.AsyncClient's connection pool is bound to the event loop of the
    # coroutine that first used it; recycle when the loop changes.
    global _http_client, _loop_id
    current_loop_id = id(asyncio.get_running_loop())
    if _http_client is None or _loop_id != current_loop_id:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(envs.LUMILAKE_HTTP_TIMEOUT_SECONDS),
            limits=httpx.Limits(keepalive_expiry=0.0),
        )
        _loop_id = current_loop_id
    return _http_client


def flowmesh_for(request: Request) -> AsyncFlowMesh:
    """Build an ``AsyncFlowMesh`` carrying the request's captured bearer."""
    return flowmesh_for_token(get_runtime_token(request))


def flowmesh_for_token(token: str | None) -> AsyncFlowMesh:
    """Build an ``AsyncFlowMesh`` carrying ``token`` as the FlowMesh API key."""
    return AsyncFlowMesh(
        base_url=envs.RUNTIME_ORCHESTRATOR_URL,
        api_key=token or None,
        http_client=_shared_http_client(),
    )


def flowmesh_for_context() -> AsyncFlowMesh:
    """Build an ``AsyncFlowMesh`` carrying the current task's runtime token."""
    return flowmesh_for_token(runtime_token_var.get())


async def close_shared_http_client() -> None:
    global _http_client, _loop_id
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
        _loop_id = None
