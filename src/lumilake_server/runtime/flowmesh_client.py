"""FlowMesh SDK client accessors.

One ``httpx.AsyncClient`` is pooled per event loop. Outgoing auth is staged
per-task in ``_outgoing_token_var`` by ``flowmesh_for_token`` and applied two
ways: the request event hook sets ``Authorization`` for non-streaming calls,
and the SDK's ``api_key`` argument carries it through the SSE streaming path.
"""

import asyncio
import contextvars
import threading
from urllib.parse import urlsplit

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


_DEFAULT_API_ORIGIN = "https://lum.id"
_DEFAULT_PORTS = {"https": 443, "http": 80}


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    host = parts.hostname
    if not host:
        raise ValueError(f"invalid API endpoint origin (no host): {url!r}")
    port = parts.port
    if port is None:
        port = _DEFAULT_PORTS.get(parts.scheme.lower())
    return parts.scheme.lower(), host.lower(), port


def _trusted_origins() -> set[tuple[str, str, int | None]]:
    origins = {_origin(_DEFAULT_API_ORIGIN)}
    raw = envs.LUMILAKE_API_TRUSTED_ORIGINS.strip()
    if raw:
        origins.update(_origin(item.strip()) for item in raw.split(",") if item.strip())
    return origins


def is_api_origin_trusted(url: str) -> bool:
    """Whether ``url``'s origin is in the API trusted-origins allowlist."""
    return _origin(url) in _trusted_origins()


def resolve_api_credential(url: str) -> str | None:
    """Return the server PAT for a trusted API endpoint origin, else ``None``.

    The scheduler credential is only ever attached to an allowlisted origin;
    a caller-selected endpoint must carry its own credential. This is the only
    place ``envs.RUNTIME_TOKEN`` is read for API-mode calls.
    """
    if is_api_origin_trusted(url):
        return envs.RUNTIME_TOKEN
    return None


async def close_current_loop_http_client() -> None:
    """Close the httpx client bound to the running event loop."""
    loop_id = id(asyncio.get_running_loop())
    with _http_clients_lock:
        client = _http_clients.pop(loop_id, None)
    if client is not None:
        await client.aclose()
