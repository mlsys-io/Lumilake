import contextvars
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from fastapi import HTTPException, Request, status
from lumid_hooks import PrincipalContext, ResourceRef
from lumilake import envs
from lumilake_hook import ResourceAction, ResourceKind, UsageRow

from . import (
    IDENTITY_PROVIDERS,
    PERMISSION_CHECKERS,
    RESOURCE_REGISTRARS,
    SUBMISSION_GUARDS,
    USAGE_SINKS,
)

# Child tasks spawned with ``asyncio.create_task`` inherit this var via the
# context snapshot taken at spawn time.
runtime_token_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "lumilake_runtime_token", default=None
)


def default_principal() -> PrincipalContext:
    return PrincipalContext(
        principal_id="admin",
        org_id="default",
        external_id="local",
        principal_type="admin",
        scopes=["*"],
    )


async def authenticate_token(
    raw_token: str,
    logger: logging.Logger,
) -> PrincipalContext:
    if not IDENTITY_PROVIDERS:
        if envs.LUMILAKE_REQUIRE_IDENTITY_PROVIDER:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "No IdentityProvider plugins registered; "
                    "LUMILAKE_REQUIRE_IDENTITY_PROVIDER is set."
                ),
            )
        return default_principal()

    for provider in IDENTITY_PROVIDERS:
        resolved = await provider.resolve(raw_token, logger)
        if resolved is not None:
            return resolved

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No identity provider accepted the token",
    )


async def authenticate_request(request: Request) -> PrincipalContext:
    auth_header = request.headers.get("Authorization", "")
    raw_token = (
        auth_header.removeprefix("Bearer ") if auth_header.startswith("Bearer ") else ""
    )
    principal = await authenticate_token(raw_token, request.app.state.logger)
    token = raw_token or None
    request.state.runtime_token = token
    runtime_token_var.set(token)
    return principal


def get_runtime_token(request: Request) -> str | None:
    """Return the bearer captured by :func:`authenticate_request`.

    The same token Lumilake authenticated against is forwarded to FlowMesh on
    the principal's behalf so dispatch and billing line up on the runtime
    side. Returns ``None`` when the request arrived without an
    ``Authorization`` header — the local-deploy path.
    """
    token = request.state.runtime_token
    return token if isinstance(token, str) else None


async def run_submission_guards(
    principal: PrincipalContext,
    logger: logging.Logger,
) -> None:
    for guard in SUBMISSION_GUARDS:
        await guard.check(principal, logger)


async def resolve_accessible_ids(
    principal: PrincipalContext,
    resource_kind: ResourceKind,
    action: ResourceAction,
    logger: logging.Logger,
) -> frozenset[str] | None:
    accumulated: frozenset[str] | None = None
    for checker in PERMISSION_CHECKERS:
        result = await checker.accessible_ids(
            principal,
            resource_kind.value,
            action.value,
            logger,
        )
        if result is None:
            continue
        accumulated = result if accumulated is None else accumulated & result
    return accumulated


async def require_permission(
    principal: PrincipalContext,
    resource_kind: ResourceKind,
    resource_id: str | None,
    action: ResourceAction,
    logger: logging.Logger,
) -> None:
    resource = ResourceRef(kind=resource_kind.value, id=resource_id)
    for checker in PERMISSION_CHECKERS:
        await checker.require(principal, resource, action.value, logger)


async def register_resource(
    principal: PrincipalContext,
    resource_kind: ResourceKind,
    resource_id: str,
    metadata: Mapping[str, Any],
    logger: logging.Logger,
) -> None:
    resource = ResourceRef(kind=resource_kind.value, id=resource_id, metadata=metadata)
    for registrar in RESOURCE_REGISTRARS:
        await registrar.register(principal, resource, logger)


async def deregister_resource(
    principal: PrincipalContext,
    resource_kind: ResourceKind,
    resource_id: str,
    logger: logging.Logger,
) -> None:
    resource = ResourceRef(kind=resource_kind.value, id=resource_id)
    for registrar in RESOURCE_REGISTRARS:
        await registrar.deregister(principal, resource, logger)


async def emit_usage(
    rows: Sequence[UsageRow],
    logger: logging.Logger,
) -> None:
    for sink in USAGE_SINKS:
        try:
            await sink.emit(rows, logger)
        except Exception:
            logger.exception("Usage sink %s failed", sink.name)
