from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from flowmesh.exceptions import APIError, AuthenticationError, NotFoundError
from lumid_hooks import PrincipalContext
from lumilake_hook import ResourceAction, ResourceKind

from lumilake_server.hooks.security import (
    authenticate_request,
    require_permission,
    resolve_accessible_ids,
)
from lumilake_server.runtime.flowmesh_client import get_async_client
from lumilake_server.schemas.worker import WorkerInfo

router = APIRouter(prefix="/workers", tags=["Workers"])


# Parameters that the FlowMesh SDK's ``workers.list`` accepts as kwargs.
# Anything outside this set is passed through via the SDK's ``query_params``
# escape hatch so unknown filters reach the upstream orchestrator untouched.
_KNOWN_LIST_PARAMS = {
    "worker_id",
    "alias",
    "namespace",
    "cluster",
    "status",
    "tags",
    "stale",
}


def _split_list_params(
    request: Request,
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Return ``(kwargs, extra_query)`` derived from the request's query params.

    Known filters become typed kwargs; everything else is preserved as raw
    query-string tuples so the upstream API receives identical filters to
    the ones it would have received from the pre-SDK pass-through code.
    """
    kwargs: dict[str, Any] = {}
    extras: list[tuple[str, str]] = []

    for key, value in request.query_params.multi_items():
        if key in _KNOWN_LIST_PARAMS:
            # Multi-valued params (``status``, ``tags``) accept a list.
            existing = kwargs.get(key)
            if existing is None:
                kwargs[key] = value
            elif isinstance(existing, list):
                existing.append(value)
            else:
                kwargs[key] = [existing, value]
        else:
            extras.append((key, value))
    return kwargs, extras


@router.get(
    "",
    summary="List workers",
    description="List all registered workers with optional filtering.",
    response_description="List of workers",
    response_model=list[WorkerInfo],
)
async def list_workers(
    request: Request,
    principal: PrincipalContext = Depends(authenticate_request),
) -> Any:
    hook_logger = request.app.state.logger
    await require_permission(
        principal,
        ResourceKind.WORKER,
        None,
        ResourceAction.READ,
        hook_logger,
    )
    readable_worker_ids = await resolve_accessible_ids(
        principal,
        ResourceKind.WORKER,
        ResourceAction.READ,
        hook_logger,
    )
    list_kwargs, extras = _split_list_params(request)
    try:
        workers = await get_async_client().workers.list(
            **list_kwargs,
            query_params=extras or None,
        )
    except (APIError, AuthenticationError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if readable_worker_ids is not None:
        workers = [worker for worker in workers if worker.id in readable_worker_ids]
    return [w.model_dump() for w in workers]


@router.get(
    "/{worker_id}",
    summary="Get a worker",
    description="Get worker information by ID.",
    response_description="Worker information",
    response_model=WorkerInfo,
)
async def get_worker(
    worker_id: str,
    request: Request,
    principal: PrincipalContext = Depends(authenticate_request),
) -> Any:
    hook_logger = request.app.state.logger
    await require_permission(
        principal,
        ResourceKind.WORKER,
        worker_id,
        ResourceAction.READ,
        hook_logger,
    )
    try:
        worker = await get_async_client().workers.retrieve(worker_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (APIError, AuthenticationError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return worker.model_dump()
