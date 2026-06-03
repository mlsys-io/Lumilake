"""Optimizer routes: schedule-protocol endpoint and type listing."""

import asyncio

from fastapi import APIRouter, Depends, Request
from lumid_hooks import PrincipalContext
from lumilake_hook import ResourceAction, ResourceKind

from lumilake_server.hooks.security import (
    authenticate_request,
    require_permission,
    run_submission_guards,
)
from lumilake_server.runtime.optimizer import (
    OPTIMIZER_PROVIDERS,
    OPTIMIZER_TYPES,
    create_optimizer,
)
from lumilake_server.runtime.optimizer.schemas import (
    OptimizerListResponse,
    ScheduleRequest,
    ScheduleResponse,
)
from lumilake_server.runtime.runtime_graph import RuntimeGraph

router = APIRouter(prefix="/optimizer", tags=["Optimizer"])


@router.post("/schedule", response_model=ScheduleResponse)
async def schedule(
    request: Request,
    body: ScheduleRequest,
    principal: PrincipalContext = Depends(authenticate_request),
) -> ScheduleResponse:
    """Generate an optimizer schedule for the supplied runtime graph."""
    hook_logger = request.app.state.logger
    await require_permission(
        principal, ResourceKind.JOB, None, ResourceAction.WRITE, hook_logger
    )
    await run_submission_guards(principal, hook_logger)
    graph = RuntimeGraph.deserialize(body.graph.model_dump())
    optimizer = create_optimizer(optimizer_type=body.optimizer_type)
    schedule_result = await asyncio.to_thread(
        optimizer.generate_schedule,
        graph,
        body.worker_names,
        body.worker_profiles,
        body.data_profile_results,
    )
    return ScheduleResponse(worker_assignment=schedule_result.worker_assignment)


@router.get("", response_model=OptimizerListResponse)
async def list_optimizer_types(
    _principal: PrincipalContext = Depends(authenticate_request),
) -> OptimizerListResponse:
    """Return all optimizer types available on this server.

    Includes both locally registered types (``OPTIMIZER_TYPES``) and any
    types advertised by registered ``OptimizerProvider`` plugins.
    """
    types: list[str] = list(OPTIMIZER_TYPES.keys())
    for provider in OPTIMIZER_PROVIDERS:
        for t in provider.list_optimizers():
            lowered = t.lower()
            if lowered not in types:
                types.append(lowered)
    return OptimizerListResponse(types=sorted(types))
