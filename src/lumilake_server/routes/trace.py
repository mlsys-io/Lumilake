"""Trace forward endpoints — thin wrapper around FlowMesh's analyzer.

Job records carry the FlowMesh workflow ids produced by each run (written
by ``routes/jobs.py`` at finalize). The analyzer payload is imported from
FlowMesh on demand.
"""

import asyncio
import datetime as dt
from collections import defaultdict
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from flowmesh.models.traces import ProfileSummary
from lumid_hooks import PrincipalContext
from lumilake_hook import ResourceAction, ResourceKind
from pydantic import BaseModel, Field

from lumilake_server.hooks.security import (
    authenticate_request,
    require_permission,
    resolve_accessible_ids,
)
from lumilake_server.runtime.flowmesh_client import get_async_client
from lumilake_server.utils.job_storage import (
    InMemoryJobStorage,
    JobStorage,
    get_job_storage,
)

router = APIRouter(prefix="/trace", tags=["Trace"])

TraceStatus = Literal["ok", "error"]


class TraceListItem(BaseModel):
    trace_id: str
    status: TraceStatus = Field(
        description=(
            "Trace availability: ``ok`` when at least one source job "
            "finalized successfully; ``error`` when every source job failed."
        )
    )
    synced_at: dt.datetime | None = None
    source_job_ids: list[str] = Field(default_factory=list)
    job_count: int
    error: str | None = None


class TraceListPayload(BaseModel):
    items: list[TraceListItem] = Field(default_factory=list)
    count: int


class TraceListResponse(BaseModel):
    ok: bool
    data: TraceListPayload


class TracePayload(BaseModel):
    trace_id: str = Field(description="FlowMesh workflow id.")
    status: TraceStatus = Field(
        description=(
            "Trace status: ``ok`` when FlowMesh returned a payload, "
            "``error`` when the fetch failed."
        )
    )
    trace: ProfileSummary | None = Field(
        default=None,
        description="FlowMesh trace analysis payload, forwarded as-is.",
    )
    error: str | None = Field(default=None, description="Trace error, if any.")


class TraceResponse(BaseModel):
    ok: bool = Field(description="Whether the request succeeded.")
    data: TracePayload = Field(description="Trace payload wrapper.")


def _iter_job_records(
    storage: JobStorage | None = None,
    *,
    org_id: str,
    job_ids: frozenset[str] | None,
    page_size: int = 200,
    max_pages: int = 25,
) -> list[dict[str, Any]]:
    """Load full job records, newest-first.

    Bounded by ``page_size * max_pages`` — 5000 jobs by default. Reading
    each record is one MinIO object fetch (``PersistentJobStorage.load``),
    so this is deliberately capped. The trace UI lists the most recent
    N synced traces; unbounded pagination would cost too much per hit.
    """
    if storage is None:
        storage = get_job_storage()
    records: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        summaries, total = storage.list_summaries(
            org_id=org_id,
            user_id=None,
            job_ids=job_ids,
            statuses=None,
            page=page,
            page_size=page_size,
        )
        if not summaries:
            break
        for summary in summaries:
            record = storage.load(str(summary["job_id"]))
            if record is None:
                continue
            records.append(record)
        if page * page_size >= total:
            break
    return records


async def _collect_job_records(
    *,
    org_id: str,
    job_ids: frozenset[str] | None,
) -> list[dict[str, Any]]:
    storage = get_job_storage()
    if isinstance(storage, InMemoryJobStorage):
        return _iter_job_records(storage, org_id=org_id, job_ids=job_ids)
    return await asyncio.to_thread(
        _iter_job_records,
        org_id=org_id,
        job_ids=job_ids,
    )


def _trace_ids_from_record(record: dict[str, Any]) -> list[str]:
    raw = record.get("trace_ids") or []
    if not isinstance(raw, list):
        return []
    return [str(t).strip() for t in raw if isinstance(t, str) and t.strip()]


def _finished_at(record: dict[str, Any]) -> dt.datetime | None:
    value = record.get("finished_at")
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


async def _fetch_profile(workflow_id: str) -> tuple[ProfileSummary | None, str | None]:
    """Forward to FlowMesh's analyzer; return ``(payload, error)``."""
    try:
        return await get_async_client().traces.analyze(workflow_id), None
    except Exception as exc:  # noqa: BLE001 — surface remote failures as "error"
        return None, f"trace fetch failed: {exc}"


@router.get(
    "",
    summary="List execution traces",
    description=(
        "List FlowMesh workflow ids recorded on completed jobs. Sourced "
        "from ``JobStorage``; FlowMesh is not contacted here — use "
        "``GET /trace/{trace_id}`` to fetch a payload."
    ),
    response_model=TraceListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_execution_traces(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    principal: PrincipalContext = Depends(authenticate_request),
) -> dict[str, Any]:
    hook_logger = request.app.state.logger
    await require_permission(
        principal,
        ResourceKind.TRACE,
        None,
        ResourceAction.READ,
        hook_logger,
    )
    await require_permission(
        principal,
        ResourceKind.JOB,
        None,
        ResourceAction.READ,
        hook_logger,
    )
    readable_job_ids = await resolve_accessible_ids(
        principal,
        ResourceKind.JOB,
        ResourceAction.READ,
        hook_logger,
    )
    records = await _collect_job_records(
        org_id=principal.org_id,
        job_ids=readable_job_ids,
    )
    readable_trace_ids = await resolve_accessible_ids(
        principal,
        ResourceKind.TRACE,
        ResourceAction.READ,
        hook_logger,
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for trace_id in _trace_ids_from_record(record):
            if readable_trace_ids is not None and trace_id not in readable_trace_ids:
                continue
            grouped[trace_id].append(record)

    items: list[TraceListItem] = []
    for trace_id, trace_records in grouped.items():
        ok_records = [r for r in trace_records if r.get("status") == "completed"]
        normalized_status: TraceStatus = "ok" if ok_records else "error"
        head = ok_records[0] if ok_records else trace_records[0]
        source_jobs = sorted(
            {
                str(rec.get("job_id", "")).strip()
                for rec in trace_records
                if str(rec.get("job_id", "")).strip()
            }
        )
        items.append(
            TraceListItem(
                trace_id=trace_id,
                status=normalized_status,
                synced_at=_finished_at(head),
                source_job_ids=source_jobs,
                job_count=len(source_jobs),
                error=(
                    str(head.get("error")).strip() or None
                    if head.get("error") is not None
                    else None
                ),
            )
        )
    items = sorted(
        items,
        key=lambda item: item.synced_at or dt.datetime.min.replace(tzinfo=dt.UTC),
        reverse=True,
    )[:limit]
    return {
        "ok": True,
        "data": {
            "items": [item.model_dump(mode="json") for item in items],
            "count": len(items),
        },
    }


@router.get(
    "/{trace_id}",
    summary="Get FlowMesh trace analysis",
    description=(
        "Forward a FlowMesh workflow id to FlowMesh's trace analyzer and "
        "return the ``ProfileSummary`` payload as-is. The requested "
        "``trace_id`` must match a FlowMesh workflow recorded on a job."
    ),
    response_description="FlowMesh trace analysis payload.",
    status_code=status.HTTP_200_OK,
    response_model=TraceResponse,
    openapi_extra={
        "responses": {
            "404": {"description": "Trace id not found"},
        },
    },
)
async def get_execution_trace(
    request: Request,
    trace_id: str,
    principal: PrincipalContext = Depends(authenticate_request),
) -> TraceResponse:
    hook_logger = request.app.state.logger
    await require_permission(
        principal,
        ResourceKind.JOB,
        None,
        ResourceAction.READ,
        hook_logger,
    )
    readable_job_ids = await resolve_accessible_ids(
        principal,
        ResourceKind.JOB,
        ResourceAction.READ,
        hook_logger,
    )
    records = await _collect_job_records(
        org_id=principal.org_id,
        job_ids=readable_job_ids,
    )
    source_records = [
        record for record in records if trace_id in _trace_ids_from_record(record)
    ]
    if not source_records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="trace id not found",
        )
    await require_permission(
        principal,
        ResourceKind.TRACE,
        trace_id,
        ResourceAction.READ,
        hook_logger,
    )
    # Forward whether or not source jobs are completed: FlowMesh records spans
    # as they happen, so a partially-failed run still has a fetchable trace —
    # which is exactly when an operator needs it for debugging.
    payload, err = await _fetch_profile(trace_id)
    trace_status: TraceStatus = "ok" if payload is not None and err is None else "error"
    return TraceResponse(
        ok=trace_status == "ok",
        data=TracePayload(
            trace_id=trace_id,
            status=trace_status,
            trace=payload,
            error=err,
        ),
    )
