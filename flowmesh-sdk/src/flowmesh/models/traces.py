"""Trace analyzer response payload types as seen by the SDK.

These describe the wire shape returned by
``GET /traces/workflows/analyze/{workflow_id}``.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class _ProfileBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssetSummary(_ProfileBase):
    asset_guid: str
    latest_data_id: str
    latest_version: int
    user_id: str
    versions: int
    created_at: str | None = None


class LineageEdge(_ProfileBase):
    data_id: str
    source_data_id: str
    created_at: str | None = None


class EventSummary(_ProfileBase):
    """Per-event-type duration aggregates as parallel lists."""

    event_type: list[str]
    count: list[int]
    total_seconds: list[float]
    avg_seconds: list[float]
    min_seconds: list[float]
    max_seconds: list[float]


class E2EBreakdown(_ProfileBase):
    hardware_summary: EventSummary
    network_summary: EventSummary
    workflow_duration_seconds: float
    total_network_seconds: float


class ActiveWaitBreakdown(_ProfileBase):
    data_id: list[str]
    active_seconds: list[float]
    wait_seconds: list[float]


class TaskTiming(_ProfileBase):
    data_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    queuing_delay_seconds: float
    parent_data_ids: list[str]
    blocking_parent_data_id: str | None = None


class CriticalPathSummary(_ProfileBase):
    path: list[str]
    critical_path_seconds: float
    active_wait_breakdown: ActiveWaitBreakdown
    hardware_summary: EventSummary
    network_summary: EventSummary
    total_network_seconds: float


class ProfileSummary(_ProfileBase):
    workflow_id: str | None = None
    event_count: int
    data_ids: list[str]
    assets: list[AssetSummary]
    lineage: list[LineageEdge]
    e2e_breakdown: E2EBreakdown
    per_data_id: list[TaskTiming]
    critical_path: CriticalPathSummary | None = None
