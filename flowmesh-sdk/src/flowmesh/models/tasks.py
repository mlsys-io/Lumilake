"""Task-related models."""

from typing import Any

from pydantic import BaseModel

from .common import TaskStatus


class HardwareUsage(BaseModel, extra="allow"):
    pass


class TaskUsage(BaseModel):
    started_at: str
    finished_at: str
    runtime_sec: float
    hardware: HardwareUsage
    cost_per_hour: float
    total_cost: float
    status: TaskStatus


class TaskInfo(BaseModel):
    task_id: str
    workflow_id: str
    owner_id: str
    org_id: str
    supplier_id: str
    source: str
    task: dict[str, Any]
    status: TaskStatus
    task_type: str | None = None
    category: str | None = None
    assigned_worker: str | None = None
    topic: str | None = None
    submitted_at: str
    submitted_ts: float
    dispatched_ts: float | None = None
    started_ts: float | None = None
    finished_ts: float | None = None
    usages: list[TaskUsage]
    error: str | None = None
    attempts: int
    max_attempts: int
    parent_task_id: str | None = None
    shard_index: int | None = None
    shard_total: int | None = None
    next_retry_at: str | None = None
    last_failed_worker: str | None = None
    last_error: str | None = None
    local_name: str | None = None
    graph_node_name: str | None = None
    load: int
    position_in_epoch: int | None = None
    selected_worker: list[str] | None = None
    merged_children: list[str] | None = None
    merged_parent_id: str | None = None
    merge_slice: dict[str, int] | None = None
    merge_key: str | None = None
    latest_update: dict[str, Any] | None = None
    depends_on: list[str]
    pending_dependencies: list[str]
    dependents: list[str]
    completed: bool
    failed: bool
