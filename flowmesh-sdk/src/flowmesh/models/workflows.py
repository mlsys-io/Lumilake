"""Workflow-related models."""

from pydantic import BaseModel

from .common import TaskStatus, WorkflowStatus


class WorkflowSubmitTaskEntry(BaseModel):
    task_id: str
    status: TaskStatus | None = None
    assigned_worker: str | None = None
    topic: str | None = None
    waiting_on: list[str] | None = None
    depends_on: list[str] | None = None
    attempts: int | None = None
    max_attempts: int | None = None
    load: int | None = None


class WorkflowSubmitResponse(BaseModel):
    ok: bool
    workflow_id: str
    count: int
    tasks: list[WorkflowSubmitTaskEntry]


class WorkflowValidateTaskEntry(BaseModel):
    task_id: str
    graph_node_name: str | None = None
    depends_on: list[str]


class WorkflowValidateResponse(BaseModel):
    ok: bool
    count: int
    tasks: list[WorkflowValidateTaskEntry]


class Workflow(BaseModel):
    workflow_id: str
    task_ids: list[str]
    submitted_at: str
    updated_at: str
    status: WorkflowStatus
    dispatched_tasks: list[str]
    completed_tasks: list[str]
    failed_tasks: list[str]
    cancelled_tasks: list[str]
