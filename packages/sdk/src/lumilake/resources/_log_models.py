"""Pydantic models for job workflow log payloads returned by the server."""

from typing import Any

from pydantic import BaseModel, Field


class LogEvent(BaseModel):
    """Structured log event fields, mirroring the server-side payload."""

    ts: str | None = None
    workflow_id: str | None = None
    task_id: str | None = None
    worker_id: str | None = None
    node_id: str | None = None
    level: str | None = None
    stream: str | None = None
    source: str | None = None
    message: str | None = None
    fields: dict[str, Any] | None = None


class LogEntry(BaseModel):
    """One log line plus the cursor it advances past."""

    cursor: str = Field(description="Opaque pagination cursor for this entry.")
    event: LogEvent = Field(description="Structured log event fields.")


class LogQueryResponse(BaseModel):
    """A single page of workflow logs, oldest-first."""

    job_id: str
    workflow_id: str
    entries: list[LogEntry] = Field(default_factory=list)
    next_cursor: str | None = None
    prev_cursor: str | None = None


class WorkflowUsage(BaseModel):
    """Token usage summed over a workflow's task results."""

    prompt_tokens: int = Field(description="Prompt token count.")
    completion_tokens: int = Field(description="Completion token count.")
    reasoning_tokens: int = Field(description="Reasoning token count.")
    calls: int = Field(description="Number of model calls.")
    failures: int = Field(description="Number of failed calls.")
    retries: int = Field(description="Number of retried calls.")
    truncated_calls: int = Field(description="Number of truncated calls.")
    wall_sec: float = Field(description="Wall-clock seconds.")


class JobWorkflowInfo(BaseModel):
    """Summary of one FlowMesh workflow associated with a job."""

    workflow_id: str = Field(description="FlowMesh workflow identifier.")
    status: str = Field(description="FlowMesh workflow status.")
    submitted_at: str | None = Field(
        default=None, description="ISO timestamp when the workflow was submitted."
    )
    task_count: int | None = Field(
        default=None, description="Total number of tasks in the workflow."
    )
    succeeded_count: int | None = Field(
        default=None, description="Number of succeeded tasks."
    )
    failed_count: int | None = Field(
        default=None, description="Number of failed tasks."
    )
    usage: WorkflowUsage | None = Field(
        default=None, description="Token usage summed over the workflow's task results."
    )
