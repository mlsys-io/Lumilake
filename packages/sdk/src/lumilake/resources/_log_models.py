"""Pydantic models for job task log payloads returned by the server."""

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
    """A single page of task logs, oldest-first."""

    job_id: str
    task_id: str
    entries: list[LogEntry] = Field(default_factory=list)
    next_cursor: str | None = None
    prev_cursor: str | None = None
