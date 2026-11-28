"""The stored shape of a job: its record, its lifecycle status and its summary."""

import datetime as dt
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter

from lumilake_server.runtime.protocol import LumilakeResponse
from lumilake_server.schemas.dispatch import WorkflowDispatch
from lumilake_server.schemas.io import IOLocation
from lumilake_server.schemas.progress import JobProgress

JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
JOB_STATUS_VALUES: tuple[JobStatus, ...] = (
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
)
JOB_STATUS_DESCRIPTION = (
    "Job lifecycle status: `pending` (queued), `running` (executing), "
    "`completed` (finished successfully), `failed` (finished with error), "
    "`cancelled` (cancelled before completion)."
)

TERMINAL_JOB_STATUSES: frozenset[JobStatus] = frozenset(
    {"completed", "failed", "cancelled"}
)

IO_LOCATION_ADAPTER: TypeAdapter[IOLocation] = TypeAdapter(IOLocation)


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    submitted_at: str
    inputs: dict[str, dict[str, list[str]]]
    output_location: dict[str, IOLocation]
    org_id: str = "default"
    user_id: str = "local"
    seed: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    optimization_seconds: float | None = None
    selection_seconds: float | None = None
    clustering_seconds: float | None = None
    error: str | None = None
    progress: JobProgress = field(default_factory=JobProgress)
    terminal_usage_finalized: bool = False
    result: LumilakeResponse | None = None
    folder_inputs: dict[str, str] = field(default_factory=dict)
    trace_ids: list[str] = field(default_factory=list)
    job_task_ids: dict[str, list[str]] = field(default_factory=dict)
    workflow_dispatches: list[WorkflowDispatch] = field(default_factory=list)
    parent_job_id: str | None = None
    child_job_ids: list[str] = field(default_factory=list)
    lumid_job_id: str | None = None

    def __post_init__(self) -> None:
        if self.status not in JOB_STATUS_VALUES:
            raise ValueError(f"unknown job status {self.status!r}")
        if not self.output_location:
            raise ValueError("output_location is required")
        self.progress = JobProgress.model_validate(self.progress)
        if self.result is not None:
            self.result = LumilakeResponse.model_validate(self.result)
        self.workflow_dispatches = [
            WorkflowDispatch.model_validate(row) for row in self.workflow_dispatches
        ]
        self.output_location = {
            key: IO_LOCATION_ADAPTER.validate_python(loc)
            for key, loc in self.output_location.items()
        }


class JobSummary(BaseModel):
    job_id: str
    org_id: str
    user_id: str
    status: JobStatus
    submitted_at: dt.datetime
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    optimization_seconds: float | None = None
    selection_seconds: float | None = None
    clustering_seconds: float | None = None
    error: str | None = None
    parent_job_id: str | None = None

    model_config = ConfigDict(extra="forbid")
