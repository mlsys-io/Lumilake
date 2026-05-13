import datetime as dt
from typing import TypedDict

from lumid_hooks import UsageSink


class UsageRow(TypedDict, total=False):
    org_id: str
    principal_id: str
    job_id: str
    status: str
    submitted_at: str
    started_at: str | None
    finished_at: str | None
    optimization_seconds: float | None
    trace_ids: list[str]
    emitted_at: dt.datetime


type LumilakeUsageSink = UsageSink[UsageRow]
