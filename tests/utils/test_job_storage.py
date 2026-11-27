from typing import Any

import pytest

from lumilake_server.records import JobRecord, JobStatus
from lumilake_server.schemas.io import S3Location
from lumilake_server.utils.job_storage import InMemoryJobStorage


def _record(job_id: str, status: JobStatus, submitted_at: str) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        org_id="test-org",
        user_id="test-user",
        status=status,
        submitted_at=submitted_at,
        inputs={},
        output_location={"graph": S3Location(type="s3", prefix="out.txt")},
    )


def test_list_summaries_filters_by_status() -> None:
    storage = InMemoryJobStorage()
    storage.save(_record("job-1", "pending", "2026-02-20T00:00:00+00:00"))
    storage.save(_record("job-2", "completed", "2026-02-21T00:00:00+00:00"))
    storage.save(_record("job-3", "running", "2026-02-22T00:00:00+00:00"))

    items, total = storage.list_summaries(
        org_id="test-org",
        user_id=None,
        job_ids=None,
        statuses={"completed", "pending"},
        page=1,
        page_size=20,
    )

    assert total == 2
    assert [item.job_id for item in items] == ["job-2", "job-1"]


def test_list_summaries_paginates_descending_by_submission_time() -> None:
    storage = InMemoryJobStorage()
    storage.save(_record("job-1", "pending", "2026-02-20T00:00:00+00:00"))
    storage.save(_record("job-2", "pending", "2026-02-21T00:00:00+00:00"))
    storage.save(_record("job-3", "pending", "2026-02-22T00:00:00+00:00"))

    page_1, total = storage.list_summaries(
        org_id="test-org",
        user_id=None,
        job_ids=None,
        statuses=None,
        page=1,
        page_size=2,
    )
    page_2, _ = storage.list_summaries(
        org_id="test-org",
        user_id=None,
        job_ids=None,
        statuses=None,
        page=2,
        page_size=2,
    )

    assert total == 3
    assert [item.job_id for item in page_1] == ["job-3", "job-2"]
    assert [item.job_id for item in page_2] == ["job-1"]


def test_record_rejects_invalid_status() -> None:
    payload: dict[str, Any] = {
        "job_id": "job-bad",
        "status": "queued",
        "submitted_at": "2026-02-20T00:00:00+00:00",
        "inputs": {},
        "output_location": {"graph": S3Location(type="s3", prefix="out.txt")},
    }
    with pytest.raises(ValueError, match="unknown job status"):
        JobRecord(**payload)


def test_list_summaries_filters_by_job_ids_before_pagination() -> None:
    storage = InMemoryJobStorage()
    storage.save(_record("job-1", "pending", "2026-02-20T00:00:00+00:00"))
    storage.save(_record("job-2", "pending", "2026-02-21T00:00:00+00:00"))
    storage.save(_record("job-3", "pending", "2026-02-22T00:00:00+00:00"))

    page_1, total = storage.list_summaries(
        org_id="test-org",
        user_id=None,
        job_ids=frozenset({"job-1", "job-3"}),
        statuses=None,
        page=1,
        page_size=1,
    )
    page_2, _ = storage.list_summaries(
        org_id="test-org",
        user_id=None,
        job_ids=frozenset({"job-1", "job-3"}),
        statuses=None,
        page=2,
        page_size=1,
    )

    assert total == 2
    assert [item.job_id for item in page_1] == ["job-3"]
    assert [item.job_id for item in page_2] == ["job-1"]


def test_iter_summaries_filters_by_status() -> None:
    storage = InMemoryJobStorage()
    storage.save(_record("job-1", "pending", "2026-02-20T00:00:00+00:00"))
    storage.save(_record("job-2", "running", "2026-02-21T00:00:00+00:00"))
    storage.save(_record("job-3", "completed", "2026-02-22T00:00:00+00:00"))

    in_flight = {s.job_id for s in storage.iter_summaries({"pending", "running"})}
    assert in_flight == {"job-1", "job-2"}

    all_ids = {s.job_id for s in storage.iter_summaries()}
    assert all_ids == {"job-1", "job-2", "job-3"}
