from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from lumilake.utils.job_storage import InMemoryJobStorage


@dataclass
class _Record:
    job_id: str
    org_id: str
    user_id: str
    status: str
    submitted_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    inputs: dict[str, object] = field(default_factory=dict)
    output_location: dict[str, object] = field(default_factory=dict)
    progress: dict[str, object] = field(default_factory=dict)
    result: dict[str, object] | None = None


def _record(job_id: str, status: str, submitted_at: str) -> _Record:
    return _Record(
        job_id=job_id,
        org_id="test-org",
        user_id="test-user",
        status=status,
        submitted_at=submitted_at,
        output_location={"graph": {"type": "s3", "prefix": "out.txt"}},
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
    assert [item["job_id"] for item in items] == ["job-2", "job-1"]


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
    assert [item["job_id"] for item in page_1] == ["job-3", "job-2"]
    assert [item["job_id"] for item in page_2] == ["job-1"]


def test_summary_validation_rejects_invalid_status() -> None:
    storage = InMemoryJobStorage()
    with pytest.raises(ValidationError):
        storage.save(_record("job-bad", "queued", "2026-02-20T00:00:00+00:00"))


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
    assert [item["job_id"] for item in page_1] == ["job-3"]
    assert [item["job_id"] for item in page_2] == ["job-1"]
