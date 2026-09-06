"""Behaviour tests for SqliteJobStorage.

The bar here is PARITY, not "sqlite works": every assertion that matters is
phrased against what PersistentJobStorage already does, because this backend is
swapped in behind the same interface and any divergence in filtering, ordering
or output-location semantics is a silent behaviour change in production.

Blob access goes through the same in-memory stub the persistent-storage tests
use, so nothing here touches the network.
"""

import datetime as dt
import json
import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

import lumilake_server.utils.job_storage as job_storage_module
from lumilake_server.utils.job_storage import (
    JobStorage,
    PersistentJobStorage,
    SqliteJobStorage,
)
from lumilake_server.utils.lumid_data_client import BlobNotFound


class _MemBlobStore:
    def __init__(self) -> None:
        self.store: dict[str, tuple[bytes, str]] = {}

    def put_blob(self, key: str, body: bytes, content_type: str) -> None:
        self.store[key] = (body, content_type)

    def get_blob(self, key: str) -> tuple[bytes, str]:
        if key not in self.store:
            raise BlobNotFound(key)
        return self.store[key]


@dataclass
class _Rec:
    job_id: str
    org_id: str = "test-org"
    user_id: str = "test-user"
    status: str = "pending"
    submitted_at: str = "2026-01-01T00:00:00+00:00"
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    output_location: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None


def _wire_blobs(prefix: str, monkeypatch: pytest.MonkeyPatch) -> _MemBlobStore:
    backend = _MemBlobStore()
    monkeypatch.setattr(job_storage_module.envs, "S3_ARCHIVE_PREFIX", prefix)
    monkeypatch.setattr(
        job_storage_module.lumid_data_client, "put_blob", backend.put_blob
    )
    monkeypatch.setattr(
        job_storage_module.lumid_data_client, "get_blob", backend.get_blob
    )
    return backend


def _sqlite(tmp_path, monkeypatch: pytest.MonkeyPatch, prefix: str = "arc"):
    backend = _wire_blobs(prefix, monkeypatch)
    monkeypatch.setattr(job_storage_module.envs, "LUMILAKE_JOB_INDEX_RETENTION_DAYS", 0)
    storage = SqliteJobStorage(db_path=str(tmp_path / "jobs.sqlite"))
    return storage, backend


def _blob(prefix: str, monkeypatch: pytest.MonkeyPatch) -> PersistentJobStorage:
    _wire_blobs(prefix, monkeypatch)
    storage = PersistentJobStorage.__new__(PersistentJobStorage)
    JobStorage.__init__(storage)
    storage.key_prefix = prefix.strip("/")
    storage._index_lock = threading.Lock()
    return storage


def test_record_blobs_still_written(tmp_path, monkeypatch) -> None:
    """Records/artifacts must keep their blob keys -- only indexes moved."""
    storage, backend = _sqlite(tmp_path, monkeypatch, "myarchive/2026")
    storage.save(_Rec(job_id="job-abc"))
    assert "myarchive/2026/job-abc/record.json" in backend.store
    assert "myarchive/2026/job-abc/inputs.json" in backend.store
    assert "myarchive/2026/job-abc/progress.json" in backend.store
    # the monolithic index is exactly what should NOT be written any more
    assert "myarchive/2026/jobs_index.json" not in backend.store


def test_save_and_load_round_trip(tmp_path, monkeypatch) -> None:
    storage, _ = _sqlite(tmp_path, monkeypatch)
    storage.save(_Rec(job_id="job-xyz", status="running"))
    loaded = storage.load("job-xyz")
    assert loaded is not None and loaded["status"] == "running"


def test_save_is_idempotent_upsert(tmp_path, monkeypatch) -> None:
    storage, _ = _sqlite(tmp_path, monkeypatch)
    storage.save(_Rec(job_id="j1", status="pending"))
    storage.save(_Rec(job_id="j1", status="completed"))
    rows, total = storage.list_summaries(
        org_id="test-org",
        user_id=None,
        job_ids=None,
        statuses=None,
        page=1,
        page_size=10,
    )
    assert total == 1
    assert rows[0]["status"] == "completed"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"user_id": None, "job_ids": None, "statuses": None},
        {"user_id": "u1", "job_ids": None, "statuses": None},
        {"user_id": None, "job_ids": frozenset({"j2"}), "statuses": None},
        {"user_id": None, "job_ids": None, "statuses": {"completed"}},
        {"user_id": "u1", "job_ids": None, "statuses": {"pending"}},
        {"user_id": None, "job_ids": frozenset(), "statuses": None},
    ],
)
def test_list_summaries_matches_blob_backend(tmp_path, monkeypatch, kwargs) -> None:
    """Filtering AND ordering must agree with the backend being replaced."""
    recs = [
        _Rec(
            job_id="j1",
            user_id="u1",
            status="pending",
            submitted_at="2026-01-03T00:00:00+00:00",
        ),
        _Rec(
            job_id="j2",
            user_id="u2",
            status="completed",
            submitted_at="2026-01-01T00:00:00+00:00",
        ),
        _Rec(
            job_id="j3",
            user_id="u1",
            status="completed",
            submitted_at="2026-01-02T00:00:00+00:00",
        ),
    ]
    sq, _ = _sqlite(tmp_path, monkeypatch)
    for r in recs:
        sq.save(r)
    bl = _blob("arc2", monkeypatch)
    for r in recs:
        bl.save(r)

    got = sq.list_summaries(org_id="test-org", page=1, page_size=10, **kwargs)
    want = bl.list_summaries(org_id="test-org", page=1, page_size=10, **kwargs)
    assert got == want


def test_pagination_matches_blob_backend(tmp_path, monkeypatch) -> None:
    recs = [
        _Rec(job_id=f"j{i}", submitted_at=f"2026-01-{i:02d}T00:00:00+00:00")
        for i in range(1, 8)
    ]
    sq, _ = _sqlite(tmp_path, monkeypatch)
    bl = _blob("arc2", monkeypatch)
    for r in recs:
        sq.save(r)
        bl.save(r)
    for page in (1, 2, 3, 4):
        assert sq.list_summaries(
            org_id="test-org",
            user_id=None,
            job_ids=None,
            statuses=None,
            page=page,
            page_size=3,
        ) == bl.list_summaries(
            org_id="test-org",
            user_id=None,
            job_ids=None,
            statuses=None,
            page=page,
            page_size=3,
        )


def test_iter_summaries_status_filter(tmp_path, monkeypatch) -> None:
    storage, _ = _sqlite(tmp_path, monkeypatch)
    storage.save(_Rec(job_id="a", status="running"))
    storage.save(_Rec(job_id="b", status="completed"))
    assert {s.job_id for s in storage.iter_summaries()} == {"a", "b"}
    assert {s.job_id for s in storage.iter_summaries({"running"})} == {"a"}


def test_output_location_reserve_conflicts(tmp_path, monkeypatch) -> None:
    storage, _ = _sqlite(tmp_path, monkeypatch)
    storage.save(_Rec(job_id="owner", status="running"))
    storage.reserve_output_location("s3://bucket/out", "owner")
    with pytest.raises(ValueError):
        storage.reserve_output_location("s3://bucket/out", "intruder")


def test_output_location_reassigned_when_owner_completed(tmp_path, monkeypatch) -> None:
    storage, _ = _sqlite(tmp_path, monkeypatch)
    storage.save(_Rec(job_id="owner", status="completed"))
    storage.reserve_output_location("s3://bucket/out", "owner")
    storage.reserve_output_location("s3://bucket/out", "next")  # must not raise


def test_output_location_release_is_owner_scoped(tmp_path, monkeypatch) -> None:
    storage, _ = _sqlite(tmp_path, monkeypatch)
    storage.save(_Rec(job_id="owner", status="running"))
    storage.reserve_output_location("k", "owner")
    storage.release_output_location("k", "someone-else")  # no-op
    with pytest.raises(ValueError):
        storage.reserve_output_location("k", "intruder")
    storage.release_output_location("k", "owner")
    storage.reserve_output_location("k", "intruder")  # now free


def test_backfill_imports_legacy_indexes(tmp_path, monkeypatch) -> None:
    """A first start on an existing archive must adopt the blob history."""
    backend = _wire_blobs("arc", monkeypatch)
    monkeypatch.setattr(job_storage_module.envs, "LUMILAKE_JOB_INDEX_RETENTION_DAYS", 0)
    backend.store["arc/jobs_index.json"] = (
        json.dumps(
            {
                "old-1": {
                    "job_id": "old-1",
                    "org_id": "test-org",
                    "user_id": "u",
                    "status": "completed",
                    "submitted_at": "2025-12-31T00:00:00+00:00",
                },
                "bogus": {"not": "a summary"},
            }
        ).encode(),
        "application/json",
    )
    backend.store["arc/output_index.json"] = (
        json.dumps({"loc-a": "old-1"}).encode(),
        "application/json",
    )
    storage = SqliteJobStorage(db_path=str(tmp_path / "jobs.sqlite"))
    rows, total = storage.list_summaries(
        org_id="test-org",
        user_id=None,
        job_ids=None,
        statuses=None,
        page=1,
        page_size=10,
    )
    assert total == 1 and rows[0]["job_id"] == "old-1"  # invalid entry skipped
    with pytest.raises(ValueError):  # output reservation came across too
        storage.reserve_output_location("loc-a", "someone-else")


def test_retention_prunes_only_terminal_jobs(tmp_path, monkeypatch) -> None:
    storage, _ = _sqlite(tmp_path, monkeypatch)
    monkeypatch.setattr(
        job_storage_module.envs, "LUMILAKE_JOB_INDEX_RETENTION_DAYS", 30
    )
    old = (dt.datetime.now(dt.UTC) - dt.timedelta(days=90)).isoformat()
    storage.save(_Rec(job_id="old-done", status="completed", submitted_at=old))
    storage.save(_Rec(job_id="old-running", status="running", submitted_at=old))
    # _Rec's default submitted_at is a fixed 2026-01-01, which is itself outside
    # a 30-day window -- "fresh" has to be genuinely recent or this asserts the
    # wrong thing.
    now = dt.datetime.now(dt.UTC).isoformat()
    storage.save(_Rec(job_id="fresh", status="completed", submitted_at=now))
    storage._last_prune = None  # force: None means never pruned
    storage._maybe_prune()
    remaining = {s.job_id for s in storage.iter_summaries()}
    # a stuck job must survive its retention window; finished ones age out
    assert remaining == {"old-running", "fresh"}


def test_first_prune_runs_on_a_fresh_process(tmp_path, monkeypatch) -> None:
    """Regression: the hourly guard must not swallow the FIRST prune.

    The guard was `now - self._last_prune < 3600` against a 0.0 sentinel, and
    time.monotonic()'s origin is arbitrary -- on a freshly-booted host it is
    small, so that comparison is true and pruning silently never ran for the
    first hour of uptime. Caught by CI, where the runner's uptime is seconds.
    Here _last_prune is deliberately left at its constructed value.
    """
    storage, _ = _sqlite(tmp_path, monkeypatch)
    monkeypatch.setattr(
        job_storage_module.envs, "LUMILAKE_JOB_INDEX_RETENTION_DAYS", 30
    )
    monkeypatch.setattr(job_storage_module.time, "monotonic", lambda: 12.0)
    old = (dt.datetime.now(dt.UTC) - dt.timedelta(days=90)).isoformat()
    storage.save(_Rec(job_id="old-done", status="completed", submitted_at=old))
    storage._maybe_prune()
    assert {s.job_id for s in storage.iter_summaries()} == set()


def test_retention_disabled_keeps_everything(tmp_path, monkeypatch) -> None:
    storage, _ = _sqlite(tmp_path, monkeypatch)
    old = (dt.datetime.now(dt.UTC) - dt.timedelta(days=900)).isoformat()
    storage.save(_Rec(job_id="ancient", status="completed", submitted_at=old))
    storage._last_prune = None  # force: None means never pruned
    storage._maybe_prune()
    assert {s.job_id for s in storage.iter_summaries()} == {"ancient"}


def test_survives_reopen(tmp_path, monkeypatch) -> None:
    db = str(tmp_path / "jobs.sqlite")
    _wire_blobs("arc", monkeypatch)
    monkeypatch.setattr(job_storage_module.envs, "LUMILAKE_JOB_INDEX_RETENTION_DAYS", 0)
    s1 = SqliteJobStorage(db_path=db)
    s1.save(_Rec(job_id="persisted"))
    s2 = SqliteJobStorage(db_path=db)
    assert {s.job_id for s in s2.iter_summaries()} == {"persisted"}
