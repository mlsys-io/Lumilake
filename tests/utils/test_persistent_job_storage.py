"""Behavior tests for PersistentJobStorage backed by the lumid_data_client seam.

Uses an in-memory stub keyed on the blob HTTP API to avoid any network calls.
Verifies exact blob key shapes and ArchiveNotFound semantics.
"""

from dataclasses import dataclass, field

import pytest

import lumilake_server.utils.job_storage as job_storage_module
from lumilake_server.utils.job_storage import (
    ArchiveNotFound,
    JobStorage,
    PersistentJobStorage,
)
from lumilake_server.utils.lumid_data_client import BlobNotFound


class _MemBlobStore:
    """In-memory drop-in for the lumid_data_client put_blob/get_blob seam."""

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
    inputs: dict[str, object] = field(default_factory=dict)
    output_location: dict[str, object] = field(default_factory=dict)
    progress: dict[str, object] = field(default_factory=dict)
    result: dict[str, object] | None = None


def _make_storage(
    prefix: str, monkeypatch: pytest.MonkeyPatch
) -> tuple[PersistentJobStorage, _MemBlobStore]:
    """Build a PersistentJobStorage wired to an in-memory blob store."""
    backend = _MemBlobStore()
    monkeypatch.setattr(job_storage_module.envs, "S3_ARCHIVE_PREFIX", prefix)
    monkeypatch.setattr(
        job_storage_module.lumid_data_client, "put_blob", backend.put_blob
    )
    monkeypatch.setattr(
        job_storage_module.lumid_data_client, "get_blob", backend.get_blob
    )

    storage = PersistentJobStorage.__new__(PersistentJobStorage)
    JobStorage.__init__(storage)
    storage.key_prefix = prefix.strip("/")
    return storage, backend


def test_save_uses_correct_blob_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    storage, backend = _make_storage("myarchive/2026", monkeypatch)
    rec = _Rec(
        job_id="job-abc",
        output_location={"g": {"type": "s3", "prefix": "out/"}},
    )
    storage.save(rec)

    assert "myarchive/2026/job-abc/record.json" in backend.store
    assert "myarchive/2026/job-abc/inputs.json" in backend.store
    assert "myarchive/2026/job-abc/progress.json" in backend.store
    assert "myarchive/2026/jobs_index.json" in backend.store


def test_save_and_load_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    storage, _ = _make_storage("arc", monkeypatch)
    rec = _Rec(
        job_id="job-xyz",
        status="running",
        output_location={"g": {"type": "s3", "prefix": "out/"}},
    )
    storage.save(rec)
    loaded = storage.load("job-xyz")
    assert loaded is not None
    assert loaded["job_id"] == "job-xyz"
    assert loaded["status"] == "running"


def test_load_missing_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    storage, _ = _make_storage("arc", monkeypatch)
    assert storage.load("no-such-job") is None


def test_save_artifact_returns_bare_blob_key(monkeypatch: pytest.MonkeyPatch) -> None:
    storage, backend = _make_storage("archive/v1", monkeypatch)
    uri = storage.save_artifact(
        "job-123", "graph.yaml", b"data: {}", "application/x-yaml"
    )
    expected_key = "archive/v1/job-123/artifacts/graph.yaml"
    assert uri == expected_key
    assert backend.store[expected_key][0] == b"data: {}"


def test_get_artifact_raises_key_error_on_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, _ = _make_storage("arc", monkeypatch)
    with pytest.raises(KeyError):
        storage.get_artifact("job-999", "no-file.txt")


def test_get_artifact_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    storage, _ = _make_storage("archive", monkeypatch)
    storage.save_artifact("j1", "result.json", b'{"ok":1}', "application/json")
    data, ct = storage.get_artifact("j1", "result.json")
    assert data == b'{"ok":1}'
    assert ct == "application/json"


def test_reserve_output_location_raises_on_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, _ = _make_storage("arc", monkeypatch)
    rec = _Rec(
        job_id="job-a",
        status="running",
        output_location={"g": {"type": "s3", "prefix": "out/"}},
    )
    storage.save(rec)
    storage.reserve_output_location("loc-1", "job-a")
    with pytest.raises(ValueError, match="already reserved"):
        storage.reserve_output_location("loc-1", "job-b")


def test_prefix_empty_uses_bare_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    storage, backend = _make_storage("", monkeypatch)
    storage.save(
        _Rec(
            job_id="j0",
            output_location={"g": {"type": "s3", "prefix": "o/"}},
        )
    )
    assert "j0/record.json" in backend.store


def test_get_translates_blob_not_found_to_archive_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_get_blob converts BlobNotFound from lumid_data_client to ArchiveNotFound."""
    storage, _ = _make_storage("arc", monkeypatch)
    with pytest.raises(ArchiveNotFound):
        storage._get_blob("some/missing/key.json")
