import pytest

from lumilake_server.routes import jobs as jobs_routes
from lumilake_server.schemas.io import S3Location
from lumilake_server.utils import job_storage as job_storage_module
from lumilake_server.utils.job_storage import InMemoryJobStorage


def _seed(storage: InMemoryJobStorage, job_id: str, status: str) -> None:
    storage.save(
        jobs_routes.JobRecord(
            job_id=job_id,
            status=status,  # type: ignore[arg-type]
            submitted_at="2026-05-25T00:00:00+00:00",
            inputs={},
            output_location={
                "graph": S3Location(type="s3", prefix=f"{job_id}/out.txt")
            },
        )
    )


def _loaded_status(storage: InMemoryJobStorage, job_id: str) -> dict[str, object]:
    record = storage.load(job_id)
    assert record is not None
    return record


@pytest.mark.asyncio
async def test_recover_in_flight_jobs_marks_running_and_pending_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = InMemoryJobStorage()
    _seed(storage, "running-job", "running")
    _seed(storage, "pending-job", "pending")
    _seed(storage, "completed-job", "completed")

    monkeypatch.setattr(jobs_routes, "_job_storage", storage)
    monkeypatch.setattr(job_storage_module, "_job_storage", storage)
    monkeypatch.setattr(jobs_routes, "jobs", {})

    affected = await jobs_routes.recover_in_flight_jobs(reason="restart")

    assert affected == 2
    assert _loaded_status(storage, "running-job")["status"] == "failed"
    assert _loaded_status(storage, "running-job")["error"] == "restart"
    assert _loaded_status(storage, "pending-job")["status"] == "failed"
    assert _loaded_status(storage, "completed-job")["status"] == "completed"


@pytest.mark.asyncio
async def test_recover_in_flight_jobs_skips_in_memory_active_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = InMemoryJobStorage()
    _seed(storage, "still-running", "running")

    monkeypatch.setattr(jobs_routes, "_job_storage", storage)
    monkeypatch.setattr(job_storage_module, "_job_storage", storage)
    in_memory = jobs_routes.JobRecord(
        job_id="still-running",
        status="running",
        submitted_at="2026-05-25T00:00:00+00:00",
        inputs={},
        output_location={
            "graph": S3Location(type="s3", prefix="still-running/out.txt")
        },
    )
    monkeypatch.setattr(jobs_routes, "jobs", {"still-running": in_memory})

    affected = await jobs_routes.recover_in_flight_jobs()

    assert affected == 0
    assert _loaded_status(storage, "still-running")["status"] == "running"
