import asyncio
import datetime as dt
import os
import sys
from contextlib import contextmanager

import pytest

# Importing lumilake_server.main runs envs.load_env_file_or_raise() and
# envs.validate() at import time; satisfy those before importing it.
os.environ.setdefault("LUMILAKE_SKIP_DOTENV_CHECK", "1")
os.environ.setdefault("LUMILAKE_SERVER_HOST", "0.0.0.0")
os.environ.setdefault("LUMILAKE_SERVER_PORT", "9000")
os.environ.setdefault("LUMILAKE_RUNTIME_ORCHESTRATOR_URL", "http://127.0.0.1:18000")
os.environ.setdefault("LUMILAKE_CPU_WORKER_GROUP_SIZE", "1")
os.environ.setdefault("LUMILAKE_GPU_WORKER_GROUP_SIZE", "0")

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


@pytest.mark.asyncio
async def test_recover_leaves_a_terminal_record_behind_a_stale_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = InMemoryJobStorage()
    _seed(storage, "done-late", "running")
    storage._storage["done-late"][
        "status"
    ] = "completed"  # record landed, index did not
    assert [s.job_id for s in storage.iter_summaries({"running"})] == ["done-late"]

    monkeypatch.setattr(jobs_routes, "_job_storage", storage)
    monkeypatch.setattr(job_storage_module, "_job_storage", storage)
    monkeypatch.setattr(jobs_routes, "jobs", {})

    affected = await jobs_routes.recover_in_flight_jobs()

    assert affected == 0
    assert _loaded_status(storage, "done-late")["status"] == "completed"
    assert [s.job_id for s in storage.iter_summaries({"running"})] == []


@pytest.mark.asyncio
async def test_recover_spares_jobs_submitted_after_the_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = InMemoryJobStorage()
    _seed(storage, "old-job", "running")  # submitted 2026-05-25
    monkeypatch.setattr(jobs_routes, "_job_storage", storage)
    monkeypatch.setattr(job_storage_module, "_job_storage", storage)
    monkeypatch.setattr(jobs_routes, "jobs", {})

    affected = await jobs_routes.recover_in_flight_jobs(
        submitted_before=dt.datetime(2026, 5, 24, tzinfo=dt.UTC)
    )
    assert affected == 0
    assert _loaded_status(storage, "old-job")["status"] == "running"

    affected = await jobs_routes.recover_in_flight_jobs(
        submitted_before=dt.datetime(2026, 5, 26, tzinfo=dt.UTC)
    )
    assert affected == 1
    assert _loaded_status(storage, "old-job")["status"] == "failed"


@pytest.mark.asyncio
async def test_recover_retries_until_storage_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    async def flaky(**kwargs: object) -> int:
        calls.append(kwargs)
        if len(calls) < 3:
            raise ConnectionError("storage not up yet")
        return 0

    monkeypatch.setattr(jobs_routes, "recover_in_flight_jobs", flaky)
    ok = await jobs_routes.recover_in_flight_jobs_until_done(
        dt.datetime(2026, 5, 26, tzinfo=dt.UTC), attempts=5, first_delay_s=0
    )
    assert ok and len(calls) == 3
    assert calls[0] == {"submitted_before": dt.datetime(2026, 5, 26, tzinfo=dt.UTC)}


@pytest.mark.asyncio
async def test_startup_recovery_failure_schedules_background_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the startup recovery pass raises, a background
    ``recover_in_flight_jobs_until_done`` task is scheduled in
    ``app.state.background_tasks`` and cancelled cleanly on shutdown."""
    # main.py runs ``app = build_app()`` at import, which parses sys.argv.
    monkeypatch.setattr(sys, "argv", ["pytest"])
    from lumilake_server import main as main_module

    async def boom(**kwargs: object) -> int:
        raise ConnectionError("storage not up yet")

    monkeypatch.setattr(main_module.envs, "LUMILAKE_RECOVER_IN_FLIGHT_JOBS", True)
    monkeypatch.setattr(main_module.jobs, "recover_in_flight_jobs", boom)
    monkeypatch.setattr(
        main_module.jobs, "recover_in_flight_jobs_until_done", _noop_until_done
    )
    monkeypatch.setattr(main_module, "_load_plugins", _noop_load_plugins)
    monkeypatch.setattr(main_module, "reconcile_registrars", _noop_reconcile)
    monkeypatch.setattr(
        main_module.LumilakeServer, "serve_instance", _noop_serve_instance
    )

    app = main_module.build_app()
    async with app.router.lifespan_context(app):
        assert len(app.state.background_tasks) == 1
        task = next(iter(app.state.background_tasks))
        assert not task.done()
    # Shutdown cancels the background task cleanly.
    assert task.cancelled()


async def _noop_until_done(submitted_before: dt.datetime, **kwargs: object) -> bool:
    await asyncio.sleep(3600)
    return True


async def _noop_load_plugins(stack: object, logger: object) -> None:
    return None


async def _noop_reconcile(logger: object) -> None:
    return None


@contextmanager
def _noop_serve_instance(config: object = None):
    yield None
