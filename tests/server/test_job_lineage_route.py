"""GET /jobs/{job_id}/lineage returns a stored record's chain from the archive."""

import datetime as dt
import logging
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from lumid_hooks import PrincipalContext

import lumilake_server.utils.job_storage as job_storage_module
from lumilake_server import hooks
from lumilake_server.middleware import TraceIdMiddleware
from lumilake_server.routes import jobs as job_routes_module
from lumilake_server.routes.jobs import JobRecord
from lumilake_server.runtime.record_store import (
    EffectVersions,
    RecordAddress,
    RecordEntry,
    RecordIndex,
    RecordIndexEntry,
    RecordRealization,
    RecordSource,
    digest_of,
    entry_digest,
    entry_object_name,
    index_object_name,
)
from lumilake_server.schemas.io import S3Location

_JOB = "job-1"
_PRINCIPAL = PrincipalContext(
    principal_id="alice",
    org_id="demo",
    external_id="alice@example.com",
    principal_type="user",
    scopes=["admin"],
)


class _Identity:
    name = "test.identity"

    async def resolve(
        self, token: str, logger: logging.Logger
    ) -> PrincipalContext | None:
        return _PRINCIPAL.model_copy(deep=True) if token == "token" else None


@pytest.fixture(autouse=True)
def _hooks() -> Iterator[None]:
    hooks.IDENTITY_PROVIDERS.clear()
    hooks.SUBMISSION_GUARDS.clear()
    hooks.USAGE_SINKS.clear()
    hooks.RESOURCE_REGISTRARS.clear()
    hooks.IDENTITY_PROVIDERS.append(_Identity())
    yield
    hooks.IDENTITY_PROVIDERS.clear()


@pytest.fixture
def app() -> tuple[FastAPI, Any]:
    storage = job_storage_module.InMemoryJobStorage()
    job_storage_module._job_storage = storage
    job_routes_module.jobs.clear()
    job_routes_module._job_storage = storage
    job_routes_module.jobs[_JOB] = JobRecord(
        job_id=_JOB,
        status="completed",
        submitted_at=dt.datetime.now(dt.UTC).isoformat(),
        inputs={},
        output_location={"out": S3Location(type="s3", prefix="x/y")},
        org_id="demo",
        user_id="alice@example.com",
    )
    application = FastAPI()
    application.state.logger = logging.getLogger("test.job_lineage")
    application.state.background_tasks = set()
    application.add_middleware(TraceIdMiddleware)
    application.include_router(job_routes_module.router)
    return application, storage


def _entry(
    op: str,
    *,
    task_id: str,
    worker: str,
    round_number: int = 0,
    producer: RecordEntry | None = None,
) -> RecordEntry:
    """A stored record: an input of ``producer`` when one is given."""
    address = RecordAddress(op=op, record="a", round=round_number)
    inputs = [] if producer is None else [producer.digest]
    values = [] if producer is None else [producer.value_digest or ""]
    value = {"id": "a", "text": "ONE"} if producer is None else {}
    return RecordEntry(
        digest=entry_digest(
            spec_digest=digest_of({"op": op}),
            address=address,
            input_digests=inputs,
            input_values=values,
            realization=RecordRealization(),
            versions=EffectVersions(),
        ),
        address=address,
        value=value,
        value_digest=digest_of(value),
        input_digests=inputs,
        input_values=values,
        source=RecordSource(
            job=_JOB,
            round=round_number,
            task_id=task_id,
            worker=worker,
            member_jobs=[_JOB],
        ),
        written_at=0.0,
    )


@pytest.mark.anyio
async def test_lineage_reads_a_stored_chain(
    app: tuple[FastAPI, Any],
) -> None:
    """A record's chain comes back level by level from the archive alone, each
    entry naming the job, task, worker and round that produced it, and an
    unknown record is 404."""
    application, storage = app
    upstream = _entry("Mark", task_id="task-mark", worker="w1")
    consumer = _entry(
        "Join", task_id="task-join", worker="w2", round_number=2, producer=upstream
    )
    for entry in (upstream, consumer):
        storage.save_blob(
            entry_object_name(entry.digest), entry.body(), "application/json"
        )
    storage.save_blob(
        index_object_name(_JOB),
        RecordIndex(
            job=_JOB,
            entries=[
                RecordIndexEntry(
                    op=entry.address.op,
                    record=entry.address.record,
                    round=entry.address.round,
                    digest=entry.digest,
                )
                for entry in (upstream, consumer)
            ],
        ).body(),
        "application/json",
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://test",
        headers={"Authorization": "Bearer token"},
    ) as client:
        found = await client.get(
            f"/jobs/{_JOB}/lineage", params={"op": "Join", "record": "a"}
        )
        missing = await client.get(
            f"/jobs/{_JOB}/lineage", params={"op": "Join", "record": "zzz"}
        )
    entries = found.json()["data"]["entries"]
    assert [(entry["op"], entry["round"]) for entry in entries] == [
        ("Join", 2),
        ("Mark", 0),
    ]
    assert (entries[0]["task_id"], entries[0]["worker"]) == ("task-join", "w2")
    assert missing.status_code == 404


@pytest.mark.anyio
async def test_job_get_exposes_a_reuse_decisions_field(
    app: tuple[FastAPI, Any],
) -> None:
    """A job that reused the store reports one decision per op on its GET, and
    a job that did not reports an empty list."""
    application, _ = app
    jobs_module = job_routes_module
    jobs_module.jobs[_JOB].reuse_decisions = [
        {"op": "Answer", "decision": "hit", "reason": None, "records": 1},
        {"op": "Score", "decision": "miss", "reason": "no entry", "records": 1},
    ]
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs/" + _JOB, headers={"Authorization": "Bearer token"}
        )
    decisions = resp.json()["data"]["reuse_decisions"]
    assert [(d["op"], d["decision"], d["reason"]) for d in decisions] == [
        ("Answer", "hit", None),
        ("Score", "miss", "no entry"),
    ]
