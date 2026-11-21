"""GET /jobs/{job_id}/lineage returns a stored record's chain from the archive."""

import datetime as dt
import logging
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from lumid_hooks import PrincipalContext, ResourceRef

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
from lumilake_server.utils.job_storage import InMemoryJobStorage

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


class _Permissions:
    name = "test.permissions"

    async def accessible_ids(
        self,
        principal: PrincipalContext,
        kind: str,
        action: str,
        logger: logging.Logger,
    ) -> frozenset[str] | None:
        return None

    async def require(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        action: str,
        logger: logging.Logger,
    ) -> None:
        return None


@pytest.fixture(autouse=True)
def _hooks() -> Iterator[None]:
    hooks.IDENTITY_PROVIDERS.clear()
    hooks.SUBMISSION_GUARDS.clear()
    hooks.USAGE_SINKS.clear()
    hooks.PERMISSION_CHECKERS.clear()
    hooks.RESOURCE_REGISTRARS.clear()
    hooks.IDENTITY_PROVIDERS.append(_Identity())
    hooks.PERMISSION_CHECKERS.append(_Permissions())
    yield
    hooks.IDENTITY_PROVIDERS.clear()
    hooks.SUBMISSION_GUARDS.clear()
    hooks.USAGE_SINKS.clear()
    hooks.PERMISSION_CHECKERS.clear()
    hooks.RESOURCE_REGISTRARS.clear()


@pytest.fixture
def app() -> tuple[FastAPI, InMemoryJobStorage, Any]:
    storage = InMemoryJobStorage()
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
    return application, storage, job_routes_module


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
    inputs = [producer.digest] if producer is not None else []
    values = [producer.value_digest or ""] if producer is not None else []
    value = {} if producer is not None else {"id": "a", "text": "ONE"}
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


def _store_chain(
    storage: InMemoryJobStorage, extra_rounds: tuple[int, ...] = ()
) -> None:
    """An upstream entry and its consumer, plus the index that names them, and
    the consumer again at each later round."""
    upstream = _entry("Mark", task_id="task-mark", worker="w1")
    entries = [
        upstream,
        _entry("Join", task_id="task-join", worker="w2", producer=upstream),
        *[
            _entry(
                "Join",
                task_id="task-join",
                worker="w2",
                round_number=number,
                producer=upstream,
            )
            for number in extra_rounds
        ],
    ]
    for entry in entries:
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
                for entry in entries
            ],
        ).body(),
        "application/json",
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("params", "rounds", "status", "expected"),
    [
        ({"op": "Join", "record": "a"}, (), 200, [("Join", 0), ("Mark", 0)]),
        ({"op": "Join", "record": "zzz"}, (), 404, []),
        ({"op": "Join", "record": "a"}, (2,), 200, [("Join", 2), ("Mark", 0)]),
    ],
)
async def test_lineage_reads_a_stored_chain(
    app: tuple[FastAPI, InMemoryJobStorage, Any],
    params: dict[str, str],
    rounds: tuple[int, ...],
    status: int,
    expected: list[tuple[str, int]],
) -> None:
    """A record's chain comes back level by level, each entry naming the job,
    task, worker and round that produced it; an unknown record is 404; and a
    read with no round takes the newest round the index lists."""
    application, storage, _ = app
    _store_chain(storage, rounds)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://test",
        headers={"Authorization": "Bearer token"},
    ) as client:
        resp = await client.get(f"/jobs/{_JOB}/lineage", params=params)
    assert resp.status_code == status
    if status != 200:
        return
    entries = resp.json()["data"]["entries"]
    assert [(entry["op"], entry["round"]) for entry in entries] == expected
    assert entries[0]["task_id"] == "task-join"
    assert entries[0]["job_id"] == _JOB
    assert all(entry["worker"] in {"w1", "w2"} for entry in entries)
