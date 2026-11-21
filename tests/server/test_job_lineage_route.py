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
from lumilake_server.runtime.record_lineage import build_lineage_payload
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

_PRINCIPAL = PrincipalContext(
    principal_id="alice",
    org_id="demo",
    external_id="alice@example.com",
    principal_type="user",
    scopes=["admin"],
)

_JOB = "job-1"


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
def job_routes(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, InMemoryJobStorage]:
    storage = InMemoryJobStorage()
    job_storage_module._job_storage = storage
    job_routes_module.jobs.clear()
    job_routes_module._job_storage = storage
    return job_routes_module, storage


@pytest.fixture
def app(job_routes: tuple[Any, InMemoryJobStorage]) -> FastAPI:
    module, _ = job_routes
    application = FastAPI()
    application.state.logger = logging.getLogger("test.job_lineage")
    application.state.background_tasks = set()
    application.add_middleware(TraceIdMiddleware)
    application.include_router(module.router)
    return application


def _seed_job(module: Any, job_id: str) -> None:
    module.jobs[job_id] = JobRecord(
        job_id=job_id,
        status="completed",
        submitted_at=dt.datetime.now(dt.UTC).isoformat(),
        inputs={},
        output_location={"out": S3Location(type="s3", prefix="x/y")},
        org_id="demo",
        user_id="alice@example.com",
    )


async def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer token"},
    )


def _chain(job_id: str = _JOB) -> tuple[list[RecordEntry], RecordIndex]:
    """An upstream entry and its consumer, plus the index that names them."""
    upstream_address = RecordAddress(op="Mark", record="a", round=0)
    upstream_value = {"id": "a", "text": "ONE"}
    upstream = RecordEntry(
        digest=entry_digest(
            spec_digest=digest_of({"op": "Mark"}),
            address=upstream_address,
            input_digests=[],
            input_values=[],
            realization=RecordRealization(),
            versions=EffectVersions(),
        ),
        address=upstream_address,
        value=upstream_value,
        value_digest=digest_of(upstream_value),
        source=RecordSource(
            job=job_id, round=0, task_id="task-mark", worker="w1", member_jobs=[job_id]
        ),
        written_at=0.0,
    )
    downstream_address = RecordAddress(op="Join", record="a", round=0)
    downstream = RecordEntry(
        digest=entry_digest(
            spec_digest=digest_of({"op": "Join"}),
            address=downstream_address,
            input_digests=[upstream.digest],
            input_values=[upstream.value_digest or ""],
            realization=RecordRealization(),
            versions=EffectVersions(),
        ),
        address=downstream_address,
        value={},
        source=RecordSource(
            job=job_id, round=0, task_id="task-join", worker="w2", member_jobs=[job_id]
        ),
        written_at=0.0,
        input_digests=[upstream.digest],
        input_values=[upstream.value_digest or ""],
    )
    index = RecordIndex(
        job=job_id,
        entries=[
            RecordIndexEntry(op="Join", record="a", round=0, digest=downstream.digest),
            RecordIndexEntry(op="Mark", record="a", round=0, digest=upstream.digest),
        ],
    )
    return [upstream, downstream], index


def _write_chain(storage: InMemoryJobStorage) -> None:
    entries, index = _chain()
    for entry in entries:
        storage.save_blob(
            entry_object_name(entry.digest), entry.body(), "application/json"
        )
    storage.save_blob(index_object_name(_JOB), index.body(), "application/json")


@pytest.mark.anyio
async def test_lineage_returns_the_chain(
    app: FastAPI, job_routes: tuple[Any, InMemoryJobStorage]
) -> None:
    module, storage = job_routes
    _seed_job(module, _JOB)
    _write_chain(storage)
    async with await _client(app) as client:
        resp = await client.get(
            "/jobs/job-1/lineage", params={"op": "Join", "record": "a"}
        )
    assert resp.status_code == 200
    entries = resp.json()["data"]["entries"]
    assert [(entry["op"], entry["record"]) for entry in entries] == [
        ("Join", "a"),
        ("Mark", "a"),
    ]
    assert entries[0]["task_id"] == "task-join"
    assert entries[1]["worker"] == "w1"


@pytest.mark.anyio
async def test_lineage_unknown_record_is_404(
    app: FastAPI, job_routes: tuple[Any, InMemoryJobStorage]
) -> None:
    module, storage = job_routes
    _seed_job(module, _JOB)
    _write_chain(storage)
    async with await _client(app) as client:
        resp = await client.get(
            "/jobs/job-1/lineage", params={"op": "Join", "record": "zzz"}
        )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_lineage_defaults_to_the_newest_stored_round(
    job_routes: tuple[Any, InMemoryJobStorage],
) -> None:
    _, storage = job_routes
    entries, index = _chain()
    latest = entries[0].model_copy(
        update={
            "address": RecordAddress(op="Join", record="a", round=2),
            "source": entries[0].source.model_copy(update={"round": 2}),
        }
    )
    for entry in [*entries, latest]:
        storage.save_blob(
            entry_object_name(entry.digest), entry.body(), "application/json"
        )
    storage.save_blob(
        index_object_name(_JOB),
        RecordIndex(
            job=_JOB,
            entries=[
                *index.entries,
                RecordIndexEntry(op="Join", record="a", round=2, digest=latest.digest),
            ],
        ).body(),
        "application/json",
    )
    payload = build_lineage_payload(
        storage, job_id=_JOB, op="Join", record="a", round_number=None
    )
    assert payload is not None
    assert payload["entries"][0]["round"] == 2
