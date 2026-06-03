"""Tests for the ResourceRegistrar.reconcile startup sweep in main.lifespan."""

import datetime as dt
import logging
from collections.abc import Collection
from typing import Any

import pytest
from lumid_hooks import PrincipalContext, ResourceRef

import lumilake_server.utils.job_storage as job_storage_module
from lumilake_server import hooks
from lumilake_server.startup import reconcile_registrars
from lumilake_server.utils.job_storage import InMemoryJobStorage, JobSummary


def _make_storage_with_jobs(job_ids: list[str]) -> InMemoryJobStorage:
    storage = InMemoryJobStorage()
    for job_id in job_ids:
        storage._summaries[job_id] = JobSummary(
            job_id=job_id,
            org_id="org",
            user_id="user",
            status="completed",
            submitted_at=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        )
    return storage


class _RecordingRegistrar:
    name = "recording"
    reconciled: list[Collection[ResourceRef]]

    def __init__(self) -> None:
        self.reconciled = []

    async def register(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        logger: logging.Logger,
    ) -> None:
        pass

    async def deregister(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        logger: logging.Logger,
    ) -> None:
        pass

    async def reconcile(
        self,
        resources: Collection[ResourceRef],
        logger: logging.Logger,
    ) -> None:
        self.reconciled.append(list(resources))


class _RegistrarWithoutReconcile:
    name = "no-reconcile"

    async def register(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        logger: logging.Logger,
    ) -> None:
        pass

    async def deregister(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        logger: logging.Logger,
    ) -> None:
        pass


@pytest.mark.asyncio
async def test_reconcile_calls_registrars_with_job_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _make_storage_with_jobs(["job-1", "job-2"])
    monkeypatch.setattr(job_storage_module, "_job_storage", storage)

    registrar = _RecordingRegistrar()
    monkeypatch.setattr(hooks, "RESOURCE_REGISTRARS", [registrar])

    logger = logging.getLogger("test.reconcile")
    await reconcile_registrars(logger)

    assert len(registrar.reconciled) == 1
    refs = registrar.reconciled[0]
    reconciled_ids = {r.id for r in refs}
    assert reconciled_ids == {"job-1", "job-2"}
    assert all(r.kind == "job" for r in refs)


@pytest.mark.asyncio
async def test_reconcile_skips_registrar_without_reconcile_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _make_storage_with_jobs(["job-1"])
    monkeypatch.setattr(job_storage_module, "_job_storage", storage)

    no_reconcile = _RegistrarWithoutReconcile()
    recording = _RecordingRegistrar()
    monkeypatch.setattr(hooks, "RESOURCE_REGISTRARS", [no_reconcile, recording])

    logger = logging.getLogger("test.reconcile")
    await reconcile_registrars(logger)

    assert len(recording.reconciled) == 1


@pytest.mark.asyncio
async def test_reconcile_continues_on_registrar_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _make_storage_with_jobs(["job-x"])
    monkeypatch.setattr(job_storage_module, "_job_storage", storage)

    class _FailingRegistrar:
        name = "failing"

        async def register(self, *_: Any, **__: Any) -> None:
            pass

        async def deregister(self, *_: Any, **__: Any) -> None:
            pass

        async def reconcile(
            self,
            resources: Collection[ResourceRef],
            logger: logging.Logger,
        ) -> None:
            raise RuntimeError("db unavailable")

    failing = _FailingRegistrar()
    recording = _RecordingRegistrar()
    monkeypatch.setattr(hooks, "RESOURCE_REGISTRARS", [failing, recording])

    logger = logging.getLogger("test.reconcile")
    await reconcile_registrars(logger)

    assert len(recording.reconciled) == 1


@pytest.mark.asyncio
async def test_reconcile_empty_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    storage = _make_storage_with_jobs([])
    monkeypatch.setattr(job_storage_module, "_job_storage", storage)

    registrar = _RecordingRegistrar()
    monkeypatch.setattr(hooks, "RESOURCE_REGISTRARS", [registrar])

    logger = logging.getLogger("test.reconcile")
    await reconcile_registrars(logger)

    assert registrar.reconciled[0] == []
