"""Regression: ``_run_job`` finalization must not overwrite a status the
``cancel_job`` endpoint set during the unlocked artifact-storage gap.

Race window: ``_run_job`` releases ``jobs_lock`` to call ``_store_artifacts``
(I/O), and ``cancel_job`` can flip ``record.status = "cancelled"`` during
that gap. The finalize block re-acquires the lock and must re-check status
before committing ``completed``/``failed`` and ``finished_at``.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from lumid_hooks import PrincipalContext

import lumilake_server.routes.jobs as job_routes_module
from lumilake_server.routes.jobs import (
    JobRecord,
    _run_job,
)
from lumilake_server.runtime.protocol import Priority
from lumilake_server.schemas.io import S3Location


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_record(job_id: str) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        status="running",
        submitted_at="2026-01-01T00:00:00+00:00",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at=None,
        inputs={},
        output_location={"g": S3Location(type="s3", prefix="out/")},
    )


@pytest.mark.anyio
async def test_finalize_skips_overwrite_when_cancel_won_during_unlocked_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation flipped during the artifact-storage gap must survive.

    Drives ``_run_job`` with a fake server whose ``execute`` returns a normal
    response. The mocked ``_store_artifacts`` (which runs OUTSIDE the lock)
    flips ``record.status`` to ``"cancelled"`` mid-call — simulating a
    concurrent ``cancel_job`` HTTP request. After ``_run_job`` returns, the
    record must still read ``cancelled`` (not ``completed``) and
    ``finished_at`` must keep the cancel-set value.
    """
    record = _make_record("job-cancel-race")

    cancel_finished_at = "2026-06-12T12:00:00+00:00"

    def _fake_store_artifacts(_job_id: str, payload: Any) -> Any:
        # Simulate the cancel_job HTTP handler firing during the unlocked
        # I/O window: flip status + finished_at as cancel_job would.
        record.status = "cancelled"
        record.error = "cancelled by user"
        record.finished_at = cancel_finished_at
        return payload

    response_payload: dict[str, Any] = {"outputs": {}, "error_info": None}

    class _FakeServer:
        runtime_manager: Any = type(
            "_RM",
            (),
            {
                "set_dispatch_token": staticmethod(lambda *a, **kw: None),
            },
        )()

        def parse_query(self, graph_specs: Any) -> Any:
            return graph_specs

        async def execute(self, *args: Any, **kwargs: Any) -> Any:
            return _FakeResponse(response_payload)

        def trace_ids_for_request(self, *_a: Any) -> list[str]:
            return []

        def optimization_seconds_for_request(self, *_a: Any) -> float:
            return 0.0

        def selection_seconds_for_request(self, *_a: Any) -> float:
            return 0.0

        def clustering_seconds_for_request(self, *_a: Any) -> float:
            return 0.0

        def release_request_workflows(self, *_a: Any) -> None:
            return None

    class _FakeResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def model_dump(self) -> dict[str, Any]:
            return dict(self._payload)

    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_started_instance",
        classmethod(lambda cls: _FakeServer()),
    )
    monkeypatch.setattr(
        job_routes_module, "build_request_data_profile_tasks", lambda **_: []
    )
    monkeypatch.setattr(job_routes_module, "_store_artifacts", _fake_store_artifacts)
    monkeypatch.setattr(
        job_routes_module._job_storage, "save", lambda _r: None, raising=False
    )
    monkeypatch.setattr(
        job_routes_module, "_dump_output_locations", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(job_routes_module, "emit_usage", AsyncMock(return_value=None))
    monkeypatch.setattr(
        job_routes_module, "register_resource", AsyncMock(return_value=None)
    )

    principal = PrincipalContext(
        principal_id="p1",
        external_id="u1",
        org_id="o1",
        principal_type="user",
        scopes=["admin"],
    )

    with patch.object(job_routes_module, "LumilakeResponse") as fake_response_cls:
        fake_validated = type(
            "_V",
            (),
            {
                "error_info": None,
                "outputs": {},
                "model_validate": classmethod(lambda cls, v: fake_validated),
            },
        )
        fake_response_cls.model_validate = lambda v: fake_validated
        await _run_job(
            job_id=record.job_id,
            graph_specs={},
            workflow_slices={},
            record=record,
            priority=Priority.MEDIUM,
            principal=principal,
            runtime_token=None,
            trace_id=record.job_id,
            optimizer_type="halo",
        )

    assert record.status == "cancelled", (
        f"finalize overwrote cancellation: status={record.status!r} — the "
        "race-recheck guard at routes/jobs.py around the post-execute lock "
        "is missing or wrong"
    )
    assert (
        record.finished_at == cancel_finished_at
    ), "finalize overwrote cancel-set finished_at"
    assert record.error == "cancelled by user"


def _smoke() -> None:
    # Make the module importable in environments without pytest-anyio.
    asyncio.run(asyncio.sleep(0))
