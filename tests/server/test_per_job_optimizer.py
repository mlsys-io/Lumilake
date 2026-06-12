"""Tests for per-job optimizer selection.

Covers:
- _validate_optimizer_type accepts known local types and provider-advertised types.
- _validate_optimizer_type raises HTTP 422 for unknown types with a helpful message.
- Optimizer names submitted in mixed/upper case are stored lowercased so the
  priority-queue partition key is stable across equivalent spellings.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from lumid_hooks import PrincipalContext, ResourceRef

import lumilake_server.utils.job_storage as job_storage_module
from lumilake_server import hooks
from lumilake_server.middleware import TraceIdMiddleware
from lumilake_server.routes import jobs as job_routes_module
from lumilake_server.routes.jobs import (
    _validate_optimizer_type,
)
from lumilake_server.runtime.optimizer import OPTIMIZER_PROVIDERS
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.protocol import LumilakeRequestConfig
from lumilake_server.runtime.server import SchedulePreview
from lumilake_server.utils.job_storage import InMemoryJobStorage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_providers():
    snapshot = list(OPTIMIZER_PROVIDERS)
    OPTIMIZER_PROVIDERS.clear()
    yield
    OPTIMIZER_PROVIDERS.clear()
    OPTIMIZER_PROVIDERS.extend(snapshot)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def test_validate_optimizer_type_known_local_passes() -> None:
    # "halo" is a built-in local type — must not raise
    _validate_optimizer_type("halo")


def test_validate_optimizer_type_provider_advertised_passes() -> None:
    class _FakeProvider:
        def list_optimizers(self) -> list[str]:
            return ["provider-special"]

        def create_optimizer(self, optimizer_type: str, **kwargs: Any) -> Any:
            raise NotImplementedError

    OPTIMIZER_PROVIDERS.append(_FakeProvider())
    _validate_optimizer_type("provider-special")  # must not raise


def test_validate_optimizer_type_unknown_raises_http_422() -> None:
    with pytest.raises(HTTPException) as exc_info:
        _validate_optimizer_type("does-not-exist-xyz")
    assert exc_info.value.status_code == 422
    assert "does-not-exist-xyz" in exc_info.value.detail
    assert "Local:" in exc_info.value.detail


# ---------------------------------------------------------------------------
# Route-level normalization: mixed-case optimizer names must be lowercased
# before being stored in LumilakeRequestConfig.optimizer_type so that the
# priority-queue partition key is stable across equivalent spellings.
#
# Guards routes/jobs.py:1640 (preview path) and :1910 (submit path).
# Dropping either `optimizer = optimizer.lower()` line would reintroduce the
# bug and fail these tests.
# ---------------------------------------------------------------------------

_DEMO_PRINCIPAL = PrincipalContext(
    principal_id="alice",
    org_id="demo",
    external_id="alice@example.com",
    principal_type="user",
    scopes=["admin"],
)


class _AllowAllIdentity:
    name = "test.identity"

    async def resolve(
        self, token: str, logger: logging.Logger
    ) -> PrincipalContext | None:
        return _DEMO_PRINCIPAL.model_copy(deep=True) if token == "token" else None


class _AllowAllPermissions:
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


@dataclass
class _PreviewCapture:
    """Holds the LumilakeRequestConfig captured by _FakePreviewServer."""

    config: LumilakeRequestConfig | None = None


class _FakePreviewServer:
    """Minimal fake server for preview-path normalization tests."""

    is_started = True

    def __init__(self, capture: _PreviewCapture) -> None:
        self._capture = capture

    def parse_query(self, graph_specs: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return graph_specs

    async def preview_schedule(
        self,
        graphs: dict[str, Any],
        *,
        request_id: str | None = None,
        data_profile_results: dict[str, list[dict[str, Any]]] | None = None,
        data_profile_sources: dict[str, list[Any]] | None = None,
        config: LumilakeRequestConfig | None = None,
    ) -> SchedulePreview:
        self._capture.config = config
        return SchedulePreview(
            request_id=request_id or "preview-test",
            selected_workers=[],
            worker_profiles={},
            runtime_graph_node_counts={},
            merged_runtime_node_count=0,
            schedule=Schedule(worker_assignment={}),
            selection_seconds=0.0,
            clustering_seconds=0.0,
            optimization_seconds=0.0,
        )


@pytest.fixture(autouse=True)
def _reset_hook_state_normalization() -> Iterator[None]:
    hooks.IDENTITY_PROVIDERS.clear()
    hooks.SUBMISSION_GUARDS.clear()
    hooks.USAGE_SINKS.clear()
    hooks.PERMISSION_CHECKERS.clear()
    hooks.RESOURCE_REGISTRARS.clear()
    hooks.IDENTITY_PROVIDERS.append(_AllowAllIdentity())
    hooks.PERMISSION_CHECKERS.append(_AllowAllPermissions())
    yield
    hooks.IDENTITY_PROVIDERS.clear()
    hooks.SUBMISSION_GUARDS.clear()
    hooks.USAGE_SINKS.clear()
    hooks.PERMISSION_CHECKERS.clear()
    hooks.RESOURCE_REGISTRARS.clear()


def _make_normalization_app(
    monkeypatch: pytest.MonkeyPatch,
    fake_server: Any,
) -> FastAPI:
    storage = InMemoryJobStorage()
    job_storage_module._job_storage = storage
    job_routes_module.jobs.clear()
    job_routes_module._job_storage = storage
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_started_instance",
        classmethod(lambda cls: fake_server),
    )
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_instance",
        classmethod(lambda cls: fake_server),
    )
    monkeypatch.setattr(
        job_routes_module, "build_request_data_profile_tasks", lambda **_: []
    )
    app = FastAPI()
    app.state.logger = logging.getLogger("test.optimizer_normalization")
    app.state.background_tasks = set()
    app.add_middleware(TraceIdMiddleware)
    app.include_router(job_routes_module.router)
    return app


@pytest.mark.anyio
async def test_submit_stores_optimizer_lowercase_for_partition_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_validate_optimizer_type`` accepts mixed case but the stored
    ``optimizer_type`` must be lowercased for consistent partition keys.

    Submitting optimizer="HALO" must result in optimizer_type="halo" being passed
    to _run_job (and therefore into LumilakeRequestConfig.optimizer_type).
    """
    captured: list[str | None] = []

    async def _fake_run_job(
        job_id: str,
        graph_specs: Any,
        workflow_slices: Any,
        record: Any,
        priority: Any,
        principal: Any,
        runtime_token: Any,
        trace_id: str,
        optimizer_type: str | None = None,
    ) -> None:
        captured.append(optimizer_type)

    monkeypatch.setattr(job_routes_module, "_run_job", _fake_run_job)

    app = _make_normalization_app(monkeypatch, object())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json={
                "data": [
                    {
                        "workflow": "{}",
                        "inputs": {"x": ["v"]},
                        "output_location": {"type": "s3", "prefix": "out/"},
                    }
                ],
                "optimizer": "HALO",
            },
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 200, resp.text
    # Drain background tasks so _fake_run_job has run.
    for task in list(app.state.background_tasks):
        await task
    assert captured == ["halo"], (
        f"expected optimizer_type='halo' but got {captured!r}; "
        "dropping optimizer.lower() in the submit handler would reproduce this failure"
    )


@pytest.mark.anyio
async def test_preview_stores_optimizer_lowercase_for_partition_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards routes/jobs.py:1640 — _validate_optimizer_type accepts mixed case
    but the stored optimizer_type must be lowercased for consistent partition keys.

    Previewing with optimizer="HALO" must result in
    LumilakeRequestConfig.optimizer_type="halo" being passed to preview_schedule.
    """
    capture = _PreviewCapture()
    fake_server = _FakePreviewServer(capture)
    app = _make_normalization_app(monkeypatch, fake_server)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs/preview",
            json={
                "data": [
                    {
                        "workflow": "{}",
                        "inputs": {"x": ["v"]},
                    }
                ],
                "optimizer": "HALO",
            },
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 200, resp.text
    assert capture.config is not None
    assert capture.config.optimizer_type == "halo", (
        f"expected optimizer_type='halo' but got {capture.config.optimizer_type!r}; "
        "dropping optimizer.lower() at routes/jobs.py:1640 would reproduce this failure"
    )


# ---------------------------------------------------------------------------
# _validate_optimizer_type case-insensitive provider comparison (Finding 1)
#
# Before the fix, _validate_optimizer_type compared the lowercased request
# against the RAW names from provider.list_optimizers() — so a provider
# advertising "RemoteX" would reject "remotex" with a 422 even though
# /optimizer already displays it as "remotex".
# ---------------------------------------------------------------------------


class _MixedCaseProvider:
    """Provider that advertises a name with mixed casing."""

    def list_optimizers(self) -> list[str]:
        return ["RemoteX"]

    def create_optimizer(self, optimizer_type: str, **kwargs: Any) -> Any:
        raise NotImplementedError


@pytest.mark.parametrize("variant", ["remotex", "RemoteX", "REMOTEX"])
def test_validate_optimizer_type_accepts_provider_mixed_case(variant: str) -> None:
    """_validate_optimizer_type must accept any casing of a provider's "RemoteX".

    Pre-fix (comparing lowercased request against raw provider names), only the
    lowercased variant matched and the other two raised HTTPException 422.
    """
    OPTIMIZER_PROVIDERS.append(_MixedCaseProvider())
    _validate_optimizer_type(variant)  # must not raise
