"""Behavior tests for the DBLocation output guard and S3/DB input wiring.

Covers:
- 422 at submit when output_location is a DBLocation
- DB input validates via a stubbed acatalog_column_exists
- S3 input listing expands prefix via stubbed alist_blob_keys
- S3 input listing raises 422 on empty result
"""

import logging
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from lumid_hooks import PrincipalContext, ResourceRef

import lumilake_server.routes.jobs as jmod
import lumilake_server.utils.job_storage as job_storage_module
from lumilake_server import hooks
from lumilake_server.middleware import TraceIdMiddleware
from lumilake_server.routes import jobs as job_routes_module
from lumilake_server.routes.jobs import (
    _resolve_s3_input_values,
    _validate_db_location_live,
)
from lumilake_server.runtime.protocol import LumilakeResponse
from lumilake_server.schemas.io import DBLocation, S3Location
from lumilake_server.utils.job_storage import InMemoryJobStorage

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


class _FakeRuntimeServer:
    is_started = True

    def parse_query(self, graph_specs: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return graph_specs

    async def execute(
        self,
        graphs: dict[str, Any],
        request_id: str | None = None,
        config: Any | None = None,
        workflow_slices: dict[str, Any] | None = None,
    ) -> LumilakeResponse:
        return LumilakeResponse(outputs={})

    def trace_ids_for_request(self, job_id: str) -> list[str]:
        return []

    def optimization_seconds_for_request(self, job_id: str) -> float:
        return 0.0

    def release_request_workflows(self, job_id: str) -> None:
        return None

    async def cancel_request(self, job_id: str) -> None:
        pass

    def selection_seconds_for_request(self, job_id: str) -> float:
        return 0.0

    def clustering_seconds_for_request(self, job_id: str) -> float:
        return 0.0

    async def get_request_status(self, job_id: str) -> dict[str, Any]:
        return {}

    async def preview_schedule(self, **kwargs: Any) -> Any:
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _reset_hook_state() -> Iterator[None]:
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


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    storage = InMemoryJobStorage()
    job_storage_module._job_storage = storage
    job_routes_module.jobs.clear()
    job_routes_module._job_storage = storage
    fake_server = _FakeRuntimeServer()
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
    application = FastAPI()
    application.state.logger = logging.getLogger("test.dbloc_guard")
    application.state.background_tasks = set()
    application.add_middleware(TraceIdMiddleware)
    application.include_router(job_routes_module.router)
    return application


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---- DBLocation output: 422 at submit ----


@pytest.mark.anyio
async def test_db_output_location_rejected_at_submit(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Submitting a job with a DBLocation output must return 422."""
    body = {
        "data": [
            {
                "name": "g",
                "workflow": "{}",
                "inputs": {"x": ["hello"]},
                "output_location": {
                    "type": "db",
                    "table": "public.results",
                    "column": "val",
                },
            }
        ]
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=body,
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 422
    assert "DBLocation output is not supported" in resp.json()["detail"]


# ---- DB input validates via acatalog_column_exists ----


@pytest.mark.anyio
async def test_db_input_validates_column_via_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_validate_db_location_live returns normalized DBLocation when column exists."""
    with patch(
        "lumilake_server.routes.jobs.acatalog_column_exists",
        new=AsyncMock(return_value=True),
    ):
        result = await _validate_db_location_live(
            DBLocation(type="db", table="myschema.mytable", column="user_id")
        )
    assert result.table == "myschema.mytable"
    assert result.column == "user_id"


@pytest.mark.anyio
async def test_db_input_column_not_found_raises_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_validate_db_location_live raises HTTPException(422) when column absent."""
    with patch(
        "lumilake_server.routes.jobs.acatalog_column_exists",
        new=AsyncMock(return_value=False),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _validate_db_location_live(
                DBLocation(type="db", table="public.tbl", column="missing_col")
            )
    assert exc_info.value.status_code == 422
    assert "missing_col" in exc_info.value.detail


@pytest.mark.anyio
async def test_db_input_empty_column_raises_422() -> None:
    """_validate_db_location_live raises 422 when column field is empty."""
    with pytest.raises(HTTPException) as exc_info:
        await _validate_db_location_live(
            DBLocation(type="db", table="public.tbl", column="  ")
        )
    assert exc_info.value.status_code == 422


@pytest.mark.anyio
async def test_db_input_invalid_identifier_raises_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An identifier the catalog client rejects (``ValueError``) surfaces as 422,
    not a 500 — malformed user input must not escape the validator."""
    with patch(
        "lumilake_server.routes.jobs.acatalog_column_exists",
        new=AsyncMock(side_effect=ValueError("invalid table identifier")),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _validate_db_location_live(
                DBLocation(type="db", table="bad-name.t", column="c")
            )
    assert exc_info.value.status_code == 422
    assert "invalid table identifier" in exc_info.value.detail


# ---- S3 input listing via alist_blob_keys ----


@pytest.mark.anyio
async def test_s3_input_expands_prefix_via_blob_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_resolve_s3_input_values should expand blob keys to s3://bucket/key URLs."""
    monkeypatch.setattr(jmod.envs, "S3_DATA_PREFIX", "mybucket/data")

    with patch(
        "lumilake_server.routes.jobs.alist_blob_keys",
        new=AsyncMock(
            return_value=["mybucket/data/inputs/a.txt", "mybucket/data/inputs/b.txt"]
        ),
    ):
        result = await _resolve_s3_input_values(
            input_name="my_input",
            location=S3Location(type="s3", prefix="inputs/"),
        )
    assert result == [
        "s3://mybucket/data/inputs/a.txt",
        "s3://mybucket/data/inputs/b.txt",
    ]


@pytest.mark.anyio
async def test_s3_input_empty_result_raises_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_resolve_s3_input_values must raise 422 when no keys are found."""
    monkeypatch.setattr(jmod.envs, "S3_DATA_PREFIX", "mybucket/data")

    with patch(
        "lumilake_server.routes.jobs.alist_blob_keys",
        new=AsyncMock(return_value=[]),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _resolve_s3_input_values(
                input_name="empty_input",
                location=S3Location(type="s3", prefix="empty/"),
            )
    assert exc_info.value.status_code == 422
    assert "empty_input" in exc_info.value.detail
