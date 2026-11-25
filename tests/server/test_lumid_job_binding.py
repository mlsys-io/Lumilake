"""Binding a submission to a lum.id/data job: validation and token fetch."""

import uuid
from typing import Any

import httpx
import pytest
from fastapi import HTTPException
from lumid_hooks import PrincipalContext
from pydantic import ValidationError

from lumilake import envs
from lumilake_server.hooks.security import require_scope
from lumilake_server.routes import jobs as jobs_routes
from lumilake_server.routes.jobs import (
    LUMID_JOB_BIND_SCOPE,
    JobSubmitRequest,
    LumidJobBinding,
)

_JOB_ID = "3f2b8c1e-8d4a-4f6b-9c2d-1a2b3c4d5e6f"
_JOB_UUID = uuid.UUID(_JOB_ID)


def test_binding_requires_a_uuid_job_id() -> None:
    assert str(LumidJobBinding(job_id=_JOB_UUID).job_id) == _JOB_ID
    with pytest.raises(ValidationError):
        LumidJobBinding.model_validate({"job_id": "../../admin"})


def test_binding_refuses_a_data_url() -> None:
    with pytest.raises(ValidationError, match="data_url"):
        LumidJobBinding.model_validate({"job_id": _JOB_ID, "data_url": "https://x"})


def test_submit_request_accepts_an_optional_binding() -> None:
    req = JobSubmitRequest.model_validate(
        {
            "data": [
                {
                    "workflow": "{}",
                    "inputs": {"x": ["1"]},
                    "output_location": {"type": "s3", "prefix": "out/"},
                }
            ],
            "lumid_job": {"job_id": _JOB_ID},
        }
    )
    assert req.lumid_job is not None
    assert str(req.lumid_job.job_id) == _JOB_ID
    unbound = JobSubmitRequest.model_validate(
        {
            "data": [
                {
                    "workflow": "{}",
                    "inputs": {"x": ["1"]},
                    "output_location": {"type": "s3", "prefix": "out/"},
                }
            ]
        }
    )
    assert unbound.lumid_job is None


def _patch_client(monkeypatch: pytest.MonkeyPatch, handler: Any) -> list[httpx.Request]:
    seen: list[httpx.Request] = []
    real = httpx.AsyncClient

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(wrapped)
        return real(*args, **kwargs)

    monkeypatch.setattr(jobs_routes.httpx, "AsyncClient", factory)
    return seen


@pytest.fixture
def jobs_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_JOBS_URL", "https://lum.id/data")
    monkeypatch.setattr(envs, "LUMID_JOBS_RUNTIME_KEY", "rt-key")
    monkeypatch.setattr(envs, "LUMID_DATA_WORKER_URL", "https://lum.id/data")


@pytest.mark.asyncio
async def test_token_is_fetched_with_the_runtime_key(
    monkeypatch: pytest.MonkeyPatch, jobs_env: None
) -> None:
    seen = _patch_client(
        monkeypatch,
        lambda r: httpx.Response(200, json={"token": "ljt_abc", "job_id": _JOB_ID}),
    )
    cred = await jobs_routes._lumid_job_credential(LumidJobBinding(job_id=_JOB_UUID))
    assert cred.token == "ljt_abc"
    assert cred.url == "https://lum.id/data"
    assert str(seen[0].url) == f"https://lum.id/data/jobs/v1/jobs/{_JOB_ID}/token"
    assert seen[0].headers["authorization"] == "Bearer rt-key"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(404, json={}),
        httpx.Response(200, json={"token": ""}),
        httpx.Response(200, json=["token"]),
        httpx.Response(200, content=b"not json"),
    ],
)
async def test_refused_or_malformed_token_is_a_502(
    monkeypatch: pytest.MonkeyPatch, jobs_env: None, response: httpx.Response
) -> None:
    _patch_client(monkeypatch, lambda r: response)
    with pytest.raises(HTTPException) as exc:
        await jobs_routes._lumid_job_credential(LumidJobBinding(job_id=_JOB_UUID))
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_unreachable_jobs_api_is_a_502(
    monkeypatch: pytest.MonkeyPatch, jobs_env: None
) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    _patch_client(monkeypatch, boom)
    with pytest.raises(HTTPException) as exc:
        await jobs_routes._lumid_job_credential(LumidJobBinding(job_id=_JOB_UUID))
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_missing_runtime_key_is_refused_before_any_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "LUMID_JOBS_RUNTIME_KEY", None)
    seen = _patch_client(
        monkeypatch, lambda r: httpx.Response(200, json={"token": "t"})
    )
    with pytest.raises(HTTPException) as exc:
        await jobs_routes._lumid_job_credential(LumidJobBinding(job_id=_JOB_UUID))
    assert exc.value.status_code == 422
    assert seen == []


def _principal(scopes: list[str]) -> PrincipalContext:
    return PrincipalContext(
        principal_id="p",
        org_id="o",
        external_id="e",
        principal_type="user",
        scopes=scopes,
    )


def test_binding_requires_the_bind_scope() -> None:
    require_scope(_principal(["lumid-job:bind"]), LUMID_JOB_BIND_SCOPE)
    require_scope(_principal(["*"]), LUMID_JOB_BIND_SCOPE)
    with pytest.raises(HTTPException) as exc:
        require_scope(_principal(["jobs:read"]), LUMID_JOB_BIND_SCOPE)
    assert exc.value.status_code == 403
    assert LUMID_JOB_BIND_SCOPE in exc.value.detail
