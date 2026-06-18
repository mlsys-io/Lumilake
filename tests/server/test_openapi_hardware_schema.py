"""Regression: the hand-written ``openapi_extra`` request schemas for
``POST /api/v1/jobs`` and ``POST /api/v1/jobs/preview`` must expose the
``hardware`` field so generated clients and ``/docs`` users can discover
it. The route models accept ``hardware``; without this schema entry the
field is invisible at the contract boundary.
"""

from typing import Any

import pytest
from fastapi import FastAPI

from lumilake_server.routes.jobs import _OPENAPI_HARDWARE_SCHEMA, router


@pytest.fixture
def app() -> FastAPI:
    application = FastAPI()
    application.include_router(router)
    return application


def _request_schema(spec: dict[str, Any], path: str) -> dict[str, Any]:
    post = spec["paths"][path]["post"]
    body = post["requestBody"]["content"]["application/json"]["schema"]
    return body


def test_submit_jobs_openapi_includes_hardware_field(app: FastAPI) -> None:
    spec = app.openapi()
    schema = _request_schema(spec, "/jobs")
    assert "hardware" in schema["properties"]
    assert schema["properties"]["hardware"]["type"] == "object"


def test_preview_jobs_openapi_includes_hardware_field(app: FastAPI) -> None:
    spec = app.openapi()
    schema = _request_schema(spec, "/jobs/preview")
    assert "hardware" in schema["properties"]
    assert schema["properties"]["hardware"]["type"] == "object"


def test_hardware_schema_advertises_all_four_fields() -> None:
    """If any of cpu/memory/gpu/gpu_memory disappears from the published
    schema, generated clients silently lose support for that knob."""
    properties = _OPENAPI_HARDWARE_SCHEMA["properties"]
    assert set(properties) == {"cpu", "memory", "gpu", "gpu_memory"}


def test_hardware_schema_documents_role_split() -> None:
    """The published schema must spell out that CPU/memory filter every
    worker but gpu/gpu_memory only filter GPU-capable workers — otherwise
    users will read ``--gpu 1`` as a job-wide GPU-only switch."""
    description = _OPENAPI_HARDWARE_SCHEMA["description"]
    assert "GPU-capable workers" in description
    assert "CPU workers" in description


def test_hardware_schema_documents_gpu_zero_rejection() -> None:
    """The HTTP 422 fail-fast for ``gpu=0`` + a GPU op must be visible at
    the contract boundary, not buried in the changelog."""
    description = _OPENAPI_HARDWARE_SCHEMA["description"]
    assert "gpu=0" in description
    assert "422" in description
