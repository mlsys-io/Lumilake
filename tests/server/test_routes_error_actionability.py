"""Coverage for the route-level error actionability surface.

Exercises:
- 422 (validation) vs 400 (syntax) split on ``/jobs`` and ``/jobs/preview``.
- The cancel 409 response body naming the actual terminal status.
- Empty-inputs rejection echoing parsed input names.
- YAML parse failures preserving line/column information.
- Request-scoped trace id middleware propagating into log records.
"""

import datetime as dt
import logging
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from lumid_hooks import PrincipalContext, ResourceRef
from lumilake.log import trace_id_var

import lumilake_server.utils.job_storage as job_storage_module
from lumilake_server import hooks
from lumilake_server.middleware import TraceIdMiddleware
from lumilake_server.parser import YamlParseError, parse_yaml_payload
from lumilake_server.routes import jobs as job_routes_module
from lumilake_server.routes.jobs import JobRecord, JobStatus
from lumilake_server.runtime.protocol import LumilakeResponse
from lumilake_server.schemas.io import S3Location
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

    def __init__(self) -> None:
        self.cancel_calls: list[str] = []

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
        self.cancel_calls.append(job_id)


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
def job_routes(monkeypatch: pytest.MonkeyPatch) -> Any:
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
    return job_routes_module


@pytest.fixture
def app(job_routes: Any) -> FastAPI:
    app = FastAPI()
    app.state.logger = logging.getLogger("test.routes_error_actionability")
    app.state.background_tasks = set()
    app.add_middleware(TraceIdMiddleware)
    app.include_router(job_routes.router)
    return app


def _submit_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "data": [
            {
                "name": "demo",
                "workflow": "{}",
                "inputs": {"input": ["hello"]},
                "output_location": {"type": "s3", "prefix": "demo/output.txt"},
            }
        ]
    }
    body.update(overrides)
    return body


@pytest.mark.anyio
async def test_invalid_json_body_returns_400(app: FastAPI) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            content="{not json",
            headers={
                "Authorization": "Bearer token",
                "Content-Type": "application/json",
            },
        )
    assert resp.status_code == 400
    assert "Invalid JSON body" in resp.json()["detail"]


@pytest.mark.anyio
async def test_invalid_yaml_workflow_keeps_400_with_line_col(app: FastAPI) -> None:
    body = _submit_body()
    body["data"][0]["workflow"] = "name: demo\n  bad: : :"
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=body,
            headers={
                "Authorization": "Bearer token",
                "Workflow-Format": "yaml",
            },
        )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "line" in detail and "column" in detail


@pytest.mark.anyio
async def test_pydantic_validation_returns_422(app: FastAPI) -> None:
    body = _submit_body()
    body["data"][0]["inputs"] = {"q": "single-string-not-list"}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=body,
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_unsupported_workflow_format_returns_422(app: FastAPI) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(),
            headers={
                "Authorization": "Bearer token",
                "Workflow-Format": "nonsense",
            },
        )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_invalid_status_filter_returns_422(app: FastAPI) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs?status=bogus",
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 422
    assert "invalid status filters" in resp.json()["detail"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "terminal_status",
    ["completed", "failed", "cancelled"],
)
async def test_cancel_409_body_names_actual_status(
    app: FastAPI, job_routes: Any, terminal_status: JobStatus
) -> None:
    job_id = f"req-{terminal_status}"
    record = JobRecord(
        job_id=job_id,
        status=terminal_status,
        submitted_at=dt.datetime.now(dt.UTC).isoformat(),
        inputs={},
        output_location={"out": S3Location(type="s3", prefix="x/y")},
        org_id="demo",
        user_id="alice@example.com",
    )
    job_routes.jobs[job_id] = record
    job_routes._job_storage.save(record)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/jobs/{job_id}/cancel",
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert isinstance(detail, dict)
    assert detail["status"] == terminal_status
    assert detail["job_id"] == job_id
    assert detail["message"] == "job already finished"


def test_yaml_parse_error_preserves_line_and_column() -> None:
    bad_yaml = "name: demo\n  bad: : :"
    with pytest.raises(YamlParseError) as info:
        parse_yaml_payload(bad_yaml)
    err = info.value
    assert err.line is not None
    assert err.column is not None
    assert "line" in str(err) and "column" in str(err)


def test_yaml_parse_error_is_a_value_error() -> None:
    # Existing callers that catch ``ValueError`` still work.
    with pytest.raises(ValueError):
        parse_yaml_payload("name: demo\n  bad: : :")


@pytest.mark.anyio
async def test_trace_id_middleware_uses_inbound_header(app: FastAPI) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs",
            headers={
                "Authorization": "Bearer token",
                "X-Request-ID": "trace-from-client",
            },
        )
    assert resp.status_code == 200
    assert resp.headers["X-Request-ID"] == "trace-from-client"


@pytest.mark.anyio
async def test_trace_id_middleware_mints_when_missing(app: FastAPI) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/jobs",
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 200
    assert resp.headers["X-Request-ID"]
    assert resp.headers["X-Request-ID"].startswith("req-")


@pytest.mark.anyio
async def test_trace_id_propagates_into_log_records(
    app: FastAPI, caplog: pytest.LogCaptureFixture
) -> None:
    """The TraceIdFilter must attach ``trace_id`` to records emitted while a
    request is in flight. We attach a recording handler with the filter, hit
    an endpoint that logs from the route layer, and assert the captured
    record carries the inbound header value.
    """
    from lumilake.log import TraceIdFilter

    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _Capture()
    handler.addFilter(TraceIdFilter())

    # The trace ID middleware sets the context var while the handler runs;
    # log directly inside a request-scoped dependency to confirm propagation.
    request_logger = logging.getLogger("test.trace_id_propagation")
    request_logger.addHandler(handler)
    request_logger.setLevel(logging.INFO)

    from fastapi import Request as FastapiRequest

    @app.get("/__trace_test")
    async def _probe(request: FastapiRequest) -> dict[str, str]:
        request_logger.info("hit probe")
        return {"trace_id": trace_id_var.get()}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/__trace_test",
            headers={"X-Request-ID": "trace-ctx-1"},
        )
    request_logger.removeHandler(handler)
    assert resp.status_code == 200
    assert resp.json()["trace_id"] == "trace-ctx-1"
    matched = [r for r in captured if r.getMessage() == "hit probe"]
    assert matched, "probe log not captured"
    assert getattr(matched[0], "trace_id") == "trace-ctx-1"


@pytest.mark.anyio
async def test_inputs_required_validator_errors_include_input_path(
    app: FastAPI,
) -> None:
    body = _submit_body()
    body["data"][0]["inputs"] = {}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=body,
            headers={"Authorization": "Bearer token"},
        )
    # Pydantic rejects empty inputs at the schema layer; that's a validation
    # failure, hence 422.
    assert resp.status_code == 422
    assert "inputs" in resp.json()["detail"]


@pytest.mark.anyio
async def test_empty_inputs_after_resolution_echoes_input_name(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single-input request whose value list is empty after resolution
    surfaces the offending input name in the 422 message, so a CLI user can
    tell which ``--input`` flag misfired.
    """

    async def _noop_validate(*args: Any, **kwargs: Any) -> Any:
        loc = kwargs.get("location")
        if loc is None and args:
            loc = args[0]
        return loc

    async def _noop_perm(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(job_routes_module, "_validate_location", _noop_validate)
    monkeypatch.setattr(job_routes_module, "_require_location_permission", _noop_perm)

    async def _resolve_empty_raw(
        *,
        input_name: str,
        raw: Any,
        principal: Any,
        hook_logger: Any,
    ) -> list[str]:
        return []

    monkeypatch.setattr(
        job_routes_module, "_resolve_input_values_raw", _resolve_empty_raw
    )

    body = _submit_body()
    body["data"][0]["inputs"] = {"my_input": ["any"]}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=body,
            headers={"Authorization": "Bearer token"},
        )
    assert resp.status_code == 422
    assert resp.json()["detail"] == {
        "message": "input 'my_input' resolved to an empty value list",
        "parsed_input_names": ["my_input"],
    }


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
