import asyncio
import logging
from collections.abc import Iterator
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from lumid_hooks import PrincipalContext, ResourceRef
from lumilake_hook import ResourceAction, ResourceKind, UsageRow

import lumilake_server.utils.job_storage as job_storage_module
from examples.plugins import simple_plugin
from examples.plugins.simple_plugin import state as simple_plugin_state
from lumilake_server import hooks
from lumilake_server.routes import jobs as job_routes_module
from lumilake_server.routes import trace as trace_routes
from lumilake_server.runtime.optimizer import create_optimizer
from lumilake_server.runtime.protocol import LumilakeResponse
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp
from lumilake_server.utils.job_storage import InMemoryJobStorage

_DEFAULT_TOKENS = {
    token: principal.model_copy(deep=True)
    for token, principal in simple_plugin_state.TOKENS.items()
}


class _RecordingRuntimeManager:
    def __init__(self) -> None:
        self.dispatch_token_sets: list[tuple[str, str | None]] = []
        self.dispatch_token_clears: list[str] = []

    def set_dispatch_token(self, request_id: str, token: str | None) -> None:
        self.dispatch_token_sets.append((request_id, token))

    def get_dispatch_token(self, request_id: str) -> str | None:
        return None

    def clear_dispatch_token(self, request_id: str) -> None:
        self.dispatch_token_clears.append(request_id)

    def release_executions(self, execution_ids: set[str]) -> None:
        return None


class _FakeRuntimeServer:
    is_started = True

    def __init__(self) -> None:
        self.cancel_calls: list[str] = []
        self.runtime_manager = _RecordingRuntimeManager()

    def parse_query(self, graph_specs: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return graph_specs

    async def execute(
        self,
        graphs: dict[str, Any],
        request_id: str | None = None,
        config: Any | None = None,
        workflow_slices: dict[str, Any] | None = None,
    ) -> LumilakeResponse:
        return LumilakeResponse(
            outputs={
                "demo": {
                    "output": ["ok"],
                    "image": ["data:image/png;base64,aGVsbG8="],
                }
            }
        )

    def trace_ids_for_request(self, job_id: str) -> list[str]:
        return [f"trace-{job_id}"]

    def optimization_seconds_for_request(self, job_id: str) -> float:
        return 0.01

    def release_request_workflows(self, job_id: str) -> None:
        self.runtime_manager.clear_dispatch_token(job_id)

    async def cancel_request(self, job_id: str) -> None:
        self.cancel_calls.append(job_id)


@pytest.fixture(autouse=True)
def _reset_hook_state() -> Iterator[None]:
    hooks.IDENTITY_PROVIDERS.clear()
    hooks.SUBMISSION_GUARDS.clear()
    hooks.USAGE_SINKS.clear()
    hooks.PERMISSION_CHECKERS.clear()
    hooks.RESOURCE_REGISTRARS.clear()
    yield
    hooks.IDENTITY_PROVIDERS.clear()
    hooks.SUBMISSION_GUARDS.clear()
    hooks.USAGE_SINKS.clear()
    hooks.PERMISSION_CHECKERS.clear()
    hooks.RESOURCE_REGISTRARS.clear()


@pytest.fixture
def sample_plugin() -> Iterator[Any]:
    simple_plugin_state.TOKENS.clear()
    simple_plugin_state.TOKENS.update(
        {
            token: principal.model_copy(deep=True)
            for token, principal in _DEFAULT_TOKENS.items()
        }
    )
    simple_plugin_state.BLOCKED_PRINCIPALS.clear()
    simple_plugin_state.OWNERSHIP.clear()
    simple_plugin_state.USAGE_LEDGER.clear()
    hooks.register(simple_plugin.install())
    yield simple_plugin_state
    simple_plugin_state.TOKENS.clear()
    simple_plugin_state.TOKENS.update(
        {
            token: principal.model_copy(deep=True)
            for token, principal in _DEFAULT_TOKENS.items()
        }
    )
    simple_plugin_state.BLOCKED_PRINCIPALS.clear()
    simple_plugin_state.OWNERSHIP.clear()
    simple_plugin_state.USAGE_LEDGER.clear()


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
    monkeypatch.setattr(
        job_routes_module,
        "build_request_data_profile_tasks",
        lambda **kwargs: [],
    )

    async def _dump_output_locations(**kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        job_routes_module,
        "_dump_output_locations",
        _dump_output_locations,
    )
    return job_routes_module


@pytest.fixture
def app(job_routes: Any) -> FastAPI:
    app = FastAPI()
    app.state.logger = logging.getLogger("test.simple_plugin_e2e")
    app.state.compute_db_pool = None
    app.state.background_tasks = set()
    app.include_router(job_routes.router)
    app.include_router(trace_routes.router)
    return app


async def _wait_for_completed_job(job_routes: Any, job_id: str) -> None:
    for _ in range(100):
        record = await job_routes._load_job_record(job_id)
        if record is not None and record.status == "completed":
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} did not complete")


def _submit_payload() -> dict[str, Any]:
    return {
        "data": [
            {
                "name": "demo",
                "workflow": "{}",
                "inputs": {"input": ["hello"]},
                "output_location": {"type": "s3", "prefix": "demo/output.txt"},
            }
        ]
    }


@pytest.mark.anyio
async def test_sample_plugin_exercises_every_plugin_component(
    sample_plugin: Any,
) -> None:
    logger = logging.getLogger("test.simple_plugin_components")

    assert [provider.name for provider in hooks.IDENTITY_PROVIDERS] == [
        "simple_plugin.identity"
    ]
    assert [guard.name for guard in hooks.SUBMISSION_GUARDS] == [
        "simple_plugin.submission"
    ]
    assert [sink.name for sink in hooks.USAGE_SINKS] == ["simple_plugin.usage"]
    assert [checker.name for checker in hooks.PERMISSION_CHECKERS] == [
        "simple_plugin.permissions"
    ]
    assert [registrar.name for registrar in hooks.RESOURCE_REGISTRARS] == [
        "simple_plugin.registrar"
    ]

    identity = hooks.IDENTITY_PROVIDERS[0]
    admin = await identity.resolve("demo-admin", logger)
    user = await identity.resolve("demo-user", logger)
    unknown = await identity.resolve("unknown", logger)
    assert admin is not None
    assert admin.principal_id == "alice"
    assert user is not None
    assert user.principal_id == "bob"
    assert unknown is None

    guard = hooks.SUBMISSION_GUARDS[0]
    await guard.check(user, logger)
    sample_plugin.BLOCKED_PRINCIPALS.add(user.principal_id)
    with pytest.raises(HTTPException) as blocked:
        await guard.check(user, logger)
    assert blocked.value.status_code == 403
    assert blocked.value.detail == "principal is blocked"
    sample_plugin.BLOCKED_PRINCIPALS.clear()

    registrar = hooks.RESOURCE_REGISTRARS[0]
    job_ref = ResourceRef(kind=ResourceKind.JOB.value, id="job-1")
    await registrar.register(user, job_ref, logger)
    assert sample_plugin.OWNERSHIP[(ResourceKind.JOB.value, "job-1")] == "bob"

    checker = hooks.PERMISSION_CHECKERS[0]
    assert await checker.accessible_ids(
        user,
        ResourceKind.JOB.value,
        ResourceAction.READ.value,
        logger,
    ) == frozenset({"job-1"})
    assert (
        await checker.accessible_ids(
            admin,
            ResourceKind.JOB.value,
            ResourceAction.READ.value,
            logger,
        )
        is None
    )
    await checker.require(user, job_ref, ResourceAction.READ.value, logger)
    await checker.require(admin, job_ref, ResourceAction.ADMIN.value, logger)

    no_scope = PrincipalContext(
        principal_id="dana",
        org_id="demo",
        external_id="dana@example.com",
        principal_type="user",
        scopes=[],
    )
    with pytest.raises(HTTPException) as no_scope_denied:
        await checker.require(
            no_scope,
            ResourceRef(kind=ResourceKind.JOB.value),
            ResourceAction.READ.value,
            logger,
        )
    assert no_scope_denied.value.status_code == 403
    assert no_scope_denied.value.detail == "principal has no scopes"

    no_data = PrincipalContext(
        principal_id="charlie",
        org_id="demo",
        external_id="charlie@example.com",
        principal_type="user",
        scopes=["user"],
    )
    with pytest.raises(HTTPException) as data_denied:
        await checker.require(
            no_data,
            ResourceRef(kind=ResourceKind.TABLE.value, id="public.orders"),
            ResourceAction.READ.value,
            logger,
        )
    assert data_denied.value.status_code == 403
    assert data_denied.value.detail == "principal may not access data resources"

    usage_row: UsageRow = {
        "org_id": "demo",
        "principal_id": user.principal_id,
        "job_id": "job-1",
        "status": "completed",
    }
    await hooks.USAGE_SINKS[0].emit([usage_row], logger)
    assert sample_plugin.USAGE_LEDGER == [usage_row]

    await registrar.deregister(user, job_ref, logger)
    assert (ResourceKind.JOB.value, "job-1") not in sample_plugin.OWNERSHIP

    optimizer = create_optimizer("simple")
    graph = RuntimeGraph(
        nodes={
            "node-a": RuntimeOp(
                node_id="node-a",
                task_type="llm",
                backend="mock",
                model="mock",
                data_spec={},
                model_spec={},
                inference_spec={},
            ),
            "node-b": RuntimeOp(
                node_id="node-b",
                task_type="llm",
                backend="mock",
                model="mock",
                data_spec={},
                model_spec={},
                inference_spec={},
                dependencies=("node-a",),
            ),
        },
        node_order=["node-a", "node-b"],
        output_node_map={},
    )
    schedule = optimizer.generate_schedule(graph, ["worker-1", "worker-2"], {})
    assert schedule.worker_assignment == {
        "worker-1": ["node-a"],
        "worker-2": ["node-b"],
    }
    with pytest.raises(ValueError, match="requires at least one worker"):
        optimizer.generate_schedule(graph, [], {})


@pytest.mark.anyio
async def test_sample_plugin_runs_job_flow_and_records_hook_effects(
    app: FastAPI,
    job_routes: Any,
    sample_plugin: Any,
) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        unauthenticated = await client.get("/jobs")
        assert unauthenticated.status_code == 401

        submit = await client.post(
            "/jobs",
            json=_submit_payload(),
            headers={"Authorization": "Bearer demo-user"},
        )
        assert submit.status_code == 200
        job_id = submit.json()["data"]["job_id"]
        await _wait_for_completed_job(job_routes, job_id)

        list_response = await client.get(
            "/jobs",
            headers={"Authorization": "Bearer demo-user"},
        )
        assert list_response.status_code == 200
        assert [item["job_id"] for item in list_response.json()["data"]["items"]] == [
            job_id
        ]

        detail = await client.get(
            f"/jobs/{job_id}",
            headers={"Authorization": "Bearer demo-user"},
        )
        assert detail.status_code == 200
        result_response = await client.get(
            f"/jobs/{job_id}/result",
            headers={"Authorization": "Bearer demo-user"},
        )
        assert result_response.status_code == 200
        artifact_uri = result_response.json()["data"]["result"]["outputs"]["demo"][
            "image"
        ][0]
        artifact_name = job_routes._artifact_name_from_uri(artifact_uri)
        assert artifact_name is not None
        artifact = await client.get(
            f"/jobs/{job_id}/artifact",
            params={"path": artifact_uri},
            headers={"Authorization": "Bearer demo-user"},
        )
        assert artifact.status_code == 200
        assert artifact.content == b"hello"
        traces = await client.get(
            "/trace",
            headers={"Authorization": "Bearer demo-user"},
        )
        assert traces.status_code == 200
        assert [item["trace_id"] for item in traces.json()["data"]["items"]] == [
            f"trace-{job_id}"
        ]
        record = await job_routes._load_job_record(job_id)
        assert record is not None
        assert record.org_id == "demo"
        assert record.user_id == "bob@example.com"

    assert sample_plugin.OWNERSHIP[("job", job_id)] == "bob"
    assert sample_plugin.OWNERSHIP[("trace", f"trace-{job_id}")] == "bob"
    assert sample_plugin.OWNERSHIP[("artifact", f"{job_id}/{artifact_name}")] == "bob"
    assert sample_plugin.USAGE_LEDGER[-1]["job_id"] == job_id
    assert sample_plugin.USAGE_LEDGER[-1]["status"] == "completed"


@pytest.mark.anyio
async def test_submit_forwards_bearer_to_runtime_dispatch_token(
    app: FastAPI,
    job_routes: Any,
    sample_plugin: Any,
) -> None:
    runtime_manager = cast(
        _RecordingRuntimeManager,
        job_routes_module.LumilakeServer.get_started_instance().runtime_manager,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        submit = await client.post(
            "/jobs",
            json=_submit_payload(),
            headers={"Authorization": "Bearer demo-user"},
        )
        assert submit.status_code == 200
        job_id = submit.json()["data"]["job_id"]
        await _wait_for_completed_job(job_routes, job_id)

    assert (job_id, "demo-user") in runtime_manager.dispatch_token_sets
    assert job_id in runtime_manager.dispatch_token_clears


@pytest.mark.anyio
async def test_submit_without_bearer_dispatches_with_no_token(
    app: FastAPI,
    job_routes: Any,
) -> None:
    runtime_manager = cast(
        _RecordingRuntimeManager,
        job_routes_module.LumilakeServer.get_started_instance().runtime_manager,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        submit = await client.post("/jobs", json=_submit_payload())
        assert submit.status_code == 200
        job_id = submit.json()["data"]["job_id"]
        await _wait_for_completed_job(job_routes, job_id)

    assert (job_id, None) in runtime_manager.dispatch_token_sets


@pytest.mark.anyio
async def test_sample_plugin_submission_and_data_guards_block_requests(
    app: FastAPI,
    sample_plugin: Any,
) -> None:
    sample_plugin.BLOCKED_PRINCIPALS.add("bob")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        blocked = await client.post(
            "/jobs",
            json=_submit_payload(),
            headers={"Authorization": "Bearer demo-user"},
        )
        assert blocked.status_code == 403
        assert blocked.json()["detail"] == "principal is blocked"

        sample_plugin.BLOCKED_PRINCIPALS.clear()
        sample_plugin.TOKENS["demo-nodata"] = PrincipalContext(
            principal_id="charlie",
            org_id="demo",
            external_id="charlie@example.com",
            principal_type="user",
            scopes=["user"],
        )
        denied = await client.post(
            "/jobs",
            json=_submit_payload(),
            headers={"Authorization": "Bearer demo-nodata"},
        )
        assert denied.status_code == 403
        assert denied.json()["detail"] == "principal may not access data resources"


@pytest.mark.anyio
async def test_sample_plugin_list_endpoints_require_type_level_read(
    app: FastAPI,
    sample_plugin: Any,
) -> None:
    sample_plugin.TOKENS["demo-noscope"] = PrincipalContext(
        principal_id="dana",
        org_id="demo",
        external_id="dana@example.com",
        principal_type="user",
        scopes=[],
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        jobs = await client.get(
            "/jobs",
            headers={"Authorization": "Bearer demo-noscope"},
        )
        assert jobs.status_code == 403
        assert jobs.json()["detail"] == "principal has no scopes"

        traces = await client.get(
            "/trace",
            headers={"Authorization": "Bearer demo-noscope"},
        )
        assert traces.status_code == 403
        assert traces.json()["detail"] == "principal has no scopes"
