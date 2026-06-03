"""Tests for /optimizer/schedule auth gate.

Covers:
- Insufficient credentials (unauthenticated or no JOB:WRITE) → 4xx rejection.
- Authenticated bearer WITH JOB:WRITE permission → 200.
- /optimizer is not gated by the schedule auth (different route contract).
"""

import logging
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from lumid_hooks import PrincipalContext, ResourceRef
from lumilake import envs

from lumilake_server import hooks
from lumilake_server.routes import optimizer as optimizer_routes
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp

# ---------------------------------------------------------------------------
# Stub identity provider
# ---------------------------------------------------------------------------


class _StubIdentityProvider:
    name = "stub.identity"

    def __init__(self, tokens: dict[str, PrincipalContext]) -> None:
        self._tokens = tokens

    async def resolve(
        self, raw_token: str, logger: logging.Logger
    ) -> PrincipalContext | None:
        return self._tokens.get(raw_token)


# ---------------------------------------------------------------------------
# Stub permission checker that enforces a deny-list
# ---------------------------------------------------------------------------


class _StubPermissionChecker:
    name = "stub.permissions"

    def __init__(self, denied_principals: set[str]) -> None:
        self._denied = denied_principals

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
        if principal.principal_id in self._denied:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="permission denied",
            )


# ---------------------------------------------------------------------------
# Principals
# ---------------------------------------------------------------------------

_PRINCIPAL_WITH_WRITE = PrincipalContext(
    principal_id="alice",
    org_id="org",
    external_id="alice@example.com",
    principal_type="user",
    scopes=["*"],
)
_PRINCIPAL_WITHOUT_WRITE = PrincipalContext(
    principal_id="bob",
    org_id="org",
    external_id="bob@example.com",
    principal_type="user",
    scopes=["read"],
)

_TOKENS = {
    "token-alice": _PRINCIPAL_WITH_WRITE,
    "token-bob": _PRINCIPAL_WITHOUT_WRITE,
}
_DENIED_PRINCIPALS = {"bob"}


# ---------------------------------------------------------------------------
# Minimal valid ScheduleRequest body
# ---------------------------------------------------------------------------


def _schedule_body() -> dict[str, Any]:
    """Minimal body that passes ScheduleRequest validation."""
    op = RuntimeOp(
        node_id="n1",
        task_type="inference",
        backend="vllm",
        model="gpt2",
        data_spec={},
        model_spec={},
        inference_spec={},
        dependencies=(),
    )
    graph = RuntimeGraph(
        nodes={"n1": op},
        node_order=["n1"],
        output_node_map={"n1": "out"},
        dsl_to_runtime={},
    )
    return {
        "graph": graph.serialize(),
        "worker_names": ["w1"],
        "worker_profiles": {"w1": {}},
        "optimizer_type": "halo",
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_hooks():
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


@pytest.fixture()
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setattr(envs, "LUMILAKE_REQUIRE_IDENTITY_PROVIDER", True)
    hooks.IDENTITY_PROVIDERS.append(_StubIdentityProvider(_TOKENS))
    hooks.PERMISSION_CHECKERS.append(_StubPermissionChecker(_DENIED_PRINCIPALS))

    _app = FastAPI()
    _app.state.logger = logging.getLogger("test.optimizer_schedule_auth")
    _app.include_router(optimizer_routes.router)
    return _app


@pytest.fixture()
def mock_optimizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace create_optimizer so schedule generation always succeeds."""

    class _AlwaysSuccessOptimizer:
        def generate_schedule(self, *args: Any, **kwargs: Any) -> Schedule:
            return Schedule(worker_assignment={"w1": ["n1"]})

    monkeypatch.setattr(
        optimizer_routes,
        "create_optimizer",
        lambda **kwargs: _AlwaysSuccessOptimizer(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "headers,expected_status",
    [
        pytest.param({}, 401, id="unauthenticated"),
        pytest.param({"Authorization": "Bearer token-bob"}, 403, id="no_job_write"),
    ],
)
def test_schedule_rejects_insufficient_credentials(
    app: FastAPI, headers: dict[str, str], expected_status: int
) -> None:
    client = TestClient(app, raise_server_exceptions=True)
    resp = client.post("/optimizer/schedule", json=_schedule_body(), headers=headers)
    assert resp.status_code == expected_status


def test_authenticated_with_job_write_succeeds(
    app: FastAPI, mock_optimizer: None
) -> None:
    client = TestClient(app, raise_server_exceptions=True)
    resp = client.post(
        "/optimizer/schedule",
        json=_schedule_body(),
        headers={"Authorization": "Bearer token-alice"},
    )
    assert resp.status_code == 200
    assert "worker_assignment" in resp.json()


def test_list_endpoint_not_gated_by_schedule_auth(app: FastAPI) -> None:
    """/optimizer must respond without JOB:WRITE — it has no permission gate."""
    client = TestClient(app, raise_server_exceptions=True)
    resp = client.get(
        "/optimizer",
        headers={"Authorization": "Bearer token-bob"},
    )
    assert resp.status_code == 200
    assert "types" in resp.json()
