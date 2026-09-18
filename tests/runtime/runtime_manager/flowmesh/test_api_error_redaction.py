from typing import Any

import pytest
from flowmesh.exceptions import APIError

from lumilake_server.runtime.runtime_manager.flowmesh import (
    FlowmeshRuntimeManager,
    _sanitize_flowmesh_api_error,
)


def test_sanitize_flowmesh_api_error_redacts_authorization_header_in_body() -> None:
    """FlowMesh's rejection body can echo back the task spec we submitted,
    Authorization header included. _sanitize_flowmesh_api_error must strip
    that credential from both the body and the exception message before the
    error is logged or re-raised."""
    original = APIError(
        "task spec invalid",
        status_code=422,
        method="POST",
        url="https://flowmesh.internal/workflows",
        body={
            "detail": "invalid task",
            "spec": {"Authorization": "Bearer sk-live-secret"},
        },
    )

    sanitized = _sanitize_flowmesh_api_error(original)

    assert isinstance(sanitized, APIError)
    assert "sk-live-secret" not in str(sanitized.body)
    assert "sk-live-secret" not in str(sanitized)
    assert "***REDACTED***" in str(sanitized.body)
    assert sanitized.status_code == 422
    assert sanitized.method == "POST"
    assert sanitized.url == "https://flowmesh.internal/workflows"


def test_sanitize_flowmesh_api_error_redacts_bearer_token_in_plain_text_body() -> None:
    """A non-JSON rejection body (plain text) can still embed a bearer
    token; the regex fallback in redact_secrets_in_text must catch it."""
    original = APIError(
        "rejected: Authorization: Bearer sk-live-secret is not permitted",
        status_code=403,
        method="POST",
        url="https://flowmesh.internal/workflows",
        body="rejected: Authorization: Bearer sk-live-secret is not permitted",
    )

    sanitized = _sanitize_flowmesh_api_error(original)

    assert "sk-live-secret" not in str(sanitized.body)
    assert "sk-live-secret" not in str(sanitized)
    assert "***REDACTED***" in str(sanitized.body)


def test_sanitize_flowmesh_api_error_truncates_long_body() -> None:
    """A pathologically large echoed body must be truncated so it cannot
    blow up log lines or the persisted job error record."""
    original = APIError(
        "task spec invalid",
        status_code=422,
        method="POST",
        url="https://flowmesh.internal/workflows",
        body={"detail": "x" * 5000},
    )

    sanitized = _sanitize_flowmesh_api_error(original)

    assert len(sanitized.body) <= 2000 + len("...[truncated]")
    assert sanitized.body.endswith("...[truncated]")


@pytest.mark.asyncio
async def test_fetch_task_status_sanitizes_api_error_before_reraising(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A FlowMesh rejection on status retrieval can echo back the submitted
    task spec, Authorization header included. fetch_task_status must re-raise
    a sanitized APIError so the credential never reaches the caller, the
    log, or the persisted job error record."""
    from flowmesh.exceptions import APIError

    class _FakeTasks:
        async def retrieve(self, task_id: str) -> Any:
            raise APIError(
                "task spec invalid",
                status_code=422,
                method="GET",
                url="https://flowmesh.internal/tasks/t1",
                body={"spec": {"Authorization": "Bearer sk-live-secret"}},
            )

    class _FakeFm:
        def __init__(self) -> None:
            self.tasks = _FakeTasks()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

    with pytest.raises(APIError) as excinfo:
        await flowmesh_manager.fetch_task_status("t1")

    assert "sk-live-secret" not in str(excinfo.value)
    assert "sk-live-secret" not in str(excinfo.value.body)
