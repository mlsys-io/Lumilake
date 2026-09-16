from flowmesh.exceptions import APIError

from lumilake_server.runtime.runtime_manager.flowmesh import (
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
