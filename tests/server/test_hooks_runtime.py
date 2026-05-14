import logging
from collections.abc import Sequence

import pytest
from lumid_hooks import BaseBindings, PrincipalContext, ResourceRef
from lumilake_hook import ResourceAction, ResourceKind, UsageRow

from lumilake_server import hooks
from lumilake_server.hooks.security import emit_usage, resolve_accessible_ids


class _Checker:
    name = "checker"

    def __init__(self, ids: frozenset[str] | None) -> None:
        self.ids = ids

    async def accessible_ids(
        self,
        principal: PrincipalContext,
        kind: str,
        action: str,
        logger: logging.Logger,
    ) -> frozenset[str] | None:
        return self.ids

    async def require(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        action: str,
        logger: logging.Logger,
    ) -> None:
        return None


class _FailingSink:
    name = "failing"

    async def emit(self, rows: Sequence[UsageRow], logger: logging.Logger) -> None:
        raise RuntimeError("sink failed")


class _RecordingSink:
    name = "recording"

    def __init__(self) -> None:
        self.rows: list[UsageRow] = []

    async def emit(self, rows: Sequence[UsageRow], logger: logging.Logger) -> None:
        self.rows.extend(rows)


@pytest.fixture(autouse=True)
def _clear_hooks():
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


def _principal() -> PrincipalContext:
    return PrincipalContext(
        principal_id="p-1",
        org_id="org-1",
        external_id="user-1",
        principal_type="user",
        scopes=[],
    )


@pytest.mark.asyncio
async def test_accessible_ids_intersects_concrete_checker_results() -> None:
    hooks.register(
        BaseBindings(
            permission_checkers=(
                _Checker(frozenset({"job-1", "job-2"})),
                _Checker(None),
                _Checker(frozenset({"job-2", "job-3"})),
            )
        )
    )

    result = await resolve_accessible_ids(
        _principal(),
        ResourceKind.JOB,
        ResourceAction.READ,
        logging.getLogger("test"),
    )

    assert result == frozenset({"job-2"})


@pytest.mark.asyncio
async def test_accessible_ids_returns_none_when_all_checkers_opt_out() -> None:
    hooks.register(
        BaseBindings(
            permission_checkers=(
                _Checker(None),
                _Checker(None),
            )
        )
    )

    result = await resolve_accessible_ids(
        _principal(),
        ResourceKind.JOB,
        ResourceAction.READ,
        logging.getLogger("test"),
    )

    assert result is None


@pytest.mark.asyncio
async def test_usage_sink_failure_does_not_block_later_sinks() -> None:
    recording = _RecordingSink()
    hooks.register(BaseBindings(usage_sinks=(_FailingSink(), recording)))

    row: UsageRow = {
        "org_id": "org-1",
        "principal_id": "p-1",
        "job_id": "job-1",
        "status": "completed",
    }
    await emit_usage([row], logging.getLogger("test"))

    assert recording.rows == [row]
