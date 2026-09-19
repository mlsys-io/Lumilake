"""The base job manager is an abstract contract: every method must raise
``NotImplementedError`` rather than silently returning ``None``. A silent
``None`` return would let a partially-implemented manager appear to work
while dropping work — the same inert-feature pattern this branch keeps
catching."""

from typing import cast

import pytest

from lumilake_server.runtime.job_manager.base import BaseJobManager


def _bare() -> BaseJobManager:
    """A stand-in ``self`` for invoking the un-overridden base bodies. The
    base methods never touch ``self``, so any object works."""
    return cast(BaseJobManager, object())


@pytest.mark.asyncio
async def test_base_enqueue_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        await BaseJobManager.enqueue(_bare(), None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_base_has_work_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        await BaseJobManager.has_work(_bare())


@pytest.mark.asyncio
async def test_base_wait_for_work_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        await BaseJobManager.wait_for_work(_bare())


@pytest.mark.asyncio
async def test_base_get_pending_stats_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        await BaseJobManager.get_pending_stats(_bare())


@pytest.mark.asyncio
async def test_base_reserve_batch_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        await BaseJobManager.reserve_batch(_bare(), 1)


@pytest.mark.asyncio
async def test_base_commit_reservation_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        await BaseJobManager.commit_reservation(_bare(), None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_base_abort_reservation_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        await BaseJobManager.abort_reservation(_bare(), None)  # type: ignore[arg-type]


def test_base_finalize_workflows_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        BaseJobManager.finalize_workflows(_bare(), [])


def test_base_get_workflow_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        BaseJobManager.get_workflow(_bare(), "wf")
