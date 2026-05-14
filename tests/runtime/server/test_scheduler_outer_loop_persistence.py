import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from lumilake import envs


class _FailThenCancelJobManager:
    def __init__(self) -> None:
        self.wait_calls = 0

    async def wait_for_work(self) -> None:
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise RuntimeError("temporary scheduler failure")
        raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_scheduler_loop_continues_after_cycle_exception(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = server_factory()
    fake_job_manager = _FailThenCancelJobManager()
    server.job_manager = cast(Any, fake_job_manager)
    sleep_calls: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr("lumilake_server.runtime.server.asyncio.sleep", _fake_sleep)

    await server._scheduler_loop()

    assert fake_job_manager.wait_calls == 2
    assert sleep_calls == [envs.LUMILAKE_POLL_INTERVAL_SECONDS]


class _TwoBatchThenCancelJobManager:
    def __init__(self, *, cancel_ready: asyncio.Event) -> None:
        self._cancel_ready = cancel_ready
        self._select_calls = 0

    async def wait_for_work(self) -> None:
        if self._select_calls >= 2:
            await self._cancel_ready.wait()
            raise asyncio.CancelledError

    async def select_batch(self, batch_size: int) -> Any:
        self._select_calls += 1
        if self._select_calls == 1:
            return SimpleNamespace(workflows=[{"id": "wf-1"}], name="batch-1")
        if self._select_calls == 2:
            return SimpleNamespace(workflows=[{"id": "wf-2"}], name="batch-2")
        return None


@pytest.mark.asyncio
async def test_scheduler_loop_can_dispatch_multiple_batches_concurrently(
    server_factory,
) -> None:
    server = server_factory()
    release_batches = asyncio.Event()
    cancel_scheduler = asyncio.Event()
    fake_job_manager = _TwoBatchThenCancelJobManager(cancel_ready=cancel_scheduler)
    server.job_manager = cast(Any, fake_job_manager)

    worker_calls = 0
    workers_used: list[list[str]] = []
    dispatched_batches: list[str] = []
    active_run_batch_tasks = 0
    max_active_run_batch_tasks = 0

    async def _no_accumulation_wait() -> None:
        return

    async def _fake_wait_for_available_worker_group(
        cpu_group_size: int,
        gpu_group_size: int,
    ) -> list[str]:
        nonlocal worker_calls
        worker_calls += 1
        if worker_calls == 1:
            workers = ["gpu-0", "cpu-0"]
        elif worker_calls == 2:
            workers = ["gpu-1", "cpu-1"]
        else:
            workers = ["gpu-2", "cpu-2"]
        workers_used.append(workers)
        return workers

    async def _blocking_run_batch(workers: list[str], batch: Any) -> None:
        nonlocal active_run_batch_tasks, max_active_run_batch_tasks
        active_run_batch_tasks += 1
        max_active_run_batch_tasks = max(
            max_active_run_batch_tasks,
            active_run_batch_tasks,
        )
        dispatched_batches.append(str(batch.name))
        if max_active_run_batch_tasks >= 2:
            cancel_scheduler.set()
        await release_batches.wait()
        active_run_batch_tasks -= 1

    server._wait_for_batch_accumulation = _no_accumulation_wait  # type: ignore[method-assign]
    server._wait_for_available_worker_group = (  # type: ignore[method-assign]
        _fake_wait_for_available_worker_group
    )
    server._run_batch = _blocking_run_batch  # type: ignore[method-assign]

    scheduler_task = asyncio.create_task(server._scheduler_loop())
    await asyncio.wait_for(cancel_scheduler.wait(), timeout=5.0)
    release_batches.set()
    await asyncio.wait_for(scheduler_task, timeout=5.0)

    assert max_active_run_batch_tasks >= 2
    assert dispatched_batches == ["batch-1", "batch-2"]
    assert workers_used[:2] == [["gpu-0", "cpu-0"], ["gpu-1", "cpu-1"]]
