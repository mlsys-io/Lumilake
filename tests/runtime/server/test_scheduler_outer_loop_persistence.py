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
        self.aborted_count = 0
        self.committed_count = 0

    async def wait_for_work(self) -> None:
        if self._select_calls >= 2:
            await self._cancel_ready.wait()
            raise asyncio.CancelledError

    async def reserve_batch(self, batch_size: int) -> Any:
        self._select_calls += 1
        if self._select_calls == 1:
            selection = SimpleNamespace(
                config=SimpleNamespace(hardware_requirements=None),
                workflows=[SimpleNamespace(request_id="req-1", id="wf-1")],
                runtime_graphs={},
                clustering_seconds=0.0,
                name="batch-1",
            )
            return SimpleNamespace(selection=selection)
        if self._select_calls == 2:
            selection = SimpleNamespace(
                config=SimpleNamespace(hardware_requirements=None),
                workflows=[SimpleNamespace(request_id="req-2", id="wf-2")],
                runtime_graphs={},
                clustering_seconds=0.0,
                name="batch-2",
            )
            return SimpleNamespace(selection=selection)
        return None

    async def commit_reservation(self, reservation: Any) -> None:
        self.committed_count += 1

    async def abort_reservation(self, reservation: Any) -> None:
        self.aborted_count += 1


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
        **_kw: Any,
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


class _CpuOnlyBatchJobManager:
    """Yields one CPU-only batch (no GPU backends) then cancels."""

    def __init__(self) -> None:
        self._select_calls = 0
        self.aborted_count = 0
        self.committed_count = 0

    async def wait_for_work(self) -> None:
        if self._select_calls >= 1:
            raise asyncio.CancelledError

    async def reserve_batch(self, batch_size: int) -> Any:
        self._select_calls += 1
        if self._select_calls == 1:
            cpu_node = SimpleNamespace(
                backend="data_retrieval", task_type="data_retrieval"
            )
            selection = SimpleNamespace(
                config=SimpleNamespace(hardware_requirements=None),
                workflows=[SimpleNamespace(request_id="req-cpu", id="wf-cpu")],
                runtime_graphs={
                    "g": SimpleNamespace(nodes={"n": cpu_node}),
                },
                name="cpu-batch",
                clustering_seconds=0.0,
            )
            return SimpleNamespace(selection=selection)
        return None

    async def commit_reservation(self, reservation: Any) -> None:
        self.committed_count += 1

    async def abort_reservation(self, reservation: Any) -> None:
        self.aborted_count += 1


@pytest.mark.asyncio
async def test_scheduler_cpu_only_batch_skips_gpu_wait(server_factory) -> None:
    """A batch with no GPU-backend ops must request gpu_group_size=0."""
    server = server_factory()
    server.config.gpu_worker_group_size = 2
    server.config.cpu_worker_group_size = 1
    server.job_manager = cast(Any, _CpuOnlyBatchJobManager())

    captured_gpu_sizes: list[int] = []

    async def _no_accumulation_wait() -> None:
        return

    async def _record_worker_group(
        cpu_group_size: int, gpu_group_size: int, **_kw: Any
    ) -> list[str]:
        captured_gpu_sizes.append(gpu_group_size)
        return ["cpu-0"]

    async def _noop_run_batch(workers: list[str], batch: Any) -> None:
        return

    server._wait_for_batch_accumulation = _no_accumulation_wait  # type: ignore[method-assign]
    server._wait_for_available_worker_group = _record_worker_group  # type: ignore[method-assign]
    server._run_batch = _noop_run_batch  # type: ignore[method-assign]

    await server._scheduler_loop()

    assert captured_gpu_sizes == [0]


class _GpuBatchJobManager:
    """Yields one batch containing a GPU-backend op then cancels."""

    def __init__(self) -> None:
        self._select_calls = 0
        self.aborted_count = 0
        self.committed_count = 0

    async def wait_for_work(self) -> None:
        if self._select_calls >= 1:
            raise asyncio.CancelledError

    async def reserve_batch(self, batch_size: int) -> Any:
        self._select_calls += 1
        if self._select_calls == 1:
            gpu_node = SimpleNamespace(backend="vllm", task_type="inference")
            selection = SimpleNamespace(
                config=SimpleNamespace(hardware_requirements=None),
                workflows=[SimpleNamespace(request_id="req-gpu", id="wf-gpu")],
                runtime_graphs={
                    "g": SimpleNamespace(nodes={"n": gpu_node}),
                },
                name="gpu-batch",
                clustering_seconds=0.0,
            )
            return SimpleNamespace(selection=selection)
        return None

    async def commit_reservation(self, reservation: Any) -> None:
        self.committed_count += 1

    async def abort_reservation(self, reservation: Any) -> None:
        self.aborted_count += 1


@pytest.mark.asyncio
async def test_scheduler_gpu_batch_requests_configured_gpu_group(
    server_factory,
) -> None:
    """A batch with a GPU-backend op must request the configured GPU group size."""
    server = server_factory()
    server.config.gpu_worker_group_size = 2
    server.config.cpu_worker_group_size = 1
    server.job_manager = cast(Any, _GpuBatchJobManager())

    captured_gpu_sizes: list[int] = []

    async def _no_accumulation_wait() -> None:
        return

    async def _record_worker_group(
        cpu_group_size: int, gpu_group_size: int, **_kw: Any
    ) -> list[str]:
        captured_gpu_sizes.append(gpu_group_size)
        return ["gpu-0", "gpu-1", "cpu-0"]

    async def _noop_run_batch(workers: list[str], batch: Any) -> None:
        return

    server._wait_for_batch_accumulation = _no_accumulation_wait  # type: ignore[method-assign]
    server._wait_for_available_worker_group = _record_worker_group  # type: ignore[method-assign]
    server._run_batch = _noop_run_batch  # type: ignore[method-assign]

    await server._scheduler_loop()

    assert captured_gpu_sizes == [2]


class _RecordingJobManager:
    """Records reserve/commit/abort calls to verify batch lifecycle."""

    def __init__(self, *, batches_to_yield: int = 1) -> None:
        self._batches_to_yield = batches_to_yield
        self._calls = 0
        self.commits: list[Any] = []
        self.aborts: list[Any] = []

    async def wait_for_work(self) -> None:
        if self._calls >= self._batches_to_yield:
            raise asyncio.CancelledError

    async def reserve_batch(self, batch_size: int) -> Any:
        self._calls += 1
        if self._calls > self._batches_to_yield:
            return None
        cpu_node = SimpleNamespace(backend="data_retrieval", task_type="data_retrieval")
        selection = SimpleNamespace(
            config=SimpleNamespace(hardware_requirements=None),
            workflows=[
                SimpleNamespace(request_id=f"req-{self._calls}", id=f"wf-{self._calls}")
            ],
            runtime_graphs={"g": SimpleNamespace(nodes={"n": cpu_node})},
            name=f"batch-{self._calls}",
            clustering_seconds=0.0,
        )
        return SimpleNamespace(selection=selection, id=self._calls)

    async def commit_reservation(self, reservation: Any) -> None:
        self.commits.append(reservation.id)

    async def abort_reservation(self, reservation: Any) -> None:
        self.aborts.append(reservation.id)


@pytest.mark.asyncio
async def test_scheduler_aborts_reservation_when_workers_unavailable(
    server_factory,
) -> None:
    """If worker acquisition times out, the reservation is aborted (not lost)."""
    server = server_factory()
    server.config.gpu_worker_group_size = 0
    server.config.cpu_worker_group_size = 1
    job_manager = _RecordingJobManager(batches_to_yield=1)
    server.job_manager = cast(Any, job_manager)

    async def _no_accumulation_wait() -> None:
        return

    async def _worker_acquisition_times_out(
        cpu_group_size: int, gpu_group_size: int, **_kw: Any
    ) -> Any:
        return None

    async def _noop_run_batch(workers: list[str], batch: Any) -> None:
        return

    server._wait_for_batch_accumulation = _no_accumulation_wait  # type: ignore[method-assign]
    server._wait_for_available_worker_group = (  # type: ignore[method-assign]
        _worker_acquisition_times_out
    )
    server._run_batch = _noop_run_batch  # type: ignore[method-assign]

    await server._scheduler_loop()

    assert job_manager.aborts == [1]
    assert job_manager.commits == []


@pytest.mark.asyncio
async def test_scheduler_commits_reservation_when_workers_acquired(
    server_factory,
) -> None:
    """Happy path: workers acquired → commit, no abort."""
    server = server_factory()
    server.config.gpu_worker_group_size = 0
    server.config.cpu_worker_group_size = 1
    job_manager = _RecordingJobManager(batches_to_yield=1)
    server.job_manager = cast(Any, job_manager)

    async def _no_accumulation_wait() -> None:
        return

    async def _grant_workers(
        cpu_group_size: int, gpu_group_size: int, **_kw: Any
    ) -> Any:
        return ["cpu-0"]

    async def _noop_run_batch(workers: list[str], batch: Any) -> None:
        return

    server._wait_for_batch_accumulation = _no_accumulation_wait  # type: ignore[method-assign]
    server._wait_for_available_worker_group = _grant_workers  # type: ignore[method-assign]
    server._run_batch = _noop_run_batch  # type: ignore[method-assign]

    await server._scheduler_loop()

    assert job_manager.commits == [1]
    assert job_manager.aborts == []


class _InferenceWithoutBackendBatchJobManager:
    """Yields a batch whose op has task_type=inference but an unrecognized
    backend. Exercises that the scheduler's GPU peek classifies it as GPU on
    task_type, not just backend."""

    def __init__(self) -> None:
        self._select_calls = 0
        self.aborted_count = 0
        self.committed_count = 0

    async def wait_for_work(self) -> None:
        if self._select_calls >= 1:
            raise asyncio.CancelledError

    async def reserve_batch(self, batch_size: int) -> Any:
        self._select_calls += 1
        if self._select_calls == 1:
            mystery_op = SimpleNamespace(backend="", task_type="inference")
            selection = SimpleNamespace(
                config=SimpleNamespace(hardware_requirements=None),
                workflows=[SimpleNamespace(request_id="req-mystery", id="wf-mystery")],
                runtime_graphs={"g": SimpleNamespace(nodes={"n": mystery_op})},
                name="mystery-batch",
                clustering_seconds=0.0,
            )
            return SimpleNamespace(selection=selection)
        return None

    async def commit_reservation(self, reservation: Any) -> None:
        self.committed_count += 1

    async def abort_reservation(self, reservation: Any) -> None:
        self.aborted_count += 1


@pytest.mark.asyncio
async def test_scheduler_inference_task_type_requests_gpu_even_without_backend(
    server_factory,
) -> None:
    """task_type=inference must be classified GPU even if backend isn't one of
    the canonical names — keeps the scheduler peek in sync with FlowMesh's
    dispatcher (`_runtime_op_requires_gpu`), which also looks at task_type."""
    server = server_factory()
    server.config.gpu_worker_group_size = 2
    server.config.cpu_worker_group_size = 1
    server.job_manager = cast(Any, _InferenceWithoutBackendBatchJobManager())

    captured_gpu_sizes: list[int] = []

    async def _no_accumulation_wait() -> None:
        return

    async def _record_worker_group(
        cpu_group_size: int, gpu_group_size: int, **_kw: Any
    ) -> list[str]:
        captured_gpu_sizes.append(gpu_group_size)
        return ["gpu-0", "gpu-1", "cpu-0"]

    async def _noop_run_batch(workers: list[str], batch: Any) -> None:
        return

    server._wait_for_batch_accumulation = _no_accumulation_wait  # type: ignore[method-assign]
    server._wait_for_available_worker_group = _record_worker_group  # type: ignore[method-assign]
    server._run_batch = _noop_run_batch  # type: ignore[method-assign]

    await server._scheduler_loop()

    assert captured_gpu_sizes == [2]
