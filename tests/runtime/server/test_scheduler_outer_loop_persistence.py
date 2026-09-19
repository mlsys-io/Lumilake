import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from lumilake import envs
from support.runtime_server import attach_request_states

from lumilake_server.runtime.capacity import FreeCapacity


def _patch_capacity(server: Any, cpu_ids: list[str], gpu_ids: list[str]) -> None:
    """Feed the scheduler a fixed idle-worker pool, minus whatever is busy."""

    async def _snapshot() -> FreeCapacity:
        busy = server._busy_workers
        return FreeCapacity(
            cpu_worker_ids=tuple(w for w in cpu_ids if w not in busy),
            gpu_worker_ids=tuple(w for w in gpu_ids if w not in busy),
        )

    server._snapshot_free_capacity = _snapshot  # type: ignore[method-assign]


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
    assert sleep_calls == [server.config.poll_interval_seconds]


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

    async def reserve_batch(self, batch_size: int, *, capacity: Any = None) -> Any:
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

    async def abort_reservation(self, reservation: Any, *, reason: Any = None) -> None:
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

    workers_used: list[list[str]] = []
    dispatched_batches: list[str] = []
    active_run_batch_tasks = 0
    max_active_run_batch_tasks = 0

    async def _no_accumulation_wait(free: Any = None) -> None:
        return

    _patch_capacity(
        server,
        ["cpu-0", "cpu-1", "cpu-2"],
        ["gpu-0", "gpu-1", "gpu-2"],
    )

    async def _blocking_run_batch(workers: list[str], batch: Any) -> None:
        nonlocal active_run_batch_tasks, max_active_run_batch_tasks
        active_run_batch_tasks += 1
        max_active_run_batch_tasks = max(
            max_active_run_batch_tasks,
            active_run_batch_tasks,
        )
        workers_used.append(list(workers))
        dispatched_batches.append(str(batch.name))
        if max_active_run_batch_tasks >= 2:
            cancel_scheduler.set()
        await release_batches.wait()
        active_run_batch_tasks -= 1

    server._maybe_wait_for_batch_accumulation = _no_accumulation_wait  # type: ignore[method-assign]
    server._run_batch = _blocking_run_batch  # type: ignore[method-assign]

    scheduler_task = asyncio.create_task(server._scheduler_loop())
    await asyncio.wait_for(cancel_scheduler.wait(), timeout=5.0)
    release_batches.set()
    await asyncio.wait_for(scheduler_task, timeout=5.0)

    assert max_active_run_batch_tasks >= 2
    assert dispatched_batches == ["batch-1", "batch-2"]
    first, second = workers_used[0], workers_used[1]
    assert first and second
    assert not set(first) & set(second)


class _CpuOnlyBatchJobManager:
    """Yields one CPU-only batch (no GPU backends) then cancels."""

    def __init__(self) -> None:
        self._select_calls = 0
        self.aborted_count = 0
        self.committed_count = 0

    async def wait_for_work(self) -> None:
        if self._select_calls >= 1:
            raise asyncio.CancelledError

    async def reserve_batch(self, batch_size: int, *, capacity: Any = None) -> Any:
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

    async def abort_reservation(self, reservation: Any, *, reason: Any = None) -> None:
        self.aborted_count += 1


@pytest.mark.asyncio
async def test_scheduler_cpu_only_batch_skips_gpu_wait(server_factory) -> None:
    """A batch with no GPU-backend ops must request gpu_group_size=0."""
    server = server_factory()
    server.config.gpu_worker_group_size = 2
    server.config.cpu_worker_group_size = 1
    server.job_manager = cast(Any, _CpuOnlyBatchJobManager())

    claimed: list[list[str]] = []

    async def _no_accumulation_wait(free: Any = None) -> None:
        return

    async def _noop_run_batch(workers: list[str], batch: Any) -> None:
        claimed.append(list(workers))

    _patch_capacity(server, ["cpu-0", "cpu-1"], ["gpu-0", "gpu-1"])
    server._maybe_wait_for_batch_accumulation = _no_accumulation_wait  # type: ignore[method-assign]
    server._run_batch = _noop_run_batch  # type: ignore[method-assign]

    await server._scheduler_loop()
    await asyncio.sleep(0)

    assert claimed == [["cpu-0"]]


@pytest.mark.asyncio
async def test_scheduler_cpu_batch_requests_a_cpu_worker_when_group_size_is_zero(
    server_factory,
) -> None:
    """LUMILAKE_CPU_WORKER_GROUP_SIZE=0 is only valid alongside a nonzero GPU
    group (envs.py), but a batch containing a CPU-only op (data_retrieval /
    api) still needs one CPU worker to be claimed — matching what
    schedule preview (_select_preview_workers_and_profiles) already
    requires for the same graph. Without this, dispatch requests zero CPU
    workers for an API-mode graph and HALO later rejects the schedule."""
    server = server_factory()
    server.config.gpu_worker_group_size = 2
    server.config.cpu_worker_group_size = 0
    server.job_manager = cast(Any, _CpuOnlyBatchJobManager())

    claimed: list[list[str]] = []

    async def _no_accumulation_wait(free: Any = None) -> None:
        return

    async def _noop_run_batch(workers: list[str], batch: Any) -> None:
        claimed.append(list(workers))

    _patch_capacity(server, ["cpu-0", "cpu-1"], ["gpu-0", "gpu-1"])
    server._maybe_wait_for_batch_accumulation = _no_accumulation_wait  # type: ignore[method-assign]
    server._run_batch = _noop_run_batch  # type: ignore[method-assign]

    await server._scheduler_loop()
    await asyncio.sleep(0)

    # cpu_worker_group_size=0 is floored to 1 for a CPU-only batch inside
    # _try_claim_workers, so the dispatch still claims a CPU worker.
    assert claimed == [["cpu-0"]]


class _GpuBatchJobManager:
    """Yields one batch containing a GPU-backend op then cancels."""

    def __init__(self) -> None:
        self._select_calls = 0
        self.aborted_count = 0
        self.committed_count = 0

    async def wait_for_work(self) -> None:
        if self._select_calls >= 1:
            raise asyncio.CancelledError

    async def reserve_batch(self, batch_size: int, *, capacity: Any = None) -> Any:
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

    async def abort_reservation(self, reservation: Any, *, reason: Any = None) -> None:
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

    claimed: list[list[str]] = []

    async def _no_accumulation_wait(free: Any = None) -> None:
        return

    async def _noop_run_batch(workers: list[str], batch: Any) -> None:
        claimed.append(list(workers))

    _patch_capacity(server, ["cpu-0", "cpu-1"], ["gpu-0", "gpu-1"])
    server._maybe_wait_for_batch_accumulation = _no_accumulation_wait  # type: ignore[method-assign]
    server._run_batch = _noop_run_batch  # type: ignore[method-assign]

    await server._scheduler_loop()
    await asyncio.sleep(0)

    assert claimed == [["gpu-0", "gpu-1", "cpu-0"]]


class _RecordingJobManager:
    """Records reserve/commit/abort calls to verify batch lifecycle."""

    def __init__(self, *, batches_to_yield: int = 1) -> None:
        self._batches_to_yield = batches_to_yield
        self._calls = 0
        self.commits: list[Any] = []
        self.aborts: list[Any] = []
        self.removed: list[Any] = []
        self.abort_reasons: list[str | None] = []

    async def wait_for_work(self) -> None:
        if self._calls >= self._batches_to_yield:
            raise asyncio.CancelledError

    async def reserve_batch(self, batch_size: int, *, capacity: Any = None) -> Any:
        self._calls += 1
        if self._calls > self._batches_to_yield:
            return None
        cpu_node = SimpleNamespace(backend="data_retrieval", task_type="data_retrieval")
        selection = SimpleNamespace(
            config=SimpleNamespace(hardware_requirements=None),
            workflows=[
                SimpleNamespace(
                    request_id=f"req-{self._calls}",
                    workflow_id=f"wf-{self._calls}",
                    public_graph_name="g",
                    slice_index=0,
                )
            ],
            runtime_graphs={"g": SimpleNamespace(nodes={"n": cpu_node})},
            name=f"batch-{self._calls}",
            clustering_seconds=0.0,
        )
        return SimpleNamespace(selection=selection, id=self._calls)

    async def commit_reservation(self, reservation: Any) -> None:
        self.commits.append(reservation.id)

    async def abort_reservation(self, reservation: Any, *, reason: Any = None) -> None:
        self.aborts.append(reservation.id)
        self.abort_reasons.append(None if reason is None else str(reason))

    async def remove_workflows(self, workflow_ids: Any) -> None:
        self.removed.append(list(workflow_ids))


@pytest.mark.asyncio
async def test_scheduler_aborts_reservation_when_workers_unavailable(
    server_factory,
) -> None:
    """A lost claim race aborts the reservation with the capacity reason.

    The snapshot reports a worker idle, but it is claimed by another dispatch
    before ``_try_claim_workers`` runs. The reservation must be released as a
    capacity abort so ``miss_count`` is not inflated by capacity pressure.
    """
    server = server_factory()
    server.config.gpu_worker_group_size = 0
    server.config.cpu_worker_group_size = 1
    job_manager = _RecordingJobManager(batches_to_yield=1)
    server.job_manager = cast(Any, job_manager)

    async def _no_accumulation_wait(free: Any = None) -> None:
        return

    async def _stale_snapshot() -> FreeCapacity:
        return FreeCapacity(cpu_worker_ids=("cpu-0",), gpu_worker_ids=())

    async def _noop_run_batch(workers: list[str], batch: Any) -> None:
        return

    server._busy_workers.add("cpu-0")
    server._snapshot_free_capacity = _stale_snapshot  # type: ignore[method-assign]
    server._maybe_wait_for_batch_accumulation = _no_accumulation_wait  # type: ignore[method-assign]
    server._run_batch = _noop_run_batch  # type: ignore[method-assign]

    await server._scheduler_loop()
    await asyncio.sleep(0)

    assert job_manager.aborts == [1]
    assert job_manager.commits == []
    assert job_manager.abort_reasons == ["capacity"]


@pytest.mark.asyncio
async def test_scheduler_fails_unplaceable_batch_loudly(server_factory) -> None:
    """When worker acquisition gives up, the batch's jobs are failed with a
    clear reason and removed from the queue so they do not re-select."""
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

    # Attach a real request state so _fail_unplaceable_batch can finalize it.
    handlers = attach_request_states(
        server,
        [
            SimpleNamespace(
                request_id="req-1",
                workflow_id="wf-1",
                public_graph_name="g",
                slice_index=0,
                runtime_graph=SimpleNamespace(node_count=1),
            )
        ],
    )

    await server._scheduler_loop()

    assert job_manager.aborts == [1]
    assert job_manager.commits == []
    assert job_manager.removed == [["wf-1"]]
    handler = handlers["req-1"]
    assert len(handler.results) == 1
    result = handler.results[0]
    assert result.error_info is not None
    assert any("placement_failed" in err for err in result.error_info)


class _GpuThenCpuJobManager:
    """Yields a GPU batch first (which cannot be placed), then a CPU batch
    (which can). If the GPU batch is not removed from the queue, it re-selects
    forever and the CPU batch behind it is never admitted — the head-of-line
    assertion."""

    def __init__(self) -> None:
        self._select_calls = 0
        self._gpu_removed = False
        self._cpu_committed = False
        self.aborted_count = 0
        self.committed_count = 0
        self.removed: list[list[str]] = []

    async def wait_for_work(self) -> None:
        if self._select_calls >= 3:
            raise asyncio.CancelledError

    async def reserve_batch(self, batch_size: int) -> Any:
        self._select_calls += 1
        # While the GPU batch is still queued, it keeps re-selecting.
        if not self._gpu_removed:
            gpu_node = SimpleNamespace(backend="vllm", task_type="inference")
            selection = SimpleNamespace(
                config=SimpleNamespace(hardware_requirements=None),
                workflows=[
                    SimpleNamespace(
                        request_id="req-gpu",
                        workflow_id="wf-gpu",
                        public_graph_name="g",
                        slice_index=0,
                    )
                ],
                runtime_graphs={"g": SimpleNamespace(nodes={"n": gpu_node})},
                name="gpu-batch",
                clustering_seconds=0.0,
            )
            return SimpleNamespace(selection=selection, id="gpu")
        if self._cpu_committed:
            return None
        cpu_node = SimpleNamespace(backend="data_retrieval", task_type="data_retrieval")
        selection = SimpleNamespace(
            config=SimpleNamespace(hardware_requirements=None),
            workflows=[
                SimpleNamespace(
                    request_id="req-cpu",
                    workflow_id="wf-cpu",
                    public_graph_name="g",
                    slice_index=0,
                )
            ],
            runtime_graphs={"g": SimpleNamespace(nodes={"n": cpu_node})},
            name="cpu-batch",
            clustering_seconds=0.0,
        )
        return SimpleNamespace(selection=selection, id="cpu")

    async def commit_reservation(self, reservation: Any) -> None:
        self.committed_count += 1
        if getattr(reservation, "id", None) == "cpu":
            self._cpu_committed = True

    async def abort_reservation(self, reservation: Any) -> None:
        self.aborted_count += 1

    async def remove_workflows(self, workflow_ids: Any) -> None:
        ids = list(workflow_ids)
        self.removed.append(ids)
        if "wf-gpu" in ids:
            self._gpu_removed = True


@pytest.mark.asyncio
async def test_scheduler_unplaceable_gpu_batch_does_not_block_cpu_batch(
    server_factory,
) -> None:
    """Head-of-line assertion: a GPU batch that cannot be placed (worker
    acquisition times out) must not prevent the following CPU-only batch from
    being admitted. The failed GPU batch is dropped from the queue, so the
    scheduler loop proceeds to admit the CPU batch."""
    server = server_factory()
    server.config.gpu_worker_group_size = 1
    server.config.cpu_worker_group_size = 1
    job_manager = _GpuThenCpuJobManager()
    server.job_manager = cast(Any, job_manager)

    cpu_dispatched = asyncio.Event()
    dispatched_batches: list[str] = []

    async def _no_accumulation_wait() -> None:
        return

    async def _worker_acquisition(
        cpu_group_size: int, gpu_group_size: int, **_kw: Any
    ) -> Any:
        # No GPU worker exists on this box: any batch that needs a GPU group
        # can never be placed and times out. Only a CPU-only batch (gpu_group
        # size 0) can be admitted.
        if gpu_group_size > 0:
            return None
        return ["cpu-0"]

    async def _noop_run_batch(workers: list[str], batch: Any) -> None:
        dispatched_batches.append(str(batch.name))
        cpu_dispatched.set()

    server._wait_for_batch_accumulation = _no_accumulation_wait  # type: ignore[method-assign]
    server._wait_for_available_worker_group = (  # type: ignore[method-assign]
        _worker_acquisition
    )
    server._run_batch = _noop_run_batch  # type: ignore[method-assign]

    await server._scheduler_loop()
    await asyncio.wait_for(cpu_dispatched.wait(), timeout=5.0)

    # The GPU batch was aborted (not committed) and dropped from the queue.
    assert job_manager.aborted_count == 1
    assert job_manager.committed_count == 1
    assert job_manager.removed == [["wf-gpu"]]
    # The CPU batch behind it was admitted and dispatched.
    assert dispatched_batches == ["cpu-batch"]


@pytest.mark.asyncio
async def test_worker_group_wait_returns_none_after_bound(
    server_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_wait_for_available_worker_group returns None once the admission wait
    bound elapses instead of looping forever."""
    monkeypatch.setattr(envs, "LUMILAKE_WORKER_GROUP_WAIT_SECONDS", 0.0)
    server = server_factory()

    async def _no_workers() -> list[Any]:
        return []

    async def _no_profile(worker: str) -> Any:
        raise RuntimeError("no profile")

    server.runtime_manager.get_workers = _no_workers  # type: ignore[method-assign]
    server.runtime_manager.get_worker_profile = _no_profile  # type: ignore[method-assign]

    result = await server._wait_for_available_worker_group(
        cpu_group_size=1, gpu_group_size=0
    )

    assert result is None


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

    async def _no_accumulation_wait(free: Any = None) -> None:
        return

    async def _noop_run_batch(workers: list[str], batch: Any) -> None:
        return

    _patch_capacity(server, ["cpu-0", "cpu-1"], ["gpu-0", "gpu-1"])
    server._maybe_wait_for_batch_accumulation = _no_accumulation_wait  # type: ignore[method-assign]
    server._run_batch = _noop_run_batch  # type: ignore[method-assign]

    await server._scheduler_loop()
    await asyncio.sleep(0)

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

    async def reserve_batch(self, batch_size: int, *, capacity: Any = None) -> Any:
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

    async def abort_reservation(self, reservation: Any, *, reason: Any = None) -> None:
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

    claimed: list[list[str]] = []

    async def _no_accumulation_wait(free: Any = None) -> None:
        return

    async def _noop_run_batch(workers: list[str], batch: Any) -> None:
        claimed.append(list(workers))

    _patch_capacity(server, ["cpu-0", "cpu-1"], ["gpu-0", "gpu-1"])
    server._maybe_wait_for_batch_accumulation = _no_accumulation_wait  # type: ignore[method-assign]
    server._run_batch = _noop_run_batch  # type: ignore[method-assign]

    await server._scheduler_loop()
    await asyncio.sleep(0)

    assert claimed == [["gpu-0", "gpu-1", "cpu-0"]]


class _AlwaysReserveJobManager:
    """Yields a reservation on every call; used to exercise capacity waits."""

    def __init__(self) -> None:
        self.reserve_calls = 0

    async def wait_for_work(self) -> None:
        return

    async def reserve_batch(self, batch_size: int, *, capacity: Any = None) -> Any:
        self.reserve_calls += 1
        selection = SimpleNamespace(
            config=SimpleNamespace(hardware_requirements=None),
            workflows=[SimpleNamespace(request_id="req", id="wf")],
            runtime_graphs={},
            clustering_seconds=0.0,
            name="batch",
        )
        return SimpleNamespace(selection=selection)

    async def commit_reservation(self, reservation: Any) -> None:
        return

    async def abort_reservation(self, reservation: Any, *, reason: Any = None) -> None:
        return


@pytest.mark.asyncio
async def test_scheduler_does_not_spin_when_no_eligible_capacity(
    server_factory,
) -> None:
    """Queue non-empty + no eligible capacity must block on the capacity wait,
    not spin through snapshot -> reserve -> repeat."""
    server = server_factory()
    server.config.gpu_worker_group_size = 0
    server.config.cpu_worker_group_size = 1
    server.job_manager = cast(Any, _AlwaysReserveJobManager())

    async def _no_accumulation_wait(free: Any = None) -> None:
        return

    async def _noop_run_batch(workers: list[str], batch: Any) -> None:
        return

    async def _always_fail_claim(batch: Any, free: Any) -> None:
        return None

    # Capacity is non-empty but the claim always fails (e.g. lost a race).
    _patch_capacity(server, ["cpu-0"], [])
    server._maybe_wait_for_batch_accumulation = _no_accumulation_wait  # type: ignore[method-assign]
    server._run_batch = _noop_run_batch  # type: ignore[method-assign]
    server._try_claim_workers = _always_fail_claim  # type: ignore[method-assign]

    wait_calls = 0
    release = asyncio.Event()

    async def _blocking_wait_capacity() -> None:
        nonlocal wait_calls
        wait_calls += 1
        await release.wait()

    server._wait_capacity = _blocking_wait_capacity  # type: ignore[method-assign]

    scheduler_task = asyncio.create_task(server._scheduler_loop())
    # Let the loop reach the capacity wait. It must block there rather than
    # spinning through reserve_batch repeatedly.
    for _ in range(100):
        if wait_calls >= 1:
            break
        await asyncio.sleep(0)
    assert wait_calls == 1
    # While blocked on the capacity wait, no further reservations are made.
    reserve_calls_at_wait = server.job_manager.reserve_calls
    await asyncio.sleep(0.01)
    assert server.job_manager.reserve_calls == reserve_calls_at_wait
    # The loop is genuinely suspended in the capacity wait (release is never
    # set), so cancellation must be deliverable.
    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass


class _CapacityRecordingJobManager:
    """Records the capacity argument passed to each reserve_batch call."""

    def __init__(self) -> None:
        self.capacities: list[Any] = []
        self._reserve_calls = 0

    async def wait_for_work(self) -> None:
        if self._reserve_calls >= 1:
            raise asyncio.CancelledError

    async def reserve_batch(self, batch_size: int, *, capacity: Any = None) -> Any:
        self._reserve_calls += 1
        self.capacities.append(capacity)
        selection = SimpleNamespace(
            config=SimpleNamespace(hardware_requirements=None),
            workflows=[SimpleNamespace(request_id="req", id="wf")],
            runtime_graphs={},
            clustering_seconds=0.0,
            name="batch",
        )
        return SimpleNamespace(selection=selection)

    async def commit_reservation(self, reservation: Any) -> None:
        return

    async def abort_reservation(self, reservation: Any, *, reason: Any = None) -> None:
        return


async def _run_capacity_recording_loop(
    server: Any, job_manager: _CapacityRecordingJobManager
) -> None:
    server.job_manager = cast(Any, job_manager)

    async def _no_accumulation_wait(free: Any = None) -> None:
        return

    async def _noop_run_batch(workers: list[str], batch: Any) -> None:
        return

    _patch_capacity(server, ["cpu-0"], [])
    server._maybe_wait_for_batch_accumulation = _no_accumulation_wait  # type: ignore[method-assign]
    server._run_batch = _noop_run_batch  # type: ignore[method-assign]

    await server._scheduler_loop()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_capacity_aware_selection_flag_controls_capacity_passed(
    server_factory,
) -> None:
    """LUMILAKE_CAPACITY_AWARE_SELECTION gates whether reserve_batch sees
    capacity. On, selection is capacity-filtered; off, it is capacity-blind."""
    server = server_factory()
    server.config.gpu_worker_group_size = 0
    server.config.cpu_worker_group_size = 1

    on_manager = _CapacityRecordingJobManager()
    server.config.capacity_aware_selection = True
    await _run_capacity_recording_loop(server, on_manager)
    assert len(on_manager.capacities) == 1
    assert on_manager.capacities[0] is not None

    # The noop run never releases the claimed worker; reset busy state so the
    # second loop sees idle capacity again.
    server._busy_workers.clear()

    off_manager = _CapacityRecordingJobManager()
    server.config.capacity_aware_selection = False
    await _run_capacity_recording_loop(server, off_manager)
    assert len(off_manager.capacities) == 1
    assert off_manager.capacities[0] is None


@pytest.mark.asyncio
async def test_release_workers_wakes_capacity_waiter(server_factory) -> None:
    """The production release path frees workers and wakes a capacity waiter."""
    server = server_factory()
    server._busy_workers.add("cpu-0")

    waiter = asyncio.create_task(server._capacity_changed.wait())
    await server._release_workers(["cpu-0"])

    assert "cpu-0" not in server._busy_workers
    await asyncio.wait_for(waiter, timeout=5.0)


@pytest.mark.asyncio
async def test_try_claim_workers_rejects_undersized_worker(server_factory) -> None:
    """Hardware filtering rejects a worker that cannot meet the batch's
    hardware requirements."""
    server = server_factory()
    server.config.gpu_worker_group_size = 0
    server.config.cpu_worker_group_size = 1

    batch = SimpleNamespace(
        config=SimpleNamespace(
            hardware_requirements=SimpleNamespace(
                cpu=8, memory=None, gpu=None, gpu_memory=None
            )
        ),
        workflows=[SimpleNamespace(request_id="req", id="wf")],
        runtime_graphs={},
        clustering_seconds=0.0,
    )
    free = FreeCapacity(
        cpu_worker_ids=("cpu-0",),
        gpu_worker_ids=(),
        profiles={"cpu-0": {"cpu": {"logical_cores": 4}}},
    )
    # 4 cores < required 8, so the claim must fail.
    assert await server._try_claim_workers(batch, free) is None
    assert server._busy_workers == set()


class _CommitRaisesJobManager:
    """Yields one reservation whose commit raises."""

    def __init__(self) -> None:
        self._select_calls = 0
        self.aborted_count = 0

    async def wait_for_work(self) -> None:
        if self._select_calls >= 1:
            raise asyncio.CancelledError

    async def reserve_batch(self, batch_size: int, *, capacity: Any = None) -> Any:
        self._select_calls += 1
        if self._select_calls == 1:
            selection = SimpleNamespace(
                config=SimpleNamespace(hardware_requirements=None),
                workflows=[SimpleNamespace(request_id="req", id="wf")],
                runtime_graphs={},
                clustering_seconds=0.0,
                name="batch",
            )
            return SimpleNamespace(selection=selection)
        return None

    async def commit_reservation(self, reservation: Any) -> None:
        raise RuntimeError("commit failed")

    async def abort_reservation(self, reservation: Any, *, reason: Any = None) -> None:
        self.aborted_count += 1


@pytest.mark.asyncio
async def test_scheduler_releases_workers_when_commit_raises(server_factory) -> None:
    """If commit_reservation raises after workers are claimed, the claimed
    workers must be released so they do not stay busy forever."""
    server = server_factory()
    server.config.gpu_worker_group_size = 0
    server.config.cpu_worker_group_size = 1
    job_manager = _CommitRaisesJobManager()
    server.job_manager = cast(Any, job_manager)

    async def _no_accumulation_wait(free: Any = None) -> None:
        return

    async def _noop_run_batch(workers: list[str], batch: Any) -> None:
        return

    _patch_capacity(server, ["cpu-0", "cpu-1"], [])
    server._maybe_wait_for_batch_accumulation = _no_accumulation_wait  # type: ignore[method-assign]
    server._run_batch = _noop_run_batch  # type: ignore[method-assign]

    await server._scheduler_loop()
    await asyncio.sleep(0)

    # The commit raised, so the batch was never dispatched; the claimed worker
    # must have been released and the reservation aborted.
    assert server._busy_workers == set()
    assert job_manager.aborted_count == 1
