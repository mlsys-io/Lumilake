import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from support.runtime_server import (
    RecordingRuntimeManager,
    attach_request_states,
    make_workflow,
)

from lumilake_server.runtime.capacity import FreeCapacity
from lumilake_server.runtime.job_manager.base import Job
from lumilake_server.runtime.request import WorkflowSliceMeta
from lumilake_server.runtime.server import LumilakeServer


class _PoolRuntimeManager(RecordingRuntimeManager):
    """A fixed worker pool whose IDLE subset the test drives.

    FlowMesh lists only IDLE workers through ``get_workers`` and marks a worker
    BUSY while it runs a task, so the test models that by editing ``idle``
    around a running batch. ``dispatchable`` is the set FlowMesh would let take
    queued work; it follows the pool unless the test narrows it.
    """

    def __init__(self, workers: list[str], profiles: dict[str, dict[str, Any]]) -> None:
        super().__init__()
        self.workers = workers
        self.profiles = profiles
        self.idle = list(workers)
        self.dispatchable = list(workers)

    async def get_workers(self) -> list[str]:
        return list(self.idle)

    async def get_all_workers(self) -> list[str]:
        return list(self.workers)

    async def get_dispatchable_workers(self) -> list[str]:
        return list(self.dispatchable)

    async def get_worker_profile(self, worker_id: str) -> dict[str, Any]:
        return self.profiles[worker_id]


_CPU_PROFILE: dict[str, Any] = {"gpu": {"count": 0}, "cpu": {"logical_cores": 8}}


async def _enqueue_job(server: LumilakeServer, workflow: Any) -> Any:
    job = Job(
        request_id=workflow.request_id,
        runtime_graphs={workflow.graph_name: workflow.runtime_graph},
        data_profile_graphs={workflow.graph_name: workflow.data_profile_graph},
        dsl_graphs={workflow.graph_name: workflow.dsl_graph},
        workflow_slices={
            workflow.graph_name: WorkflowSliceMeta(
                public_graph_name=workflow.public_graph_name,
                slice_index=workflow.slice_index,
                slice_start=workflow.slice_start,
                slice_length=workflow.slice_length,
                total_length=workflow.total_length,
                template_hash=workflow.template_hash,
                varying_input_keys=workflow.varying_input_keys,
            )
        },
        config=workflow.config,
    )
    enqueued = await server.job_manager.enqueue(job)
    return enqueued[0]


async def _wait_until(predicate: Any, *, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached before timeout")
        await asyncio.sleep(0.01)


def _job_units(request_id: str, count: int) -> list[Any]:
    return [
        make_workflow(
            workflow_id=f"{request_id}-{index}",
            request_id=request_id,
            graph_name=f"{request_id}-g{index}",
            public_graph_name="shared",
            slice_index=index,
            slice_start=index,
            slice_length=1,
            total_length=count,
        )
        for index in range(count)
    ]


def _cpu_batch() -> Any:
    return SimpleNamespace(
        config=SimpleNamespace(hardware_requirements=None),
        workflows=[SimpleNamespace(request_id="req", id="wf")],
        runtime_graphs={},
        clustering_seconds=0.0,
    )


class _BlockingBatches:
    """Records admitted batches and blocks each until the test releases it."""

    def __init__(self, runtime_manager: _PoolRuntimeManager, count: int) -> None:
        self._runtime_manager = runtime_manager
        self.calls: list[tuple[str, list[str], list[Any]]] = []
        self.finish = [asyncio.Event() for _ in range(count)]
        self._running = 0

    async def process_batch(
        self,
        selected_batch: Any,
        batch_id: str,
        selected_workers: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        *,
        execution_request_id: str,
        member_request_ids: set[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        index = len(self.calls)
        self.calls.append(
            (batch_id, list(selected_workers), list(selected_batch.workflows))
        )
        self._running += 1
        self._runtime_manager.idle = []
        try:
            await self.finish[index].wait()
        finally:
            self._running -= 1
            if self._running == 0:
                self._runtime_manager.idle = list(self._runtime_manager.workers)
        outputs = {
            item.workflow_id: {"output": [f"value-{item.workflow_id}"]}
            for item in selected_batch.workflows
        }
        return outputs, {}

    def release_all(self) -> None:
        for event in self.finish:
            event.set()


@pytest.mark.asyncio
async def test_units_are_admitted_as_capacity_frees(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job with more units than one batch admits the next batch while an
    earlier one still runs on the same worker, and stops at the queue depth."""
    server = server_factory()
    server.config.batch_size = 1
    server.config.cpu_worker_group_size = 1
    server.config.gpu_worker_group_size = 0
    server.config.dispatch_queue_depth = 2
    runtime_manager = _PoolRuntimeManager(["cpu-0"], {"cpu-0": _CPU_PROFILE})
    server.runtime_manager = cast(Any, runtime_manager)

    workflows = _job_units("req-a", 3)
    attach_request_states(server, workflows)
    for workflow in workflows:
        await _enqueue_job(server, workflow)

    batches = _BlockingBatches(runtime_manager, count=3)
    monkeypatch.setattr(server, "_process_batch", batches.process_batch)

    scheduler_task = asyncio.create_task(server._scheduler_loop())
    try:
        await _wait_until(lambda: len(batches.calls) == 2)
        # The second batch was admitted onto the worker the first still runs on.
        assert [call[1] for call in batches.calls] == [["cpu-0"], ["cpu-0"]]
        assert server.busy_workers == {"cpu-0"}

        # The worker is at its queue depth, so no third batch is admitted.
        await asyncio.sleep(0.1)
        assert len(batches.calls) == 2

        # The first batch finishing frees depth and admits the next unit, while
        # the second batch is still running.
        batches.finish[0].set()
        await _wait_until(lambda: len(batches.calls) == 3)
        assert batches.calls[2][1] == ["cpu-0"]
    finally:
        batches.release_all()
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_capacity_limits_still_hold(server_factory) -> None:
    """A worker at its queue depth is not offered, a claim against it is
    refused, and the default depth of one never re-offers a claimed worker."""
    server = server_factory()
    server.config.cpu_worker_group_size = 1
    server.config.gpu_worker_group_size = 0
    profiles = {"cpu-0": _CPU_PROFILE}
    runtime_manager = _PoolRuntimeManager(["cpu-0"], profiles)
    server.runtime_manager = cast(Any, runtime_manager)
    batch = _cpu_batch()

    server.config.dispatch_queue_depth = 2
    runtime_manager.idle = []
    server._worker_reservations["cpu-0"] = 1
    assert (await server._snapshot_free_capacity()).cpu_worker_ids == ("cpu-0",)

    server._worker_reservations["cpu-0"] = 2
    assert (await server._snapshot_free_capacity()).is_empty()
    # A stale snapshot that still lists the worker cannot over-admit.
    stale = FreeCapacity(
        cpu_worker_ids=("cpu-0",), gpu_worker_ids=(), profiles=profiles
    )
    assert await server._try_claim_workers(batch, stale) is None
    assert server.busy_workers == {"cpu-0"}

    server.config.dispatch_queue_depth = 1
    server._worker_reservations["cpu-0"] = 1
    assert (await server._snapshot_free_capacity()).is_empty()

    server._worker_reservations.clear()
    runtime_manager.idle = ["cpu-0"]
    assert (await server._snapshot_free_capacity()).cpu_worker_ids == ("cpu-0",)


@pytest.mark.asyncio
async def test_a_batch_lands_on_the_free_worker_not_the_committed_one(
    server_factory,
) -> None:
    """A worker already carrying a batch is offered behind a free worker, so a
    new batch goes to the free one however FlowMesh orders its listing."""
    server = server_factory()
    server.config.cpu_worker_group_size = 1
    server.config.gpu_worker_group_size = 0
    server.config.dispatch_queue_depth = 2
    profiles = {"w1": _CPU_PROFILE, "w2": _CPU_PROFILE}
    runtime_manager = _PoolRuntimeManager(["w1", "w2"], profiles)
    server.runtime_manager = cast(Any, runtime_manager)
    server._worker_reservations["w1"] = 1

    free = await server._snapshot_free_capacity()
    assert free.cpu_worker_ids == ("w2", "w1")
    assert await server._try_claim_workers(_cpu_batch(), free) == ["w2"]
    assert server._worker_reservations == {"w1": 1, "w2": 1}


@pytest.mark.asyncio
async def test_a_committed_worker_flowmesh_no_longer_lists_is_not_offered(
    server_factory,
) -> None:
    """A worker that reserved capacity but left the cluster is dropped instead of
    given queued work it cannot run."""
    server = server_factory()
    server.config.cpu_worker_group_size = 1
    server.config.gpu_worker_group_size = 0
    server.config.dispatch_queue_depth = 2
    runtime_manager = _PoolRuntimeManager(["w1"], {"w1": _CPU_PROFILE})
    server.runtime_manager = cast(Any, runtime_manager)
    server._worker_reservations["w1"] = 1
    runtime_manager.idle = []
    runtime_manager.workers = []
    runtime_manager.dispatchable = []

    assert (await server._snapshot_free_capacity()).is_empty()


@pytest.mark.asyncio
async def test_a_cordoned_committed_worker_is_not_offered(server_factory) -> None:
    """A worker still registered but unable to take queued work, such as a
    cordoned one, is not re-offered while it carries a batch."""
    server = server_factory()
    server.config.cpu_worker_group_size = 1
    server.config.gpu_worker_group_size = 0
    server.config.dispatch_queue_depth = 2
    runtime_manager = _PoolRuntimeManager(["w1"], {"w1": _CPU_PROFILE})
    server.runtime_manager = cast(Any, runtime_manager)
    server._worker_reservations["w1"] = 1
    runtime_manager.idle = []
    runtime_manager.dispatchable = []

    assert (await server._snapshot_free_capacity()).is_empty()


@pytest.mark.asyncio
async def test_second_job_is_not_starved(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-unit job queued behind a longer job is admitted while the longer
    job still has work queued and its batch still runs."""
    server = server_factory()
    server.config.batch_size = 1
    server.config.cpu_worker_group_size = 1
    server.config.gpu_worker_group_size = 0
    server.config.dispatch_queue_depth = 2
    runtime_manager = _PoolRuntimeManager(["cpu-0"], {"cpu-0": _CPU_PROFILE})
    server.runtime_manager = cast(Any, runtime_manager)

    long_job = _job_units("req-long", 4)
    short_job = _job_units("req-short", 1)
    attach_request_states(server, long_job + short_job)
    for workflow in long_job + short_job:
        await _enqueue_job(server, workflow)

    batches = _BlockingBatches(runtime_manager, count=2)
    monkeypatch.setattr(server, "_process_batch", batches.process_batch)

    scheduler_task = asyncio.create_task(server._scheduler_loop())
    try:
        await _wait_until(lambda: len(batches.calls) == 2)
    finally:
        batches.release_all()
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.05)

    admitted_requests = [call[2][0].request_id for call in batches.calls[:2]]
    assert admitted_requests == ["req-long", "req-short"]
