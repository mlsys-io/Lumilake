"""Discrete-event simulation runner driving the real PriorityJobManager.

The runner advances a virtual clock, enqueues arrivals as they fall due, calls
the real ``reserve_batch`` / ``commit_reservation`` / ``abort_reservation``
selection path, claims workers from the pool, schedules completions, and
releases workers. It records an event log sufficient for :mod:`metrics`.

The worker model is binary busy/free: each dispatched item claims one worker of
its op kind and runs for its own service time. A batch is only committed when
every selected item can be placed on a free worker; otherwise the reservation
is aborted and the scheduler is re-polled after the next completion. This
preserves the real two-phase reserve/commit contract and naturally reproduces
head-of-line blocking when the scheduler keeps picking a partition whose
workers are busy.
"""

from dataclasses import dataclass

from lumilake_server.runtime.capacity import FreeCapacity
from lumilake_server.runtime.job_manager.priority_queue import PriorityJobManager
from tests.support.schedsim.clock import VirtualClock
from tests.support.schedsim.metrics import SimulationResult
from tests.support.schedsim.pool import WorkerPool
from tests.support.schedsim.workload import (
    ChainSpec,
    OpKind,
    Workload,
    chain_to_job,
)


@dataclass(slots=True)
class SimulationConfig:
    """Cluster and dispatch parameters for a simulation run."""

    batch_size: int = 1
    cpu_groups: int = 2
    gpu_groups: int = 1
    cpu_group_size: int = 1
    gpu_group_size: int = 1
    cpu_units_per_worker: int = 4
    gpu_units_per_worker: int = 1
    horizon: float | None = None


@dataclass(slots=True)
class _ChainState:
    chain: ChainSpec
    next_round: int = 0
    enqueued_at: float | None = None
    finished_at: float | None = None


@dataclass(slots=True)
class _ItemMeta:
    chain_id: str
    round_index: int
    service: float
    kind: OpKind


@dataclass(slots=True)
class _InFlight:
    chain_id: str
    kind: OpKind
    finish_time: float
    group_id: int | None = None


class SimulationRunner:
    """Drives one simulation run against a real :class:`PriorityJobManager`."""

    def __init__(
        self,
        workload: Workload,
        manager: PriorityJobManager,
        config: SimulationConfig | None = None,
        clock: VirtualClock | None = None,
    ) -> None:
        self._workload = workload
        self._manager = manager
        self._config = config or SimulationConfig()
        self._clock = clock or VirtualClock()
        self._pool = WorkerPool(
            cpu_groups=self._config.cpu_groups,
            gpu_groups=self._config.gpu_groups,
            cpu_group_size=self._config.cpu_group_size,
            gpu_group_size=self._config.gpu_group_size,
        )
        self._chain_state: dict[str, _ChainState] = {}
        self._item_meta: dict[str, _ItemMeta] = {}
        self._queued_counts: dict[OpKind, int] = {
            OpKind.GPU: 0,
            OpKind.DB: 0,
            OpKind.CPU: 0,
        }
        self._in_flight: dict[str, _InFlight] = {}
        self._pending: list[tuple[float, str]] = []
        self._worker_busy_seconds = 0.0
        self._bubble_seconds = 0.0

    # -- public API ---------------------------------------------------------

    async def run(self) -> SimulationResult:
        """Run the simulation to drain (or to the horizon) and return metrics."""
        self._seed_pending()
        while True:
            await self._enqueue_due_arrivals()
            made_progress = True
            while made_progress:
                made_progress = await self._dispatch_once()

            next_event = self._next_event_time()
            if next_event is None:
                break
            dt = next_event - self._clock.now()
            if dt > 0.0:
                self._accumulate_bubble(dt)
                self._clock.advance(dt)
            await self._process_completions(next_event)
            if (
                self._config.horizon is not None
                and self._clock.now() >= self._config.horizon
            ):
                break

        return self._build_result()

    # -- internals ----------------------------------------------------------

    def _seed_pending(self) -> None:
        for chain, arrival in zip(self._workload.chains, self._workload.arrival_times):
            self._chain_state[chain.chain_id] = _ChainState(chain=chain)
            self._pending.append((arrival, chain.chain_id))
        self._pending.sort(key=lambda pair: pair[0])

    async def _enqueue_due_arrivals(self) -> None:
        while self._pending and self._pending[0][0] <= self._clock.now():
            _, chain_id = self._pending.pop(0)
            await self._enqueue_round(chain_id, 0)

    async def _enqueue_round(self, chain_id: str, round_index: int) -> None:
        state = self._chain_state[chain_id]
        chain = state.chain
        if state.enqueued_at is None:
            state.enqueued_at = self._clock.now()
        kind = chain.round_kinds[round_index]
        service = chain.round_services[round_index]
        job = chain_to_job(
            chain,
            round_index=round_index,
            request_id=f"{chain.chain_id}-r{round_index}",
            priority=self._workload.params.priority,
        )
        items = await self._manager.enqueue(job)
        for item in items:
            self._item_meta[item.workflow_id] = _ItemMeta(
                chain_id=chain_id,
                round_index=round_index,
                service=service,
                kind=kind,
            )
            self._queued_counts[kind] += 1

    async def _dispatch_once(self) -> bool:
        # Build a FreeCapacity snapshot from the simulated pool so the real
        # selection path is capacity-aware (the same way the server does).
        free = FreeCapacity(
            cpu_worker_ids=tuple(str(i) for i in range(self._pool.free_cpu_groups())),
            gpu_worker_ids=tuple(str(i) for i in range(self._pool.free_gpu_groups())),
        )
        reservation = await self._manager.reserve_batch(
            self._config.batch_size, capacity=free
        )
        if reservation is None:
            return False
        items = reservation.selection.workflows
        # Claim a group per item; if any claim fails, roll back the claims made
        # so far and abort the reservation cleanly.
        claimed: list[tuple[str, OpKind, int]] = []
        for item in items:
            meta = self._item_meta[item.workflow_id]
            if meta.kind is OpKind.GPU:
                group_id = self._pool.claim_gpu_group()
            else:
                group_id = self._pool.claim_cpu_group()
            if group_id is None:
                for workflow_id, kind, gid in claimed:
                    if kind is OpKind.GPU:
                        self._pool.release_gpu_group(gid)
                    else:
                        self._pool.release_cpu_group(gid)
                await self._manager.abort_reservation(reservation)
                return False
            claimed.append((item.workflow_id, meta.kind, group_id))
        await self._manager.commit_reservation(reservation)
        for workflow_id, kind, group_id in claimed:
            meta = self._item_meta[workflow_id]
            self._queued_counts[kind] -= 1
            finish = self._clock.now() + meta.service
            self._in_flight[workflow_id] = _InFlight(
                chain_id=meta.chain_id,
                kind=kind,
                finish_time=finish,
                group_id=group_id,
            )
            # A claimed group occupies ``*_group_size`` workers, so the busy
            # accounting must scale by the group size to match the denominator
            # in ``total_worker_seconds``.
            group_size = (
                self._config.gpu_group_size
                if meta.kind is OpKind.GPU
                else self._config.cpu_group_size
            )
            self._worker_busy_seconds += meta.service * group_size
        return True

    def _next_event_time(self) -> float | None:
        candidates: list[float] = []
        if self._in_flight:
            candidates.append(min(f.finish_time for f in self._in_flight.values()))
        if self._pending:
            candidates.append(self._pending[0][0])
        if self._config.horizon is not None:
            candidates.append(self._config.horizon)
        if not candidates:
            return None
        return min(candidates)

    def _accumulate_bubble(self, dt: float) -> None:
        """Accumulate idle-worker-seconds where an idle worker could have served
        a queued item of its kind."""
        if self._queued_counts[OpKind.GPU] > 0:
            self._bubble_seconds += (
                self._pool.free_gpu_groups() * self._config.gpu_group_size * dt
            )
        if self._queued_counts[OpKind.CPU] > 0 or self._queued_counts[OpKind.DB] > 0:
            self._bubble_seconds += (
                self._pool.free_cpu_groups() * self._config.cpu_group_size * dt
            )

    async def _process_completions(self, event_time: float) -> None:
        finished: list[str] = []
        for workflow_id, in_flight in list(self._in_flight.items()):
            if in_flight.finish_time > event_time:
                continue
            finished.append(workflow_id)
            if in_flight.kind is OpKind.GPU:
                assert in_flight.group_id is not None
                self._pool.release_gpu_group(in_flight.group_id)
            else:
                assert in_flight.group_id is not None
                self._pool.release_cpu_group(in_flight.group_id)
            del self._in_flight[workflow_id]
            state = self._chain_state[in_flight.chain_id]
            state.next_round += 1
            if state.next_round >= state.chain.rounds:
                state.finished_at = event_time
            else:
                await self._enqueue_round(in_flight.chain_id, state.next_round)
        if finished:
            self._manager.finalize_workflows(finished)

    def _build_result(self) -> SimulationResult:
        result = SimulationResult()
        cpu_capacity = (
            self._config.cpu_groups
            * self._config.cpu_group_size
            * self._config.cpu_units_per_worker
        )
        gpu_capacity = (
            self._config.gpu_groups
            * self._config.gpu_group_size
            * self._config.gpu_units_per_worker
        )
        for chain_id, state in self._chain_state.items():
            if state.enqueued_at is None or state.finished_at is None:
                continue
            latency = state.finished_at - state.enqueued_at
            service = state.chain.total_service_time
            share = state.chain.dominant_share(cpu_capacity, gpu_capacity)
            area = share * service
            result.chain_latencies[chain_id] = latency
            result.chain_service_times[chain_id] = service
            result.chain_areas[chain_id] = area
            result.user_areas[state.chain.user_id] = (
                result.user_areas.get(state.chain.user_id, 0.0) + area
            )
        result.worker_busy_seconds = self._worker_busy_seconds
        result.total_worker_seconds = self._pool.total_worker_seconds(self._clock.now())
        result.bubble_seconds = self._bubble_seconds
        return result


async def run_simulation(
    workload: Workload,
    manager: PriorityJobManager,
    config: SimulationConfig | None = None,
    clock: VirtualClock | None = None,
) -> SimulationResult:
    """Convenience wrapper: build a runner and run it."""
    runner = SimulationRunner(workload, manager, config, clock)
    return await runner.run()
