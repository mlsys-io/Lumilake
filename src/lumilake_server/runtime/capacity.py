"""Capacity snapshot for capacity-aware batch selection."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from lumilake_server.data_specs import GpuRequest
from lumilake_server.runtime.worker_capability import (
    advertised_task_types,
    gated_nodes_gpu_requirements,
    gpu_worker_can_run,
    holds_gpu_utilizations,
)


@dataclass(frozen=True, slots=True)
class FreeCapacity:
    """Workers currently idle, split by class.

    ``profiles`` maps each idle worker id to its raw FlowMesh profile so
    capacity-aware selection can filter candidates by hardware requirements
    (``_worker_meets_hardware``) before claiming.
    """

    cpu_worker_ids: tuple[str, ...]
    gpu_worker_ids: tuple[str, ...]
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def cpu_count(self) -> int:
        return len(self.cpu_worker_ids)

    @property
    def gpu_count(self) -> int:
        return len(self.gpu_worker_ids)

    def is_empty(self) -> bool:
        return not self.cpu_worker_ids and not self.gpu_worker_ids

    def has_cpu_worker_advertising(self, task_types: frozenset[str]) -> bool:
        """True when some CPU worker in this snapshot advertises every task type."""
        return any(
            task_types <= advertised_task_types(self.profiles.get(w, {}))
            for w in self.cpu_worker_ids
        )

    def has_gpu_worker_holding(self, utilizations: frozenset[float]) -> bool:
        """True when some GPU worker in this snapshot holds every utilization."""
        return any(
            holds_gpu_utilizations(self.profiles.get(w, {}), utilizations)
            for w in self.gpu_worker_ids
        )

    def can_satisfy(self, *, cpu_group_size: int, gpu_group_size: int) -> bool:
        """True when this snapshot holds enough idle workers of each class."""
        return self.cpu_count >= cpu_group_size and self.gpu_count >= gpu_group_size

    def dispatch_count(
        self,
        *,
        cpu_group_size: int,
        gpu_group_size: int,
        worker_meets_hardware: Any,
        hardware: Any,
        required_task_types: frozenset[str] = frozenset(),
        required_gpu_task_types: frozenset[str] = frozenset(),
        required_gpu_requests: Sequence[GpuRequest] = (),
    ) -> int:
        """Number of whole dispatches the idle workers can serve at once.

        A dispatch claims ``cpu_group_size`` CPU workers and ``gpu_group_size``
        GPU workers (a class with group size ``0`` is not claimed). The count is
        the smallest ``eligible // group size`` over the claimed classes, where
        eligible workers meet ``hardware`` when the snapshot carries profiles.
        ``required_task_types`` are the gated task types the CPU group must
        cover and ``required_gpu_task_types`` those the GPU group must cover:
        each dispatch's group holds an eligible worker advertising all of them,
        so the count is also bounded by the number of such workers, and is
        ``0`` when there is none, no group of that class, or no profiles (the
        cases in which :meth:`eligible_workers` returns ``None``). Each GPU
        requirement (a step's own request with the task types it needs) bounds
        the count by the workers that run it. ``0`` when no class is claimed.
        """
        eligible_cpu = self._meeting_hardware(
            self.cpu_worker_ids, worker_meets_hardware, hardware
        )
        eligible_gpu = self._meeting_hardware(
            self.gpu_worker_ids, worker_meets_hardware, hardware
        )
        counts = [
            len(ids) // group_size
            for ids, group_size in (
                (eligible_cpu, cpu_group_size),
                (eligible_gpu, gpu_group_size),
            )
            if group_size > 0
        ]
        if required_task_types:
            if cpu_group_size < 1 or not self.profiles:
                return 0
            capable_cpu = [
                w
                for w in eligible_cpu
                if required_task_types
                <= advertised_task_types(self.profiles.get(w, {}))
            ]
            counts.append(len(capable_cpu))
        if required_gpu_task_types or required_gpu_requests:
            if gpu_group_size < 1 or not self.profiles:
                return 0
            # Concurrent dispatches draw their groups from the same workers, so each
            # requirement bounds the count by the workers that serve it.
            gates = self._gpu_gates(
                eligible_gpu, required_gpu_task_types, required_gpu_requests
            )
            counts.extend(len(gate) for gate in gates)
        return min(counts, default=0)

    def _meeting_hardware(
        self,
        worker_ids: tuple[str, ...],
        worker_meets_hardware: Any,
        hardware: Any,
    ) -> list[str]:
        if not self.profiles:
            return list(worker_ids)
        return [
            w
            for w in worker_ids
            if worker_meets_hardware(self.profiles.get(w, {}), hardware)
        ]

    def can_satisfy_hardware(
        self,
        *,
        cpu_group_size: int,
        gpu_group_size: int,
        worker_meets_hardware: Any,
        hardware: Any,
        required_task_types: frozenset[str] = frozenset(),
        required_gpu_task_types: frozenset[str] = frozenset(),
        required_gpu_requests: Sequence[GpuRequest] = (),
    ) -> bool:
        """True when enough idle workers of each class also meet ``hardware``.

        ``worker_meets_hardware`` is a callable ``(profile, hardware) -> bool``
        (the server's ``_worker_meets_hardware``); workers without a recorded
        profile are treated as meeting the requirement so a missing profile
        degrades to "no constraint" rather than stalling selection.
        """
        return (
            self.eligible_workers(
                cpu_group_size=cpu_group_size,
                gpu_group_size=gpu_group_size,
                worker_meets_hardware=worker_meets_hardware,
                hardware=hardware,
                required_task_types=required_task_types,
                required_gpu_task_types=required_gpu_task_types,
                required_gpu_requests=required_gpu_requests,
            )
            is not None
        )

    def eligible_workers(
        self,
        *,
        cpu_group_size: int,
        gpu_group_size: int,
        worker_meets_hardware: Any,
        hardware: Any,
        required_task_types: frozenset[str] = frozenset(),
        required_gpu_task_types: frozenset[str] = frozenset(),
        required_gpu_requests: Sequence[GpuRequest] = (),
        required_gpu_utilizations: frozenset[float] = frozenset(),
    ) -> tuple[list[str], list[str]] | None:
        """Return ``(selected_cpu, selected_gpu)`` workers that meet ``hardware``, or
        ``None`` while the free set cannot serve the batch: too few idle workers, no CPU
        worker advertising a required task type, no GPU worker advertising a required
        GPU task type or carrying the step's own GPU request, or no GPU worker whose
        memory fraction holds the batch's largest utilization."""
        if self.cpu_count < cpu_group_size or self.gpu_count < gpu_group_size:
            return None
        if required_task_types and (cpu_group_size < 1 or not self.profiles):
            return None
        if (required_gpu_task_types or required_gpu_requests) and (
            gpu_group_size < 1 or not self.profiles
        ):
            return None
        fraction_gate = required_gpu_utilizations if gpu_group_size > 0 else frozenset()
        if fraction_gate and not self.profiles:
            return None
        if not self.profiles:
            return (
                list(self.cpu_worker_ids[:cpu_group_size]),
                list(self.gpu_worker_ids[:gpu_group_size]),
            )
        eligible_cpu = self._meeting_hardware(
            self.cpu_worker_ids, worker_meets_hardware, hardware
        )
        eligible_gpu = self._meeting_hardware(
            self.gpu_worker_ids, worker_meets_hardware, hardware
        )
        if len(eligible_cpu) < cpu_group_size or len(eligible_gpu) < gpu_group_size:
            return None
        selected_cpu = eligible_cpu[:cpu_group_size]
        if required_task_types:
            capable = [
                w
                for w in eligible_cpu
                if required_task_types
                <= advertised_task_types(self.profiles.get(w, {}))
            ]
            if not capable:
                return None
            if not any(w in capable for w in selected_cpu):
                selected_cpu[-1] = capable[0]
        selected_gpu = eligible_gpu[:gpu_group_size]
        gates = self._gpu_gates(
            eligible_gpu, required_gpu_task_types, required_gpu_requests
        )
        if gates:
            if not all(gates):
                return None
            if gpu_group_size == 1:
                shared = frozenset.intersection(*gates)
                if not shared:
                    return None
                if selected_gpu[0] not in shared:
                    selected_gpu[0] = next(w for w in eligible_gpu if w in shared)
            else:
                for gate in gates:
                    if any(w in gate for w in selected_gpu):
                        continue
                    if not _swap_into(selected_gpu, eligible_gpu, gate, gates):
                        return None
        if fraction_gate:
            # A worker's memory fraction holds every utilization at or below it, so a
            # group covering the largest one covers them all.
            largest = max(fraction_gate)
            capable = [
                w
                for w in eligible_gpu
                if holds_gpu_utilizations(self.profiles.get(w, {}), {largest})
            ]
            if not capable:
                return None
            if not any(w in capable for w in selected_gpu):
                if not _swap_into(
                    selected_gpu, eligible_gpu, frozenset(capable), gates
                ):
                    return None
        return (selected_cpu, selected_gpu)

    def _gpu_gates(
        self,
        workers: Sequence[str],
        required_task_types: frozenset[str],
        required_requests: Sequence[GpuRequest],
    ) -> list[frozenset[str]]:
        """The workers of ``workers`` that serve each GPU-group requirement."""
        return [
            frozenset(
                worker
                for worker in workers
                if gpu_worker_can_run(
                    self.profiles.get(worker, {}), task_types, request
                )
            )
            for task_types, request in gated_nodes_gpu_requirements(
                required_task_types, required_requests
            )
        ]


def _swap_into(
    selected: list[str],
    workers: Sequence[str],
    gate: frozenset[str],
    gates: Sequence[frozenset[str]],
) -> bool:
    """Put a worker of ``gate`` into ``selected`` at a slot no covered gate needs."""
    candidate = next((w for w in workers if w in gate and w not in selected), None)
    if candidate is None:
        return False
    covered = [g for g in gates if g != gate and any(w in g for w in selected)]
    for slot in reversed(range(len(selected))):
        rest = selected[:slot] + selected[slot + 1 :]
        if all(any(w in g for w in rest) for g in covered):
            selected[slot] = candidate
            return True
    return False
