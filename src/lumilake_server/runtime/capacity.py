"""Capacity snapshot for capacity-aware batch selection."""

from dataclasses import dataclass, field
from typing import Any

from lumilake_server.runtime.worker_capability import (
    advertised_task_types,
    gpu_memory_fraction,
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
        """True when every utilization has some GPU worker whose fraction holds it."""
        return all(
            any(
                holds_gpu_utilizations(self.profiles.get(w, {}), {utilization})
                for w in self.gpu_worker_ids
            )
            for utilization in utilizations
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
    ) -> int:
        """Number of whole dispatches the idle workers can serve at once.

        A dispatch claims ``cpu_group_size`` CPU workers and ``gpu_group_size``
        GPU workers (a class with group size ``0`` is not claimed). The count is
        the smallest ``eligible // group size`` over the claimed classes, where
        eligible workers meet ``hardware`` when the snapshot carries profiles.
        ``required_task_types`` are the gated task types the dispatch needs:
        each dispatch's CPU group must hold an eligible worker advertising all
        of them, so the count is also bounded by the number of such workers, and
        is ``0`` when there is none, no CPU group, or no profiles (the cases in
        which :meth:`eligible_workers` returns ``None``). ``0`` when no class is
        claimed.
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
            counts.append(
                sum(
                    required_task_types
                    <= advertised_task_types(self.profiles.get(w, {}))
                    for w in eligible_cpu
                )
            )
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
        required_gpu_utilizations: frozenset[float] = frozenset(),
    ) -> tuple[list[str], list[str]] | None:
        """Return ``(selected_cpu, selected_gpu)`` workers that meet ``hardware``, or
        ``None`` while the free set cannot serve the batch: too few idle workers, no CPU
        worker advertising a required task type, or no GPU worker whose memory fraction
        holds every utilization the batch reserves. The GPU group claims one worker per
        distinct utilization before filling to ``gpu_group_size``, so a batch holding
        engines of different sizes keeps them on separate workers whenever the free set
        allows it."""
        if self.cpu_count < cpu_group_size or self.gpu_count < gpu_group_size:
            return None
        if required_task_types and (cpu_group_size < 1 or not self.profiles):
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
        if fraction_gate:
            covered = self._claim_gpu_per_utilization(eligible_gpu, fraction_gate)
            if covered is None:
                return None
            remaining = [w for w in eligible_gpu if w not in covered]
            selected_gpu = covered + remaining[: max(0, gpu_group_size - len(covered))]
        else:
            selected_gpu = eligible_gpu[:gpu_group_size]
        return (selected_cpu, selected_gpu)

    def _claim_gpu_per_utilization(
        self, eligible_gpu: list[str], utilizations: frozenset[float]
    ) -> list[str] | None:
        """One eligible worker per reserved utilization, ``None`` when one is unheld.

        Utilizations are taken smallest first, and each takes the tightest worker
        holding it that is not claimed yet (the smallest memory fraction, a whole
        device last). An already-claimed worker is reused only when no unclaimed
        holder exists, so engines of different sizes stay on separate workers
        whenever the free set allows it, while a cluster with one large enough
        worker still runs them.
        """
        claimed: list[str] = []
        for utilization in sorted(utilizations):
            holders = [
                w
                for w in eligible_gpu
                if holds_gpu_utilizations(self.profiles.get(w, {}), {utilization})
            ]
            if not holders:
                return None
            unclaimed = [w for w in holders if w not in claimed]
            if not unclaimed:
                continue
            claimed.append(min(unclaimed, key=self._gpu_tightness))
        return claimed

    def _gpu_tightness(self, worker_id: str) -> tuple[bool, float]:
        """Sort key preferring the smallest memory fraction, a whole device last."""
        fraction = gpu_memory_fraction(self.profiles.get(worker_id, {}))
        if fraction is None:
            return (True, 0.0)
        return (False, fraction)
