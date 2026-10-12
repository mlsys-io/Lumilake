"""Capacity snapshot for capacity-aware batch selection."""

from dataclasses import dataclass, field
from typing import Any

from lumilake_server.runtime.worker_capability import advertised_task_types


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
    ) -> tuple[list[str], list[str]] | None:
        """Return ``(selected_cpu, selected_gpu)`` workers that meet ``hardware``.

        Returns ``None`` when the free set does not hold enough idle workers of
        each class that also meet ``hardware``. Searches the full free list so
        a later eligible worker is not missed just because an earlier one fails
        the hardware check; selection and claim share this helper so they
        cannot drift apart.

        ``required_task_types`` are the gated task types the batch needs. The
        selected CPU group must contain a worker that advertises all of them; the
        helper returns ``None`` when no free eligible CPU worker does, so the
        caller waits for one instead of claiming a group that cannot run the batch.
        """
        if self.cpu_count < cpu_group_size or self.gpu_count < gpu_group_size:
            return None
        if required_task_types and (cpu_group_size < 1 or not self.profiles):
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
        return (selected_cpu, eligible_gpu[:gpu_group_size])
