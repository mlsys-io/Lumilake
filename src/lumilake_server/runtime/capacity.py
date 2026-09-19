"""Capacity snapshot for capacity-aware batch selection."""

from dataclasses import dataclass, field
from typing import Any


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

    def can_satisfy(self, *, cpu_group_size: int, gpu_group_size: int) -> bool:
        """True when this snapshot holds enough idle workers of each class."""
        return self.cpu_count >= cpu_group_size and self.gpu_count >= gpu_group_size

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
    ) -> tuple[list[str], list[str]] | None:
        """Return ``(selected_cpu, selected_gpu)`` workers that meet ``hardware``.

        Returns ``None`` when the free set does not hold enough idle workers of
        each class that also meet ``hardware``. Searches the full free list so
        a later eligible worker is not missed just because an earlier one fails
        the hardware check; selection and claim share this helper so they
        cannot drift apart.
        """
        if self.cpu_count < cpu_group_size or self.gpu_count < gpu_group_size:
            return None
        if not self.profiles:
            return (
                list(self.cpu_worker_ids[:cpu_group_size]),
                list(self.gpu_worker_ids[:gpu_group_size]),
            )
        eligible_cpu = [
            w
            for w in self.cpu_worker_ids
            if worker_meets_hardware(self.profiles.get(w, {}), hardware)
        ]
        eligible_gpu = [
            w
            for w in self.gpu_worker_ids
            if worker_meets_hardware(self.profiles.get(w, {}), hardware)
        ]
        if len(eligible_cpu) < cpu_group_size or len(eligible_gpu) < gpu_group_size:
            return None
        return (
            eligible_cpu[:cpu_group_size],
            eligible_gpu[:gpu_group_size],
        )
