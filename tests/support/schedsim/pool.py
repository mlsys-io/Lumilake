"""Simulated worker pool honouring Lumilake's binary busy/free worker model.

Workers are grouped: a dispatched batch claims one group of its worker type
(a CPU group or a GPU group) for the batch's duration. Group sizes are
parameterised so the harness can model co-located GPU workers or multi-core
CPU groups.
"""

from dataclasses import dataclass, field


@dataclass(slots=True)
class WorkerPool:
    """A pool of CPU and GPU worker groups.

    ``cpu_groups`` / ``gpu_groups`` are the number of independently claimable
    groups; ``cpu_group_size`` / ``gpu_group_size`` are the workers per group.
    A group is binary busy/free — there is no fractional capacity.
    """

    cpu_groups: int
    gpu_groups: int
    cpu_group_size: int = 1
    gpu_group_size: int = 1

    _cpu_free: set[int] = field(default_factory=set, init=False)
    _gpu_free: set[int] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self._cpu_free = set(range(self.cpu_groups))
        self._gpu_free = set(range(self.gpu_groups))

    def free_cpu_groups(self) -> int:
        return len(self._cpu_free)

    def free_gpu_groups(self) -> int:
        return len(self._gpu_free)

    def claim_cpu_group(self) -> int | None:
        """Claim one free CPU group, returning its id (or ``None`` if none)."""
        if not self._cpu_free:
            return None
        return self._cpu_free.pop()

    def claim_gpu_group(self) -> int | None:
        """Claim one free GPU group, returning its id (or ``None`` if none)."""
        if not self._gpu_free:
            return None
        return self._gpu_free.pop()

    def release_cpu_group(self, group_id: int) -> None:
        self._cpu_free.add(group_id)

    def release_gpu_group(self, group_id: int) -> None:
        self._gpu_free.add(group_id)

    def total_worker_seconds(self, horizon: float) -> float:
        """Total worker-seconds available over ``horizon`` seconds."""
        cpu_workers = self.cpu_groups * self.cpu_group_size
        gpu_workers = self.gpu_groups * self.gpu_group_size
        return (cpu_workers + gpu_workers) * horizon
