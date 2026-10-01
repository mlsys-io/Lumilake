"""Built-in scheduling-policy registry.

``LUMILAKE_SCHEDULER_POLICY`` selects a policy by name; the registry resolves
it to a :class:`BaseSchedulingPolicy` subclass. A new policy is a class plus
one registry entry — no branches in the job manager.
"""

from lumilake import envs

from .affinity import AffinitySchedulingPolicy
from .base import BaseSchedulingPolicy
from .fifo import FifoSchedulingPolicy
from .legacy import LegacySchedulingPolicy
from .lpt import LptSchedulingPolicy
from .plas import PlasSchedulingPolicy
from .spt import SptSchedulingPolicy

SCHEDULING_POLICIES: dict[str, type[BaseSchedulingPolicy]] = {
    "legacy": LegacySchedulingPolicy,
    "fifo": FifoSchedulingPolicy,
    "spt": SptSchedulingPolicy,
    "lpt": LptSchedulingPolicy,
    "affinity": AffinitySchedulingPolicy,
    "plas": PlasSchedulingPolicy,
}


def create_scheduling_policy(
    policy: str | None = None, **kwargs
) -> BaseSchedulingPolicy:
    """Return a scheduling-policy instance for ``policy``.

    ``policy`` defaults to ``LUMILAKE_SCHEDULER_POLICY``. Raises ``ValueError``
    for a name that is not registered.
    """
    if policy is None:
        policy = envs.LUMILAKE_SCHEDULER_POLICY
    policy = policy.strip().lower()
    if policy not in SCHEDULING_POLICIES:
        raise ValueError(
            f"Unknown scheduling policy '{policy}'. "
            f"Registered: {sorted(SCHEDULING_POLICIES)}."
        )
    return SCHEDULING_POLICIES[policy](**kwargs)


__all__ = [
    "AffinitySchedulingPolicy",
    "BaseSchedulingPolicy",
    "FifoSchedulingPolicy",
    "LegacySchedulingPolicy",
    "LptSchedulingPolicy",
    "PlasSchedulingPolicy",
    "SCHEDULING_POLICIES",
    "SptSchedulingPolicy",
    "create_scheduling_policy",
]
