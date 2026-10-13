"""Built-in scheduling-policy registry.

``LUMILAKE_SCHEDULER_POLICY`` selects a policy by name; the registry resolves
it to a :class:`BaseSchedulingPolicy` subclass. A new policy is a class plus
one registry entry — no branches in the job manager.
"""

import json
from typing import Any

from lumilake import envs

from .base import BaseSchedulingPolicy
from .cohort_pack import CohortPackArgs, CohortPackSchedulingPolicy
from .default import DefaultSchedulingPolicy
from .fifo import FifoSchedulingPolicy
from .lpt import LptSchedulingPolicy
from .plas import PlasSchedulingPolicy
from .spt import SptSchedulingPolicy

SCHEDULING_POLICIES: dict[str, type[BaseSchedulingPolicy]] = {
    "default": DefaultSchedulingPolicy,
    "fifo": FifoSchedulingPolicy,
    "spt": SptSchedulingPolicy,
    "lpt": LptSchedulingPolicy,
    "plas": PlasSchedulingPolicy,
    "cohort_pack": CohortPackSchedulingPolicy,
}


def create_scheduling_policy(
    policy: str | None = None,
    custom_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> BaseSchedulingPolicy:
    """Return a scheduling-policy instance for ``policy``.

    ``policy`` defaults to ``LUMILAKE_SCHEDULER_POLICY``. ``custom_args`` are the
    policy's keyword arguments and default to the JSON object in
    ``LUMILAKE_SCHEDULER_CUSTOM_ARGS``; they are validated against the policy's
    ``args_model``. Raises ``ValueError`` for a name that is not registered, for
    custom args on a policy that takes none, and for invalid custom args.
    """
    if policy is None:
        policy = envs.LUMILAKE_SCHEDULER_POLICY
    policy = policy.strip().lower()
    if policy not in SCHEDULING_POLICIES:
        raise ValueError(
            f"Unknown scheduling policy '{policy}'. "
            f"Registered: {sorted(SCHEDULING_POLICIES)}."
        )
    cls = SCHEDULING_POLICIES[policy]
    if custom_args is None:
        custom_args = json.loads(envs.LUMILAKE_SCHEDULER_CUSTOM_ARGS)
    if cls.args_model is None:
        if custom_args:
            raise ValueError(
                f"Scheduling policy '{policy}' takes no custom args; "
                f"got {sorted(custom_args)}."
            )
        return cls(**kwargs)
    return cls(args=cls.args_model.model_validate(custom_args), **kwargs)


__all__ = [
    "BaseSchedulingPolicy",
    "CohortPackArgs",
    "CohortPackSchedulingPolicy",
    "DefaultSchedulingPolicy",
    "FifoSchedulingPolicy",
    "LptSchedulingPolicy",
    "PlasSchedulingPolicy",
    "SCHEDULING_POLICIES",
    "SptSchedulingPolicy",
    "create_scheduling_policy",
]
