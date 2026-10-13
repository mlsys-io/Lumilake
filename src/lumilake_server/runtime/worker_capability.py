"""FlowMesh task types that only some workers' executors run.

A worker profile lists the task types its executors advertise under
``supported_task_types``. A node whose task type is gated here is placed only on
a CPU worker that advertises it; every other node is placed by engine alone.
python runs only on Docker-backed workers, so it is the one gated type.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from lumilake_server.runtime import python_step

CAPABILITY_GATED_TASK_TYPES = frozenset({python_step.TASK_TYPE})

CPU_ONLY_TASK_TYPES = frozenset({"data_retrieval", "api", "python"})
"""Task types that run on a CPU worker only (HALO's CPU-only engines)."""


def advertised_task_types(profile: Mapping[str, Any]) -> frozenset[str]:
    """The task types a worker profile says its executors run."""
    return frozenset(
        str(task_type) for task_type in profile.get("supported_task_types", ())
    )


def required_task_types(task_types: Iterable[str]) -> frozenset[str]:
    """The gated task types among ``task_types``."""
    return CAPABILITY_GATED_TASK_TYPES.intersection(task_types)


def requires_cpu_worker(task_types: Iterable[str]) -> bool:
    """True when any of ``task_types`` runs on a CPU worker only.

    A batch with such a node claims a CPU worker even when the configured CPU
    group size is zero.
    """
    return any(
        (task_type or "").strip().lower() in CPU_ONLY_TASK_TYPES
        for task_type in task_types
    )
