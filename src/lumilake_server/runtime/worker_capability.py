"""FlowMesh task types that only some workers' executors run.

A worker profile lists the task types its executors advertise under
``supported_task_types``. A node whose task type is gated here is placed only on
a CPU worker that advertises it; every other node is placed by engine alone.
python runs only on Docker-backed workers, so it is the gated type today.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from lumilake_server.runtime import python_step

CAPABILITY_GATED_TASK_TYPES = frozenset({python_step.TASK_TYPE})


def advertised_task_types(profile: Mapping[str, Any]) -> frozenset[str]:
    """The task types a worker profile says its executors run."""
    return frozenset(
        str(task_type) for task_type in profile.get("supported_task_types", ())
    )


def required_task_types(task_types: Iterable[str]) -> frozenset[str]:
    """The gated task types among ``task_types``."""
    return CAPABILITY_GATED_TASK_TYPES.intersection(task_types)
