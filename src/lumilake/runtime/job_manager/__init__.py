"""Job manager implementations."""

from lumilake import envs

from .base import BaseJobManager, BatchSelection, Job, WorkflowItem
from .priority_queue import PriorityJobManager

JOB_MANAGER_TYPES: dict[str, type[BaseJobManager]] = {
    "priority": PriorityJobManager,
}


def create_job_manager(job_manager_type: str | None = None, **kwargs) -> BaseJobManager:
    """Create a job manager instance based on type."""
    if job_manager_type is None:
        job_manager_type = envs.LUMILAKE_JOB_MANAGER_TYPE
    job_manager_type = job_manager_type.lower()
    if job_manager_type not in JOB_MANAGER_TYPES:
        valid_types = ", ".join(JOB_MANAGER_TYPES.keys())
        raise ValueError(
            f"Unknown job manager type '{job_manager_type}'. Valid types: {valid_types}"
        )
    return JOB_MANAGER_TYPES[job_manager_type](**kwargs)


__all__ = [
    "BaseJobManager",
    "BatchSelection",
    "Job",
    "create_job_manager",
    "JOB_MANAGER_TYPES",
    "WorkflowItem",
    "PriorityJobManager",
]
