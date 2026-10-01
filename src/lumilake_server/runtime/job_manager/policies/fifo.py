"""First-in-first-out scheduling policy: enqueue order."""

from lumilake_server.runtime.job_manager.base import WorkflowItem

from .base import BaseSchedulingPolicy

__all__ = ["FifoSchedulingPolicy"]


class FifoSchedulingPolicy(BaseSchedulingPolicy):
    """Select the batch in enqueue order (``enqueued_at``, then ``workflow_id``)."""

    def select_batch(
        self,
        candidates: list[WorkflowItem],
        affinity_rank: dict[str, int],
        batch_size: int,
    ) -> list[str]:
        ordered = sorted(
            candidates,
            key=lambda item: (item.enqueued_at, item.workflow_id),
        )
        return [item.workflow_id for item in ordered[:batch_size]]
