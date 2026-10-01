"""Pure affinity-clustering scheduling policy."""

from lumilake_server.runtime.job_manager.base import WorkflowItem

from .base import BaseSchedulingPolicy

__all__ = ["AffinitySchedulingPolicy"]


class AffinitySchedulingPolicy(BaseSchedulingPolicy):
    """Select the batch in pure affinity clustering order.

    The ids in ``affinity_rank`` (in order) come first, then the remaining
    candidates fill up to ``batch_size`` in enqueue order. This is ``default``
    without the per-user round-robin seed.
    """

    def select_batch(
        self,
        candidates: list[WorkflowItem],
        affinity_rank: dict[str, int],
        batch_size: int,
    ) -> list[str]:
        candidate_map = {item.workflow_id: item for item in candidates}
        selected_ids = [wid for wid in affinity_rank if wid in candidate_map]
        if len(selected_ids) >= batch_size:
            return selected_ids[:batch_size]

        selected_set = set(selected_ids)
        for item in sorted(candidates, key=lambda i: (i.enqueued_at, i.workflow_id)):
            if item.workflow_id in selected_set:
                continue
            selected_ids.append(item.workflow_id)
            if len(selected_ids) >= batch_size:
                break
        return selected_ids[:batch_size]
