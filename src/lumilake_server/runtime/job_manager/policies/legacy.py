"""The default scheduling policy: per-user round-robin fairness.

Reproduces the pre-policy selection exactly. Within a partition the candidate
pool is built round-robin across users (see ``PriorityJobManager``), so the
policy only needs to preserve that ordering when affinity's clustering would
otherwise reorder it: it keeps one item per user in round-robin order before
filling the rest of the batch.
"""

from lumilake_server.runtime.job_manager.base import WorkflowItem

from .base import BaseSchedulingPolicy

__all__ = ["LegacySchedulingPolicy"]


class LegacySchedulingPolicy(BaseSchedulingPolicy):
    """Default policy: per-user round-robin fairness within a partition."""

    def select_batch(
        self,
        candidates: list[WorkflowItem],
        affinity_rank: dict[str, int],
        batch_size: int,
    ) -> list[str]:
        if batch_size <= 1:
            return list(affinity_rank)[:batch_size]
        candidate_map = {item.workflow_id: item for item in candidates}
        selected_ids = [wid for wid in affinity_rank if wid in candidate_map]
        if not selected_ids:
            return []

        user_order: list[str] = []
        user_to_ids: dict[str, list[str]] = {}
        for item in candidates:
            owner_id = item.config.user_id
            if owner_id not in user_to_ids:
                user_to_ids[owner_id] = []
                user_order.append(owner_id)
            user_to_ids[owner_id].append(item.workflow_id)
        if len(user_order) <= 1:
            return selected_ids[:batch_size]

        selected_set = set(selected_ids)
        fair_seed: list[str] = []
        for user_id in user_order:
            preferred = next(
                (wid for wid in user_to_ids[user_id] if wid in selected_set),
                None,
            )
            picked = preferred or user_to_ids[user_id][0]
            if picked in fair_seed:
                continue
            fair_seed.append(picked)
            if len(fair_seed) >= batch_size:
                return fair_seed[:batch_size]

        final_ids: list[str] = list(fair_seed)
        for workflow_id in selected_ids:
            if workflow_id in final_ids:
                continue
            final_ids.append(workflow_id)
            if len(final_ids) >= batch_size:
                return final_ids[:batch_size]
        for item in candidates:
            if item.workflow_id in final_ids:
                continue
            final_ids.append(item.workflow_id)
            if len(final_ids) >= batch_size:
                break
        return final_ids[:batch_size]
