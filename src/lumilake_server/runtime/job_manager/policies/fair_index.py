"""Fair-weighted index scheduling policy.

Orders candidates by ``w(user) / p_hat(item)`` where
``w(user) = 1 / (1 + attained(user) / fair_share_target)`` — Smith's rule with
a fairness weight, preferring cheap under-served work. Items the analytic cost
model cannot estimate fall back to least-attained-service. Attained service is
charged on commit via :meth:`on_commit`.
"""

from typing import Any

from lumilake_server.runtime.job_manager.attained import AttainedService
from lumilake_server.runtime.job_manager.base import WorkflowItem
from lumilake_server.runtime.job_manager.cost import CostParams, estimate_area

from .base import BaseSchedulingPolicy

__all__ = ["FairIndexSchedulingPolicy"]


class FairIndexSchedulingPolicy(BaseSchedulingPolicy):
    """Select the batch by the fair-weighted index ``w(user) / p_hat(item)``.

    ``w(user) = 1 / (1 + attained(user) / fair_share_target)``; higher index
    first. Items the cost model cannot estimate fall back to
    least-attained-service (ranked as if ``p_hat`` were the user's current
    decayed attained area). Affinity's ordering breaks ties so clustering
    still shapes the batch.
    """

    def __init__(
        self,
        *,
        fair_share_target: float,
        fairness_half_life_seconds: float,
        cost_params: CostParams | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._fair_share_target = fair_share_target
        self._cost_params = cost_params or CostParams()
        self._attained = AttainedService(fairness_half_life_seconds, clock=self._clock)

    def select_batch(
        self,
        candidates: list[WorkflowItem],
        affinity_rank: dict[str, int],
        batch_size: int,
    ) -> list[str]:
        rank: dict[str, float] = {}
        for item in candidates:
            user_id = item.config.user_id
            attained = self._attained.get(user_id)
            weight = 1.0 / (1.0 + attained / self._fair_share_target)
            p_hat = estimate_area(item, self._cost_params)
            if p_hat is None or p_hat <= 0:
                # LAS fallback: treat p_hat as the user's attained area.
                index = weight / max(attained, 1e-9)
            else:
                index = weight / p_hat
            rank[item.workflow_id] = index

        ordered = sorted(
            candidates,
            key=lambda item: (
                -rank[item.workflow_id],
                affinity_rank.get(item.workflow_id, len(affinity_rank)),
            ),
        )
        return [item.workflow_id for item in ordered[:batch_size]]

    def on_commit(self, workflows: list[WorkflowItem]) -> None:
        for charged in workflows:
            area = estimate_area(charged, self._cost_params)
            if area is not None and area > 0:
                self._attained.charge(charged.config.user_id, area)
