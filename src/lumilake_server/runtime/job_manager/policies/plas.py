"""Program-level least-attained-service scheduling policy (Autellix)."""

from typing import Any

from lumilake_server.runtime.job_manager.base import WorkflowItem
from lumilake_server.runtime.job_manager.cost import CostParams, estimate_area

from .base import BaseSchedulingPolicy

__all__ = ["PlasSchedulingPolicy"]


class PlasSchedulingPolicy(BaseSchedulingPolicy):
    """Select the batch by least attained service of the item's chain.

    The chain key is ``chain_id`` for a dynamic round, else the request id for
    a standalone job. Chains with the lowest attained service sort first; ties
    break by enqueue order. Each committed item charges its ``estimate_area``
    to the chain (items the cost model cannot estimate charge nothing).
    """

    def __init__(
        self,
        *,
        cost_params: CostParams | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._cost_params = cost_params or CostParams()
        self._attained: dict[str, float] = {}

    @staticmethod
    def _chain_key(item: WorkflowItem) -> str:
        return item.config.chain_id or item.request_id

    def select_batch(
        self,
        candidates: list[WorkflowItem],
        affinity_rank: dict[str, int],
        batch_size: int,
    ) -> list[str]:
        ordered = sorted(
            candidates,
            key=lambda item: (
                self._attained.get(self._chain_key(item), 0.0),
                item.enqueued_at,
                item.workflow_id,
            ),
        )
        return [item.workflow_id for item in ordered[:batch_size]]

    def on_commit(self, workflows: list[WorkflowItem]) -> None:
        for charged in workflows:
            area = estimate_area(charged, self._cost_params)
            if area is None:
                continue
            key = self._chain_key(charged)
            self._attained[key] = self._attained.get(key, 0.0) + area
