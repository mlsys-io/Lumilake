"""Shortest-processing-time scheduling policy: smallest estimated area first."""

from typing import Any

from lumilake_server.runtime.job_manager.base import WorkflowItem
from lumilake_server.runtime.job_manager.cost import CostParams, estimate_area

from .base import BaseSchedulingPolicy

__all__ = ["SptSchedulingPolicy"]


class SptSchedulingPolicy(BaseSchedulingPolicy):
    """Select the batch by smallest ``estimate_area`` first.

    Items the cost model cannot estimate sort after all estimated ones; ties
    break by enqueue order.
    """

    def __init__(
        self,
        *,
        cost_params: CostParams | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._cost_params = cost_params or CostParams()

    def select_batch(
        self,
        candidates: list[WorkflowItem],
        affinity_rank: dict[str, int],
        batch_size: int,
    ) -> list[str]:
        def key(item: WorkflowItem) -> tuple[int, float, float, str]:
            area = estimate_area(item, self._cost_params)
            if area is None:
                # Estimated items (0) sort before unestimated (1).
                return (1, 0.0, item.enqueued_at, item.workflow_id)
            return (0, area, item.enqueued_at, item.workflow_id)

        ordered = sorted(candidates, key=key)
        return [item.workflow_id for item in ordered[:batch_size]]
