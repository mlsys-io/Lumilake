"""Set-index scheduling policy.

Selects the batch maximising ``rho(S) = W(S) / T(S)`` where ``W(S)`` is the
sum of item urgency weights and ``T(S) = base(S) + sigma * |distinct models
in S not resident|``. Theorem A of the scheduling formulation proves ``rho``
is the optimal sequencing key for the batched weighted-completion-time
objective; model affinity enters through the setup term rather than a
hand-tuned distance.

A batch executes concurrently on a worker group, so ``base(S) = max`` of the
member service times (not the sum) — the harness models it this way
(``runner.py``: ``block_base = max(meta.service ...)``). Corrected Theorem B
then fixes a model set ``M`` *and* a service threshold ``p*``: among items with
``mu(i) in M`` and ``p_i <= p*`` the batch cost ``T = p* + sigma*|M \\ {m_g}|``
is constant, so the top-``B`` by weight is optimal for that ``(M, p*)`` pair.
We enumerate ``(M, p*)`` pairs — model sets, never item subsets — and keep the
best ``rho`` overall.
"""

from collections.abc import Callable
from itertools import combinations
from typing import Any

from lumilake_server.runtime.job_manager.attained import AttainedService
from lumilake_server.runtime.job_manager.base import WorkflowItem
from lumilake_server.runtime.job_manager.cost import CostParams, estimate_area

from .base import BaseSchedulingPolicy

__all__ = ["SetIndexSchedulingPolicy"]

# Largest distinct-model set enumerated per call. In practice the resident
# model alone and the resident model plus one other suffice; keeping the cap
# a named constant makes the polynomial bound explicit.
MODEL_SET_CAP = 2


def _required_model(item: WorkflowItem) -> str | None:
    """The model an item requires, or ``None`` when it needs no model.

    Mirrors the affinity extractor: the model of the first LLM-backend op.
    """
    for op in item.runtime_graph.nodes.values():
        if op.backend in {"vllm", "transformers", "diffusers"} and op.model:
            return op.model
    return None


class SetIndexSchedulingPolicy(BaseSchedulingPolicy):
    """Select the batch maximising ``rho = W(S) / T(S)`` over small model sets.

    ``setup_cost_sigma`` is the per-model setup cost: one charge per distinct
    required model not already resident on the group. ``resident_model`` is the
    model last served (``None`` for a fresh group, where every model is
    charged); it is updated on commit to the last model served.
    """

    def __init__(
        self,
        *,
        setup_cost_sigma: float,
        fair_share_target: float,
        fairness_half_life_seconds: float,
        cost_params: CostParams | None = None,
        resident_model: str | None = None,
        weight_fn: Callable[[WorkflowItem], float] | None = None,
        service_fn: Callable[[WorkflowItem], float] | None = None,
        model_set_cap: int = MODEL_SET_CAP,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if setup_cost_sigma < 0:
            raise ValueError("setup_cost_sigma must be >= 0")
        if model_set_cap < 1:
            raise ValueError("model_set_cap must be >= 1")
        self._sigma = setup_cost_sigma
        self._cost_params = cost_params or CostParams()
        self._resident_model = resident_model
        self._model_set_cap = model_set_cap
        self._fair_share_target = fair_share_target
        self._attained = AttainedService(fairness_half_life_seconds, clock=self._clock)
        self._weight_fn = weight_fn or self._default_weight
        self._service_fn = service_fn or self._default_service

    def _default_weight(self, item: WorkflowItem) -> float:
        """Urgency weight from attained service: ``1 / (1 + attained / target)``."""
        attained = self._attained.get(item.config.user_id)
        return 1.0 / (1.0 + attained / self._fair_share_target)

    def _default_service(self, item: WorkflowItem) -> float:
        """Estimated service time (resource-area). Unestimable items never set
        the block length, so they get 0.0 and stay admissible."""
        area = estimate_area(item, self._cost_params)
        return area if area is not None and area > 0 else 0.0

    def select_batch(
        self,
        candidates: list[WorkflowItem],
        affinity_rank: dict[str, int],
        batch_size: int,
    ) -> list[str]:
        if batch_size <= 0 or not candidates:
            return []

        # Group candidates by required model.
        by_model: dict[str | None, list[WorkflowItem]] = {}
        for item in candidates:
            by_model.setdefault(_required_model(item), []).append(item)

        # Items with no required model are model-agnostic: they never incur a
        # setup charge and can join any model set.
        agnostic = by_model.pop(None, [])

        # Enumerate small distinct-model sets M (|M| <= cap) over the models
        # present in the candidate pool plus the resident model. For a fixed M
        # and batch size T is constant (Theorem B), so the best batch is the
        # top-B by weight among items wanting a model in M. We enumerate model
        # sets, never item subsets. Non-resident pairs (e.g. {B, C} with A
        # resident) are included: they pay two setup charges, but a
        # weight-heavy pair can beat a resident-containing pair, so excluding
        # them would be a silent restriction.
        #
        # The empty model set is always enumerated: its pool is the agnostic
        # items alone, with zero setup charge. This keeps an all-agnostic batch
        # reachable even when models are present (it pays no setup and can beat
        # a model-set batch that does), and is what lets a CPU/DB-only partition
        # dispatch at all — without it, an empty universe yields no model sets
        # and select_batch would return [] and starve the partition.
        resident = self._resident_model
        universe = {m for m in by_model if m is not None}
        if resident is not None:
            universe.add(resident)
        model_sets: list[set[str]] = [set()]
        for size in range(1, min(self._model_set_cap, len(universe)) + 1):
            for combo in combinations(sorted(universe), size):
                model_sets.append(set(combo))

        best_ids: list[str] = []
        best_rho = -1.0
        for model_set in model_sets:
            pool = list(agnostic)
            for model in model_set:
                pool.extend(by_model.get(model, []))
            if not pool:
                continue
            rho, ids = self._best_for_model_set(pool, model_set, batch_size)
            if rho > best_rho:
                best_rho = rho
                best_ids = ids

        return best_ids

    def _best_for_model_set(
        self,
        pool: list[WorkflowItem],
        model_set: set[str],
        batch_size: int,
    ) -> tuple[float, list[str]]:
        """Best ``(rho, ids)`` over ``(M, p*)`` pairs for a fixed model set.

        Corrected Theorem B: sort by service ascending and walk the distinct
        service values as the threshold ``p*``; under each, the batch cost is
        ``p* + setup`` (constant), so the top-``B`` by weight among items with
        ``service <= p*`` is optimal for that pair.
        """
        setup = self._sigma * len(
            model_set - ({self._resident_model} if self._resident_model else set())
        )
        pool.sort(key=self._service_fn)
        admissible: list[WorkflowItem] = []
        best_ids: list[str] = []
        best_rho = -1.0
        i = 0
        while i < len(pool):
            p_star = self._service_fn(pool[i])
            while i < len(pool) and self._service_fn(pool[i]) == p_star:
                admissible.append(pool[i])
                i += 1
            top = sorted(admissible, key=lambda item: -self._weight_fn(item))[
                :batch_size
            ]
            weight = sum(self._weight_fn(item) for item in top)
            total = p_star + setup
            if total > 0:
                rho = weight / total
                if rho > best_rho:
                    best_rho = rho
                    best_ids = [item.workflow_id for item in top]
        return best_rho, best_ids

    def on_commit(self, workflows: list[WorkflowItem]) -> None:
        for charged in workflows:
            area = estimate_area(charged, self._cost_params)
            if area is not None and area > 0:
                self._attained.charge(charged.config.user_id, area)
        if workflows:
            last = workflows[-1]
            model = _required_model(last)
            if model is not None:
                self._resident_model = model
