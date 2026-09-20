"""Set-index scheduling policy.

Selects the batch maximising ``rho(S) = W(S) / T(S)`` where ``W(S)`` is the
sum of item urgency weights and ``T(S) = base(S) + sigma * |distinct models
in S not resident|``. Theorem A of the scheduling formulation proves ``rho``
is the optimal sequencing key for the batched weighted-completion-time
objective; model affinity enters through the setup term rather than a
hand-tuned distance.

Tractability follows Theorem B: for a fixed distinct-model set ``M`` and
fixed batch size, ``T`` is constant, so the best batch is the top-``B`` by
weight among items wanting a model in ``M``. We enumerate small model sets
(resident alone, resident plus one other) — never item subsets.
"""

from collections.abc import Callable
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

    ``sigma`` is the per-model setup cost: one charge per distinct required
    model not already resident on the group. ``resident_model`` is the model
    last served (``None`` for a fresh group, where every model is charged);
    it is updated on commit to the last model served.
    """

    def __init__(
        self,
        *,
        sigma: float,
        fair_share_target: float,
        fairness_half_life_seconds: float,
        cost_params: CostParams | None = None,
        resident_model: str | None = None,
        weight_fn: Callable[[WorkflowItem], float] | None = None,
        model_set_cap: int = MODEL_SET_CAP,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if sigma < 0:
            raise ValueError("sigma must be >= 0")
        if model_set_cap < 1:
            raise ValueError("model_set_cap must be >= 1")
        self._sigma = sigma
        self._cost_params = cost_params or CostParams()
        self._resident_model = resident_model
        self._model_set_cap = model_set_cap
        self._fair_share_target = fair_share_target
        self._attained = AttainedService(fairness_half_life_seconds, clock=self._clock)
        self._weight_fn = weight_fn or self._default_weight

    def _default_weight(self, item: WorkflowItem) -> float:
        """Urgency weight from attained service: ``1 / (1 + attained / target)``."""
        attained = self._attained.get(item.config.user_id)
        return 1.0 / (1.0 + attained / self._fair_share_target)

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

        # Enumerate small distinct-model sets: the resident model alone, and
        # the resident model plus one other. Never item subsets (Theorem B).
        resident = self._resident_model
        model_sets: list[set[str]] = []
        if resident is not None:
            model_sets.append({resident})
        for model in by_model:
            if model is None:
                continue
            base = {resident} if resident is not None else set()
            base.add(model)
            if len(base) <= self._model_set_cap:
                model_sets.append(base)

        best_ids: list[str] = []
        best_rho = -1.0
        for model_set in model_sets:
            pool = list(agnostic)
            for model in model_set:
                pool.extend(by_model.get(model, []))
            if not pool:
                continue
            pool.sort(key=lambda item: -self._weight_fn(item))
            chosen = pool[:batch_size]
            rho = self._rho(chosen, model_set)
            if rho > best_rho:
                best_rho = rho
                best_ids = [item.workflow_id for item in chosen]

        return best_ids

    def _rho(self, batch: list[WorkflowItem], model_set: set[str]) -> float:
        """``rho = W(S) / T(S)`` for a batch whose distinct model set is ``M``."""
        weight = sum(self._weight_fn(item) for item in batch)
        base = 0.0
        for item in batch:
            area = estimate_area(item, self._cost_params)
            base += area if area is not None and area > 0 else 0.0
        setup_models = model_set - (
            {self._resident_model} if self._resident_model else set()
        )
        total = base + self._sigma * len(setup_models)
        if total <= 0:
            return 0.0
        return weight / total

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
