"""Cohort-packing scheduling policy: template-pure units of chains."""

import itertools
import math
from dataclasses import dataclass
from typing import Any

from lumilake import envs
from lumilake_server.runtime.job_manager.base import WorkflowItem
from lumilake_server.runtime.job_manager.cost import CostParams, estimate_area

from .base import BaseSchedulingPolicy

__all__ = ["CohortPackSchedulingPolicy"]

Template = str
Key = tuple[str, int, Template]
"""``(chain, round, template)``: a group of slices that can share one batch."""


@dataclass(frozen=True)
class BatchCostModel:
    """The batch-cost constants the policy reasons with, in unit-speed time.

    A batch of ``k`` slices over ``n_d`` distinct templates costs
    ``fixed_seconds + mix_seconds * (n_d - 1)`` plus the slice work scaled by
    ``(k / batch_size) ** (beta - 1)`` per slice. The two seconds costs are
    converted to slice units by dividing by ``area_to_seconds * area``, where
    ``area`` is the per-slice ``estimate_area`` of the anchor unit's first round
    (``1`` when unestimable); candidate batches are then compared by slice count
    alone, so every slice counts as one equally costly unit of work.
    """

    fixed_seconds: float
    mix_seconds: float
    beta: float
    area_to_seconds: float
    batch_size: int

    def fixed(self, area: float) -> float:
        """The fixed cost per batch in slice units, for slices of ``area``."""
        return self.fixed_seconds / (self.area_to_seconds * max(area, 1e-9))

    def mix(self, area: float) -> float:
        """The cost per extra distinct template in slice units, for ``area``."""
        return self.mix_seconds / (self.area_to_seconds * max(area, 1e-9))

    def scale(self, k: float) -> float:
        """Per-slice work factor of a batch of ``k`` slices."""
        return (max(k, 1.0) / self.batch_size) ** (self.beta - 1.0)

    def per_work(self, k: int, n_d: int, fixed: float, mix: float) -> float:
        """Estimated unit-speed time per slice of a batch, in slice units."""
        return (fixed + mix * (n_d - 1)) / k + self.scale(k)

    def spread_n(self, ready: int, n_max: int, gamma: float, fixed: float) -> int:
        """Number of batches to split ``ready`` slices over (>= ceil(ready / B)).

        ``T(n)`` is the time of one of ``n`` equal batches, ``L(n)`` the
        lane-time of all of them; ``n`` grows while one more batch saves at
        least ``gamma`` times the lane-time it costs.
        """
        n = max(1, math.ceil(ready / self.batch_size))
        n_max = max(n, min(n_max, ready))

        def time_of(m: int) -> float:
            return fixed + (ready / m) * self.scale(ready / m)

        def lane_time(m: int) -> float:
            return m * fixed + ready * self.scale(ready / m)

        while n < n_max and time_of(n) - time_of(n + 1) >= gamma * (
            lane_time(n + 1) - lane_time(n)
        ):
            n += 1
        return n


@dataclass(frozen=True)
class _Unit:
    members: tuple[Key, ...]
    complete: bool


class CohortPackSchedulingPolicy(BaseSchedulingPolicy):
    """Dispatch units: balanced cuts of cohorts of chains sharing a template.

    Slices are grouped by ``(chain, round, template)``, so a group never mixes
    templates. The *cohort* of a ready ``(template, round r)`` is the chains
    ready on it plus every chain whose last committed round is ``r - 1``, whose
    committed template history up to ``r - 1`` matches theirs, and which is not
    itself ready on another template or round (a round-0 cohort has no mates).
    A mate counts with the size of the largest batch its last committed round
    went out in. There is no time bound, and Lumilake has no end-of-chain
    signal, so a chain that has ended keeps counting as a mate; a unit waiting
    for it goes after more than ``patience`` misses, or at once when no
    complete unit is ready. Each selection cuts the cohort into balanced
    *units* of whole chains, in chain order, aiming at ``ceil(S / B)`` units of
    about ``S / ceil(S / B)`` slices (``S`` the cohort's slices, ``B`` the batch
    size); the cut never divides a chain between units, and a chain wider than
    ``B`` is a unit alone. A unit is *complete* when every chain cut into it is
    ready. The anchor unit is template-pure. Chains batched together usually
    arrive together and so are cut together again, but a chain that joins the
    cohort later can change the cut. A dispatch never exceeds the batch size,
    so an anchor unit with more than ``B`` slices goes out over several
    dispatches: this one takes its first ``B`` slices (members in cut order,
    each group's slices by enqueue time, slice index and workflow id) and the
    rest stays queued for later selections.

    The anchor is the complete unit whose slices have the highest miss count,
    oldest first among ties (earliest enqueue time of a member's first slice,
    then the key of its first member); when no unit is complete, the same rule
    picks among all units. An incomplete unit whose slices have been passed over
    more than ``patience`` dispatches goes first instead, so nothing starves:
    the one passed over most, then the lexically smallest chain id of its first
    member (a longer id ranks ahead of an id it extends), then the oldest. The
    job manager's global starvation limit overrides every policy, so the
    effective release threshold is the smaller of ``patience + 1`` policy-level
    misses and that limit; to let units wait for their mates as configured, set
    the limit above ``patience``.

    When the free-lane count is known, a unit may be split over lanes that
    would otherwise idle (the free lanes minus one per other ready unit): its
    batch is cut to ``ceil(n / lanes)`` slices, so the rest goes out on the
    idle lanes at the following dispatches, while the cost model says each
    extra batch saves at least ``gamma`` times the lane-time it costs; that
    batch is not filled. Otherwise the remaining room is filled with up to
    three whole ready rounds of any unit, of other templates too, only when that
    lowers the estimated time per slice. With no fixed or mixing cost and
    ``beta >= 1`` merging never lowers the time per slice, so the batch stays
    template-pure; with ``beta < 1`` the economy of scale alone can make merging
    pay. Costs are compared by slice count: the fixed and mixing costs are
    converted to slice units with the per-slice estimated area of the anchor
    unit's first round, and an item the cost model cannot estimate counts as
    area 1.
    """

    def __init__(
        self,
        *,
        fixed_seconds: float = envs.LUMILAKE_SCHEDULER_COHORT_PACK_FIXED_SECONDS,
        mix_seconds: float = envs.LUMILAKE_SCHEDULER_COHORT_PACK_MIX_SECONDS,
        beta: float = envs.LUMILAKE_SCHEDULER_COHORT_PACK_BETA,
        area_to_seconds: float = envs.LUMILAKE_SCHEDULER_COHORT_PACK_AREA_TO_SECONDS,
        gamma: float = envs.LUMILAKE_SCHEDULER_COHORT_PACK_GAMMA,
        model_batch_size: int = envs.LUMILAKE_SCHEDULER_COHORT_PACK_BATCH_SIZE,
        patience: int = envs.LUMILAKE_SCHEDULER_COHORT_PACK_PATIENCE,
        cost_params: CostParams | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = BatchCostModel(
            fixed_seconds=fixed_seconds,
            mix_seconds=mix_seconds,
            beta=beta,
            area_to_seconds=area_to_seconds,
            batch_size=model_batch_size,
        )
        self._gamma = gamma
        self._patience = patience
        self._cost_params = cost_params or CostParams()
        self._free_lanes: int | None = None
        # chain -> round -> templates of the slices committed so far
        self._history: dict[str, dict[int, frozenset[Template]]] = {}
        self._last_round: dict[str, int] = {}
        # round -> chains whose last committed round it is
        self._chains_at: dict[int, set[str]] = {}
        # chain -> slices in the largest batch of its last committed round
        self._width: dict[str, int] = {}

    def observe_free_lanes(self, free_lanes: int | None) -> None:
        self._free_lanes = free_lanes

    @staticmethod
    def _chain(item: WorkflowItem) -> str:
        return item.config.chain_id or item.request_id

    def on_commit(self, workflows: list[WorkflowItem]) -> None:
        seen: dict[tuple[str, int], int] = {}
        templates: dict[tuple[str, int], set[Template]] = {}
        for item in workflows:
            round_key = (self._chain(item), item.config.chain_round)
            seen[round_key] = seen.get(round_key, 0) + 1
            templates.setdefault(round_key, set()).add(item.template_hash)
        for (chain, rnd), count in seen.items():
            history = self._history.setdefault(chain, {})
            history[rnd] = history.get(rnd, frozenset()) | templates[(chain, rnd)]
            previous = self._last_round.get(chain)
            latest = rnd if previous is None else max(previous, rnd)
            if rnd == previous:
                self._width[chain] = max(self._width.get(chain, 0), count)
            elif rnd == latest:
                self._width[chain] = count
            if latest != previous:
                if previous is not None:
                    self._chains_at[previous].discard(chain)
                self._chains_at.setdefault(latest, set()).add(chain)
                self._last_round[chain] = latest

    def _signature(
        self, chain: str, rnd: int
    ) -> tuple[frozenset[Template] | None, ...]:
        history = self._history.get(chain, {})
        return tuple(history.get(i) for i in range(rnd))

    def _units(
        self, groups: dict[Key, list[WorkflowItem]], batch_size: int
    ) -> list[_Unit]:
        """Ready units, recomputed from the current cohorts on every call.

        The cohort of a ready round ``(template, r)`` is the chains ready on it
        plus the chains whose last committed round is ``r - 1``, whose committed
        template history up to ``r - 1`` matches theirs and which are not ready
        on another template or round. A mate is expected with the size of the
        largest batch its last committed round went out in (a round dispatched
        in several batches counts as its widest batch; ``1`` if unknown); there
        is no time bound, so a chain that has ended keeps counting as a mate.
        The cohort is cut into balanced units in chain order, never dividing a
        chain between units, aiming at ``ceil(S / B)`` units; a unit is complete
        when every chain cut into it is ready. Units are not remembered between
        calls: a chain that joins the cohort later, sorting between two earlier
        ones, can change where the cut falls.
        """
        by_template: dict[tuple[Template, int], dict[str, Key]] = {}
        for key in groups:
            by_template.setdefault((key[2], key[1]), {})[key[0]] = key
        ready_at: dict[str, set[tuple[Template, int]]] = {}
        for template_key, keys in by_template.items():
            for chain in keys:
                ready_at.setdefault(chain, set()).add(template_key)
        units: list[_Unit] = []
        for (template, rnd), ready in by_template.items():
            by_signature: dict[
                tuple[frozenset[Template] | None, ...], dict[str, Key]
            ] = {}
            for chain, key in ready.items():
                by_signature.setdefault(self._signature(chain, rnd), {})[chain] = key
            for signature, ready_chains in by_signature.items():
                width = {c: len(groups[k]) for c, k in ready_chains.items()}
                if rnd > 0:
                    for chain in self._chains_at.get(rnd - 1, ()):
                        if (
                            chain in ready_chains
                            or self._signature(chain, rnd) != signature
                        ):
                            continue
                        if chain in ready_at and (template, rnd) not in ready_at[chain]:
                            continue
                        width[chain] = self._width.get(chain, 0) or 1
                cohort = sorted(width)
                total = sum(width.values())
                count = max(1, math.ceil(total / batch_size))
                target = total / count
                part: list[str] = []
                size = parts = 0
                parts_out: list[list[str]] = []
                for chain in cohort:
                    w = width[chain]
                    if part and (
                        size + w > batch_size or (parts < count - 1 and size >= target)
                    ):
                        parts_out.append(part)
                        parts += 1
                        part, size = [], 0
                    part.append(chain)
                    size += w
                if part:
                    parts_out.append(part)
                for part in parts_out:
                    members = [ready_chains[c] for c in part if c in ready_chains]
                    if members:
                        units.append(_Unit(tuple(members), len(members) == len(part)))
        return units

    def select_batch(
        self,
        candidates: list[WorkflowItem],
        affinity_rank: dict[str, int],
        batch_size: int,
    ) -> list[str]:
        if not candidates:
            return []
        groups: dict[Key, list[WorkflowItem]] = {}
        for item in candidates:
            key = (self._chain(item), item.config.chain_round, item.template_hash)
            groups.setdefault(key, []).append(item)
        for items in groups.values():
            items.sort(key=lambda it: (it.enqueued_at, it.slice_index, it.workflow_id))
        units = self._units(groups, batch_size)

        def oldest(unit: _Unit) -> tuple[float, Key]:
            return (
                min(groups[k][0].enqueued_at for k in unit.members),
                unit.members[0],
            )

        def waited(unit: _Unit) -> int:
            return max(it.miss_count for k in unit.members for it in groups[k])

        def passed_over(unit: _Unit) -> tuple[int, tuple[int, ...]]:
            return (waited(unit), tuple(-ord(ch) for ch in unit.members[0][0]))

        overdue = [u for u in units if not u.complete and waited(u) > self._patience]
        complete = [u for u in units if u.complete]
        if overdue:
            top = max(passed_over(u) for u in overdue)
            unit = min((u for u in overdue if passed_over(u) == top), key=oldest)
        else:
            unit = min(complete or units, key=lambda u: (-waited(u), oldest(u)))

        out: list[WorkflowItem] = []
        for key in unit.members:
            out += groups[key]
        area = estimate_area(groups[unit.members[0]][0], self._cost_params) or 1.0
        fixed = self._model.fixed(area)
        mix = self._model.mix(area)
        if self._free_lanes is not None and self._free_lanes > 1:
            # Lanes that would idle: the free lanes minus one per other ready unit.
            avail = self._free_lanes - (len(units) - 1)
            if avail > 1:
                lanes = self._model.spread_n(
                    len(out), min(avail, len(out)), self._gamma, fixed
                )
                if lanes > 1:
                    share = max(1, math.ceil(len(out) / lanes))
                    return [it.workflow_id for it in out[:share]][:batch_size]
        if len(out) < batch_size:
            out += self._fill(unit, units, groups, batch_size, fixed, mix)
        return [it.workflow_id for it in out[:batch_size]]

    def _value(
        self,
        keys: tuple[Key, ...],
        distinct: int,
        groups: dict[Key, list[WorkflowItem]],
        fixed: float,
        mix: float,
    ) -> float:
        """Estimated unit-speed time per slice of a candidate batch (lower is
        better), treating slices as equally costly."""
        slices = sum(len(groups[k]) for k in keys)
        return self._model.per_work(slices, distinct, fixed, mix)

    def _fill(
        self,
        unit: _Unit,
        units: list[_Unit],
        groups: dict[Key, list[WorkflowItem]],
        batch_size: int,
        fixed: float,
        mix: float,
    ) -> list[WorkflowItem]:
        """Whole ready rounds (of any unit, any template) added to ``unit`` when
        that lowers the estimated time per slice.

        It considers adding one, two or three whole rounds that fit in the room
        left and keeps the set with the lowest estimated time per slice, values
        compared after rounding to nine decimals; the unit alone wins unless a
        set is strictly lower. It merges when mixing is cheaper than a batch's
        fixed cost and tops the batch up towards ``batch_size`` when there is
        economy of scale. Among equal values it breaks the fewest other units,
        then takes the rounds with the lowest total enqueue-order rank.
        """
        room = batch_size - sum(len(groups[k]) for k in unit.members)
        unit_template = unit.members[0][2]
        in_unit = set(unit.members)
        unit_of = {k: i for i, u in enumerate(units) for k in u.members}
        unit_size = [len(u.members) for u in units]
        order = sorted(groups, key=lambda k: (groups[k][0].enqueued_at, k))
        atoms = [
            (rank, key, key[2], len(groups[key]))
            for rank, key in enumerate(order)
            if key not in in_unit and len(groups[key]) <= room
        ]
        base = tuple(unit.members)
        best: tuple[float, int, int] = (
            round(self._value(base, 1, groups, fixed, mix), 9),
            0,
            0,
        )
        best_keys: tuple[Key, ...] = ()
        for count in (1, 2, 3):
            for combo in itertools.combinations(atoms, count):
                if sum(c[3] for c in combo) > room:
                    continue
                keys = tuple(c[1] for c in combo)
                templates = {c[2] for c in combo} - {unit_template}
                value = round(
                    self._value(base + keys, 1 + len(templates), groups, fixed, mix), 9
                )
                took: dict[int, int] = {}
                for c in combo:
                    index = unit_of.get(c[1])
                    if index is not None:
                        took[index] = took.get(index, 0) + 1
                broken = sum(1 for i, n in took.items() if n < unit_size[i])
                score = (value, broken, sum(c[0] for c in combo))
                if score < best:
                    best, best_keys = score, keys
        out: list[WorkflowItem] = []
        for key in best_keys:
            out += groups[key]
        return out
