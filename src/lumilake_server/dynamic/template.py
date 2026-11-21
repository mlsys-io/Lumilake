"""Template-driven dynamic rounds: declared round bodies rendered per round.

A template round has a fixed body instead of a planner proposal. The spec
declares ``first`` (round 1), ``next`` (every later round) and the export
whose emptiness stops the loop; ``{{round}}`` renders to the round's
zero-based suffix and ``{{prev}}`` is one less, and this module renders a
body, validates a template at submit, and builds the per-round state the job
loop delegates to.
"""

import copy
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictStr

from lumilake_server.dynamic.driver import (
    DriverProtocolError,
    SubgraphPlan,
    resolve_subgraph,
    validate_emitted_subgraph,
)
from lumilake_server.ops.router_ops import expand_routers
from lumilake_server.runtime.python_step import WHOLE_RESULT_MODES

_ROUND = "{{round}}"
_PREV = "{{prev}}"


class DriverRoundBody(BaseModel):
    """One declared round body of a template-driven dynamic run."""

    model_config = ConfigDict(extra="forbid")

    ops: list[dict[str, Any]]
    export: list[StrictStr] = []


class DriverTemplate(BaseModel):
    """Fixed ``first``/``next`` round bodies and the export that stops the loop."""

    model_config = ConfigDict(extra="forbid")

    first: DriverRoundBody
    next: DriverRoundBody
    stop_when_empty: StrictStr


def _render_string(value: str, suffix: int) -> str:
    """Substitute the placeholders in one string of a template body."""
    return value.replace(_ROUND, str(suffix)).replace(_PREV, str(suffix - 1))


def _render_value(value: Any, suffix: int) -> Any:
    """Render the placeholders in every string a field carries."""
    if isinstance(value, str):
        return _render_string(value, suffix)
    if isinstance(value, list):
        return [_render_value(item, suffix) for item in value]
    if isinstance(value, dict):
        return {key: _render_value(item, suffix) for key, item in value.items()}
    return value


def render_body(body: DriverRoundBody, round_number: int) -> SubgraphPlan:
    """Render a body for the one-based round, whose suffix is one less."""
    if round_number < 1:
        raise DriverProtocolError("template rounds start at 1")
    suffix = round_number - 1
    ops = copy.deepcopy(body.ops)
    for op in ops:
        for key, value in op.items():
            if op.get("op") == "LambdaOp" and key in {"code", "_code"}:
                continue
            op[key] = _render_value(value, suffix)
    template_ids = {
        op["id"]: original["id"]
        for original, op in zip(body.ops, ops)
        if op["id"] != original["id"]
    }
    export = [_render_string(value, suffix) for value in body.export]
    return SubgraphPlan(ops=ops, export=export, template_ids=template_ids)


def plan_for_round(template: DriverTemplate, round_number: int) -> SubgraphPlan:
    """Render ``first`` for round 1 and ``next`` for every later round."""
    body = template.first if round_number == 1 else template.next
    return render_body(body, round_number)


def stop_export(template: DriverTemplate, round_number: int) -> str:
    """The previous round's export id, which gates this round."""
    if round_number < 2:
        raise DriverProtocolError("stop_when_empty applies only after round 1")
    return _render_string(template.stop_when_empty, round_number - 1)


def archive_result_items(decoded: list[Any]) -> list[Any]:
    """Flatten a decoded archive value into the whole result items it holds: a
    whole-result export archives one value whose ``output`` is the whole list
    its step returned, so its items sit one level down from where the stop
    count and a later round's forwarding read them."""
    items: list[Any] = []
    for value in decoded:
        if isinstance(value, list):
            items.extend(value)
        else:
            items.append(value)
    return items


def archived_element_count(items: list[Any]) -> int:
    """Count the rows an archived export holds: a whole-result Lambda archives
    one item whose ``output`` is the list its step returned, so an export that
    returned no rows counts 0."""
    total = 0
    for item in items:
        held = item.get("output") if isinstance(item, dict) else item
        total += len(held) if isinstance(held, list) else 1
    return total


@dataclass(frozen=True)
class TemplateRound:
    """The state one template round hands to, and keeps for, the job loop:
    ``stop_export`` names the archived export whose element count decides
    whether the next round runs, and round 1 has none."""

    round_number: int
    stop_export: str | None
    plan: SubgraphPlan

    def record(self) -> dict[str, Any]:
        """The parent's stored plan record for this round."""
        return {
            "next": "subgraph",
            "ops": self.plan.ops,
            "export": self.plan.export,
            "source": "template",
            "round": self.round_number,
        }

    def stopped(self, results: dict[str, list[Any]]) -> bool:
        """Whether the round before this one exported no rows."""
        if self.stop_export is None:
            return False
        return archived_element_count(results.get(self.stop_export) or []) == 0


def run_round(
    template: DriverTemplate,
    round_number: int,
    node_registry: dict[str, dict[str, Any]],
) -> TemplateRound:
    """Render one round, refusing a stop export no earlier round registered."""
    export_id = None if round_number == 1 else stop_export(template, round_number)
    if export_id is not None and export_id not in node_registry:
        raise DriverProtocolError(
            f"stop export {export_id!r} is not in the node registry"
        )
    return TemplateRound(
        round_number=round_number,
        stop_export=export_id,
        plan=plan_for_round(template, round_number),
    )


def _require_export(plan: SubgraphPlan, export_id: str, round_number: int) -> None:
    if export_id not in plan.export:
        raise DriverProtocolError(
            f"stop_when_empty {export_id!r} must be exported by round {round_number}"
        )


def _require_row_list_export(op: dict[str, Any], export_id: str) -> None:
    """Require the stop export to be a whole-result LambdaOp or a JoinOp, the
    ops whose archived export collapses to one list or one item per row."""
    op_type = op.get("op", "")
    if op_type == "JoinOp":
        return
    if op_type != "LambdaOp":
        raise DriverProtocolError(
            f"stop_when_empty {export_id!r} must be a LambdaOp or JoinOp, got "
            f"{op_type!r}"
        )
    if op.get("mode") not in WHOLE_RESULT_MODES:
        raise DriverProtocolError(
            f"stop_when_empty {export_id!r} must be a list, expand or reduce"
            f" mode LambdaOp or a JoinOp, got mode {op.get('mode')!r}: only"
            " those exports read as an empty list"
        )


def validate_template(
    template: DriverTemplate,
    library: dict[str, dict[str, Any]] | None,
    *,
    max_nodes_per_round: int,
) -> None:
    """Validate a template at submit: render rounds 1-3, validate each against
    the nodes the earlier rounds registered, and require the stop export to be
    a row list that round exported."""
    plans = {number: plan_for_round(template, number) for number in (1, 2, 3)}
    _require_export(plans[1], stop_export(template, 2), 1)
    _require_export(plans[2], stop_export(template, 3), 2)
    registry: dict[str, dict[str, Any]] = {}
    for number in (1, 2, 3):
        plan = plans[number]
        ops = expand_routers(plan.ops)
        validate_emitted_subgraph(
            ops,
            registry,
            max_nodes_per_round,
            library,
            export=plan.export,
            results={op_id: [] for op_id in registry},
        )
        registry.update({op["id"]: op for op in resolve_subgraph(ops, library)})
    for number in (2, 3):
        export_id = stop_export(template, number)
        _require_row_list_export(registry[export_id], export_id)
