"""Whole-op revision plan: which ops a reusing job keeps from the store.

A hit is a whole op whose every record the store already holds; it leaves the
graph, and its readers carry the stored values in their column-plan entries.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lumilake.log import Logger
from lumilake_server.data_specs import (
    PYTHON_TASK_TYPE,
    ColumnPlanEntry,
    DataFrame,
    ListData,
    NodeRead,
    PythonData,
    parse_data_spec,
)
from lumilake_server.ops import RecordKey
from lumilake_server.runtime import python_step
from lumilake_server.runtime.lambda_task_helpers import items_column
from lumilake_server.runtime.record_keys import child_count_key, child_key
from lumilake_server.runtime.record_model_rows import cut_rows, merge_id, pass_id
from lumilake_server.runtime.record_store import (
    EffectVersions,
    RecordAddress,
    RecordEntry,
    RecordRealization,
    RecordSource,
    data_spec_dict,
    digest_of,
    entry_digest,
    is_marker,
    key_text,
    row_key,
)
from lumilake_server.runtime.record_writer import (
    NodeRecords,
    RecordInput,
    RecordPlan,
    _decoded,
    content_input,
    grouped_local_chat,
    input_nodes,
    input_values,
    literal_rows,
    position_inputs,
    read_entry,
    reads_unversioned,
    realization,
    resolve_plans,
    split_inputs,
    typed_row_step,
)
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp
from lumilake_server.utils.job_storage import JobStorage

_SERVED_TASKS = frozenset({"python", "inference", "classification", "embedding", "api"})
"""Task types whose stored value a served op carries whole; a diffusion task archives
its value as an artifact instead. A hosted model call is an ``api``
task, and a fetch is one too, but a fetch reads an unversioned URL and never
reaches this rule."""


def _served_entry(values: list[Any]) -> ColumnPlanEntry:
    """The slot list a partial run reads back: one entry, its own literal, so no
    plan entry clones a stage reference into a data column."""
    return ColumnPlanEntry(kind="literal", role="served", values=values)


def _partially_servable(plan: RecordPlan) -> bool:
    """Whether an op can run on only its missed records and be merged back: a
    keyed one-to-one python step whose rows a merge step can place by key, or a
    keyed model op whose rows a merge step places in live order."""
    if not plan.keyed:
        return False
    if plan.op.task_type != PYTHON_TASK_TYPE:
        return _model_serves_rows(plan)
    spec = plan.op.data_spec
    return isinstance(spec, PythonData) and spec.mode in (None, "row", "parse")


def _model_serves_rows(plan: RecordPlan) -> bool:
    """Whether a keyed model op reads its rows one per record, so a partial run can
    cut those rows to the missed ones: a rowwise chat, or a list-fed model
    op with a literal row list or a node a pass-through narrows."""
    spec = plan.op.data_spec
    if isinstance(spec, DataFrame):
        return True
    if not isinstance(spec, ListData):
        return False
    return isinstance(spec.items, list) or (
        spec.node is not None and spec.items is None
    )


@dataclass(frozen=True)
class OutputShape:
    """How an output node archives its records."""

    path: str | None = None
    """The stored value's own field to archive, when the output names one."""
    whole_list: bool = False
    """An unkeyed list-mode output stores every item as its one record; a hit
    that finds a keyed record or a value that is not a list is not that shape,
    so it runs."""
    whole_run: bool = False
    """A list, reduce or expand emits one value over all its records, whatever
    records the store holds, so a hit rebuilds that one value."""

    def archived(self, values: Sequence[Any]) -> list[str]:
        selected = [_value_at_path(value, self.path) for value in values]
        if self.whole_run:
            return [_output_text(selected)]
        return [_output_text(value) for value in selected]


class OpDecision(BaseModel):
    """One op's reported decision, without the internal planned node."""

    model_config = ConfigDict(extra="forbid")

    op: str
    decision: Literal["hit", "miss"]
    reason: str | None = None
    records: int = 0


class RecordDecision(OpDecision):
    """What a reusing job decided for one op, and the first reason it missed."""

    node: str
    """The planned node, so a batch can attribute the decision to the members
    that own it."""


class ContentRecords(BaseModel):
    """One unversioned read this plan re-ran, as its consumers read it."""

    model_config = ConfigDict(extra="forbid")

    keyed: bool = False
    values: list[Any] = Field(default_factory=list)
    keys: list[str | None] = Field(default_factory=list)
    items: list[Any] = Field(default_factory=list)
    """One entry per item the read's task returned, as its readers take them."""

    def value_at(self, position: int) -> Any:
        return self.values[position] if self.keyed else self.values

    @property
    def lines_up(self) -> bool:
        """Whether the read hands one item per record, so its rows and the items
        its readers take stand at the same positions; one item for several
        records leaves them unkeyed."""
        return len(self.items) == len(self.values)


class GroupSlot(BaseModel):
    """One live group of a partially served reduce: its key, its live member
    positions, and the output the store holds for it, or ``None`` to compute."""

    model_config = ConfigDict(extra="forbid")

    key: str
    members: list[int] = Field(default_factory=list)
    served: Any = None


class ParentSlot(BaseModel):
    """One live parent of a partially served expand: its live position, the
    store's count entry for it, and the children the store holds for it, or
    ``None`` to re-expand."""

    model_config = ConfigDict(extra="forbid")

    parent: int
    children: list[Any] | None = None
    count: RecordEntry | None = Field(default=None, exclude=True)
    """The count entry the run indexes for this parent; the step never reads it."""


class PartialRecords(BaseModel):
    """A partially served op: every live record slot, the ones the store
    answered, and the positions the dispatched op must compute."""

    model_config = ConfigDict(extra="forbid")

    keys: list[str | None] = Field(default_factory=list)
    """Every live record's key, in the order the op emits them."""
    served: list[Any] = Field(default_factory=list)
    """Each slot's stored value, or the missed marker for one it must compute."""
    missed: list[int] = Field(default_factory=list)
    """The live positions the dispatched op computes, for the plan's record count."""
    groups: list[GroupSlot] | None = None
    """A reduce's live groups in group order; ``None`` for an op of records."""
    children: list[ParentSlot] | None = None
    """An expand's live parents in parent order, each with the children the
    store holds for it, or ``None`` for a parent this run re-expands."""
    stored: list[RecordEntry] = Field(default_factory=list)
    """The entries the store answered for this op, so the run indexes them."""


class ReusePlan(BaseModel):
    """One reusing job's plan: what the store answers and every decision."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[RecordDecision] = Field(default_factory=list)
    hits: dict[str, list[RecordEntry]] = Field(default_factory=dict)
    """Node id -> the trusted entries that answer its records, in key order."""
    outputs: dict[str, str] = Field(default_factory=dict)
    """Node id -> its output name, so a fully answered request rebuilds every
    hit output without the graph it planned from."""
    key_paths: dict[str, str] = Field(default_factory=dict)
    """Node id -> the record-key path it plans with, so a dispatched op keeps
    the key it inherited from a served input even after that input left the
    graph."""
    content: dict[str, ContentRecords] = Field(default_factory=dict)
    """Node id -> the records of an unversioned read the plan re-ran. Such a
    read never hits; its consumers name its values by value."""
    content_served: set[str] = Field(default_factory=set)
    """The re-ran reads whose every reader takes their values, so the read
    leaves the dispatched graph. A read another reader needs stays in it."""
    output_shapes: dict[str, OutputShape] = Field(default_factory=dict)
    """Node id -> how its archived output reads its stored values, for every
    output node the plan serves."""
    partial: dict[str, PartialRecords] = Field(default_factory=dict)
    """Node id -> the records a partially served op keeps and the ones it runs."""
    aliases: dict[str, str] = Field(default_factory=dict)
    """Merge step id -> the partially served op whose records it rebuilds, so a reader's
    input identity is the op's, as in a full run."""
    pass_throughs: dict[str, str] = Field(default_factory=dict)
    """Pass-through step id -> the served producer whose rows it carries, so a reader
    names that producer's records and the writer never stores the step."""

    def served_nodes(self) -> set[str]:
        return {*self.hits, *self.content_served}


@dataclass(frozen=True)
class _Input:
    """One input's writer identity, and the record key at each of its rows."""

    named: RecordInput
    keys: list[str | None]


@dataclass(frozen=True)
class _Decision:
    """One op's outcome: a reason it misses whole, its entries when it hits, or
    the earlier records a partially served op keeps."""

    reason: str | None = None
    entries: list[RecordEntry] | None = None
    partial: PartialRecords | None = None


class ReusePlanner:
    """Decides, before dispatch, which ops of a reusing job the store answers."""

    def __init__(self, *, storage: JobStorage, logger: Logger) -> None:
        self._storage = storage
        self.logger = logger

    def plan(
        self,
        *,
        runtime_graph: RuntimeGraph,
        round_number: int,
        job_id: str,
        content_values: Mapping[str, NodeRecords] | None = None,
    ) -> tuple[ReusePlan, RuntimeGraph]:
        """Decide every op in topological order and return the graph to dispatch."""
        decision = ReusePlan(outputs=runtime_graph.output_node_map.copy())
        misses: set[str] = set()
        order = runtime_graph.topological_order()
        stored: dict[str, list[RecordEntry]] = {}
        plans = resolve_plans(runtime_graph, order)
        content = {} if content_values is None else content_values
        content_keys = set(content)
        for node_id in runtime_graph.topological_order(_seed_edges(runtime_graph)):
            plan = plans[node_id]
            if plan.key_path is not None:
                decision.key_paths[node_id] = plan.key_path
            if node_id in content_keys:
                self._serve_content(decision, plan, content[node_id], runtime_graph)
                continue
            inputs = _planner_inputs(plan, decision, plans, content)
            expanded = self._expand_outcome(
                plan, decision, inputs, stored, round_number=round_number
            )
            if expanded is not None:
                children, missed_parents, counts = expanded
                if missed_parents:
                    missed = set(missed_parents)
                    decision.partial[node_id] = PartialRecords(
                        children=[
                            ParentSlot(
                                parent=position,
                                children=(
                                    None
                                    if position in missed
                                    else [entry.value for entry in children[position]]
                                ),
                                count=counts.get(position),
                            )
                            for position in range(len(children))
                        ],
                        stored=[entry for group in children for entry in group],
                    )
                    decision.decisions.append(
                        RecordDecision(
                            op=plan.op_name,
                            node=node_id,
                            decision="miss",
                            reason="partial children",
                            records=len(missed_parents),
                        )
                    )
                    misses.add(node_id)
                    continue
                answered_children = [entry for group in children for entry in group]
                decision.decisions.append(
                    RecordDecision(
                        op=plan.op_name,
                        node=node_id,
                        decision="hit",
                        records=len(answered_children),
                    )
                )
                if not answered_children:
                    decision.decisions[-1] = decision.decisions[-1].model_copy(
                        update={"decision": "miss", "reason": "no children"}
                    )
                    misses.add(node_id)
                    continue
                if not self._accept_hit(
                    decision, plan, node_id, answered_children, runtime_graph
                ):
                    misses.add(node_id)
                    continue
                decision.hits[node_id] = answered_children
                stored[node_id] = answered_children
                continue
            grouped = self._reduce_outcome(
                plan,
                inputs,
                _reduce_rows(plan, decision, stored),
                round_number=round_number,
            )
            if grouped is not None:
                group_reason, slots, answered = grouped
                hit_groups = [slot.key for slot in slots if slot.served is not None]
                missed_groups = [slot.key for slot in slots if slot.served is None]
                if group_reason != "partial group records":
                    decision.decisions.append(
                        RecordDecision(
                            op=plan.op_name,
                            node=node_id,
                            decision="miss",
                            reason=group_reason,
                        )
                    )
                    misses.add(node_id)
                    continue
                if missed_groups:
                    decision.partial[node_id] = PartialRecords(
                        groups=slots,
                        stored=[answered[key] for key in hit_groups],
                    )
                    decision.decisions.append(
                        RecordDecision(
                            op=plan.op_name,
                            node=node_id,
                            decision="miss",
                            reason=group_reason,
                            records=len(missed_groups),
                        )
                    )
                    misses.add(node_id)
                    continue
                answered_hit = [answered[key] for key in hit_groups]
                decision.decisions.append(
                    RecordDecision(
                        op=plan.op_name,
                        node=node_id,
                        decision="hit",
                        records=len(answered_hit),
                    )
                )
                if not self._accept_hit(
                    decision, plan, node_id, answered_hit, runtime_graph
                ):
                    misses.add(node_id)
                    continue
                decision.hits[node_id] = answered_hit
                stored[node_id] = answered_hit
                continue
            records = _record_keys(plan, inputs)
            outcome = self._decide(
                plan,
                inputs,
                records,
                round_number=round_number,
                job_id=job_id,
                misses=misses,
                decision=decision,
                stored=stored,
                runtime_graph=runtime_graph,
                node_id=node_id,
            )
            reason, hit = outcome.reason, outcome.entries
            if outcome.partial is not None:
                merge_step = _merge_step(plan.op.task_type, outcome.partial)
                carries = not merge_step or node_id not in runtime_graph.output_node_map
                if carries:
                    decision.partial[node_id] = outcome.partial
                decision.decisions.append(
                    RecordDecision(
                        op=plan.op_name,
                        node=node_id,
                        decision="miss",
                        reason="partial records" if carries else "value not servable",
                        records=len(outcome.partial.missed),
                    )
                )
                misses.add(node_id)
                continue
            decision.decisions.append(
                RecordDecision(
                    op=plan.op_name,
                    node=node_id,
                    decision="hit" if hit is not None else "miss",
                    reason=reason,
                    records=(
                        len(hit) if hit is not None else _counted(plan, records, inputs)
                    ),
                )
            )
            if hit is None:
                misses.add(node_id)
                continue
            if not self._accept_hit(decision, plan, node_id, hit, runtime_graph):
                misses.add(node_id)
                continue
            decision.hits[node_id] = hit
            stored[node_id] = hit
        _keep_seed_nodes_dispatched(runtime_graph, decision)
        self.logger.info(
            "Record reuse plan for job %s: %d hit op(s), %d miss op(s)",
            job_id,
            len(decision.hits),
            len(decision.decisions) - len(decision.hits),
        )
        return decision, _without_hits(runtime_graph, decision)

    def _accept_hit(
        self,
        decision: ReusePlan,
        plan: RecordPlan,
        node_id: str,
        hit: Sequence[RecordEntry],
        runtime_graph: RuntimeGraph,
    ) -> bool:
        """Whether a hit may leave the graph: every reader takes the stored values
        and an output archives them; else the decision becomes a miss."""
        unserved = _unserved_reader_reason(runtime_graph, decision, plan.op, hit)
        if unserved is None and node_id in runtime_graph.output_node_map:
            shape = output_shape(plan, runtime_graph.output_paths.get(node_id))
            if shape is None or output_runs(shape, hit):
                unserved = "value not servable"
            else:
                decision.output_shapes[node_id] = shape
        if unserved is None:
            return True
        decision.decisions[-1] = decision.decisions[-1].model_copy(
            update={"decision": "miss", "reason": unserved}
        )
        return False

    def _serve_content(
        self,
        decision: ReusePlan,
        plan: RecordPlan,
        records: NodeRecords,
        runtime_graph: RuntimeGraph,
    ) -> None:
        """Record one unversioned read this plan re-ran."""
        decision.decisions.append(
            RecordDecision(
                op=plan.op_name,
                node=plan.op.node_id,
                decision="miss",
                reason="unversioned upstream",
                records=len(records.values) if records.keyed else 1,
            )
        )
        decision.content[plan.op.node_id] = ContentRecords(
            keyed=records.keyed,
            values=records.values.copy(),
            keys=(
                [
                    _key_at(records.keys, position)
                    for position in range(len(records.values))
                ]
                if records.keyed
                else []
            ),
            items=records.items.copy(),
        )
        if _content_servable(runtime_graph, decision, plan.op.node_id):
            decision.content_served.add(plan.op.node_id)

    def _decide_children(
        self,
        plan: RecordPlan,
        parent_keys: Sequence[str],
        inputs: Sequence[_Input],
        *,
        round_number: int,
    ) -> tuple[list[list[RecordEntry]], list[int], dict[int, RecordEntry]]:
        """Each live parent's stored children, in child order, the parent
        positions the store does not answer, and each answered parent's count
        entry.

        A child's key is its parent key and its index, so each is probed by its
        own address: a parent is served when its own count entry answers and
        every child the count names answers, so a parent that expands to nothing
        is answered too."""
        named = [item.named for item in inputs]
        served: list[list[RecordEntry]] = [[] for _ in parent_keys]
        missed: list[int] = []
        counts: dict[int, RecordEntry] = {}
        for position, parent_key in enumerate(parent_keys):
            read_digests, read_values = position_inputs(named, position, keyed=True)
            count = read_entry(
                self._storage,
                entry_digest(
                    spec_digest=plan.spec_digest,
                    address=RecordAddress(
                        op=plan.op_name,
                        record=child_count_key(parent_key),
                        round=round_number,
                    ),
                    input_digests=read_digests,
                    input_values=read_values,
                    realization=RecordRealization(),
                    versions=EffectVersions(),
                ),
            )
            children: list[RecordEntry] = []
            if (
                count is not None
                and _entry_miss_reason(count) is None
                and isinstance(count.value, int)
            ):
                for index in range(count.value):
                    child = read_entry(
                        self._storage,
                        entry_digest(
                            spec_digest=plan.spec_digest,
                            address=RecordAddress(
                                op=plan.op_name,
                                record=child_key(parent_key, index),
                                round=round_number,
                            ),
                            input_digests=read_digests,
                            input_values=read_values,
                            realization=RecordRealization(),
                            versions=EffectVersions(),
                        ),
                    )
                    if child is None or _entry_miss_reason(child) is not None:
                        break
                    children.append(child)
            if count is None or len(children) != count.value:
                missed.append(position)
                continue
            served[position] = children
            counts[position] = count
        return served, missed, counts

    def _expand_outcome(
        self,
        plan: RecordPlan,
        decision: ReusePlan,
        inputs: Sequence[_Input],
        stored: Mapping[str, Sequence[RecordEntry]],
        *,
        round_number: int,
    ) -> tuple[list[list[RecordEntry]], list[int], dict[int, RecordEntry]] | None:
        """An expand's ``(served children per parent, missed parents, answered
        counts by parent)``, or ``None`` when it is not an expand or its parent
        runs in this batch."""
        spec = plan.op.data_spec
        if not isinstance(spec, PythonData) or spec.mode != "expand":
            return None
        parent_keys = _expand_parent_keys(plan, stored)
        if parent_keys is None:
            parent_keys = _content_parent_keys(plan, decision)
        if parent_keys is None:
            return None
        return self._decide_children(
            plan, parent_keys, inputs, round_number=round_number
        )

    def _reduce_outcome(
        self,
        plan: RecordPlan,
        inputs: Sequence[_Input],
        rows: Sequence[Any] | None,
        *,
        round_number: int,
    ) -> tuple[str, list[GroupSlot], dict[str, RecordEntry]] | None:
        """A reduce's ``(reason, live groups, answered entries)`` or ``None`` when
        it is not a reduce; all groups hitting serves the op, the rest dispatch it."""
        spec = plan.op.data_spec
        if not isinstance(spec, PythonData) or spec.mode != "reduce":
            return None
        if rows is None:
            # The input is dispatched in this batch, so its groups are not known yet.
            return "input dispatched", [], {}
        decided = self._decide_groups(plan, inputs, rows, round_number=round_number)
        if isinstance(decided, str):
            return decided, [], {}
        slots, answered = decided
        return "partial group records", slots, answered

    def _decide_groups(
        self,
        plan: RecordPlan,
        inputs: Sequence[_Input],
        rows: Sequence[Any] | None,
        *,
        round_number: int,
    ) -> tuple[list[GroupSlot], dict[str, RecordEntry]] | str:
        """A reduce's live groups in group order, each holding the output the
        store answers it with, and the entries that answered, by group key."""
        groups = _group_members(plan, rows)
        if groups is None:
            return "unkeyed row"
        slots: list[GroupSlot] = []
        entries: dict[str, RecordEntry] = {}
        for group_key, members in groups:
            input_digests, input_values = _group_inputs(inputs, members)
            entry = read_entry(
                self._storage,
                entry_digest(
                    spec_digest=plan.spec_digest,
                    address=RecordAddress(
                        op=plan.op_name, record=group_key, round=round_number
                    ),
                    input_digests=input_digests,
                    input_values=input_values,
                    realization=RecordRealization(),
                    versions=EffectVersions(),
                ),
            )
            answered = _entry_miss_reason(entry) is None
            if answered and entry is not None:
                entries[group_key] = entry
            slots.append(
                GroupSlot(
                    key=group_key,
                    members=members,
                    served=entry.value if answered and entry is not None else None,
                )
            )
        return slots, entries

    def _decide(
        self,
        plan: RecordPlan,
        inputs: Sequence[_Input],
        records: list[str | None] | None,
        *,
        round_number: int,
        job_id: str,
        misses: set[str],
        decision: ReusePlan,
        stored: Mapping[str, list[RecordEntry]],
        runtime_graph: RuntimeGraph,
        node_id: str,
    ) -> _Decision:
        """One op's outcome: a miss reason, its entries, or its kept records."""
        if reads_unversioned(plan.op):
            return _Decision("unversioned upstream")
        if plan.op.task_type not in _SERVED_TASKS:
            return _Decision("value not servable")
        if any(node in misses for node in input_nodes(plan)):
            return _Decision("upstream miss")
        if records is None:
            return _Decision("unkeyed row")
        named = [item.named for item in inputs]
        entries: list[RecordEntry] = []
        source = RecordSource(job=job_id, round=round_number, task_id="")
        seeded = _seeded_records(plan, stored, runtime_graph)
        served = _partially_servable(plan)
        first_reason: str | None = None
        slots: list[Any] = []
        missed: list[int] = []
        for position, record in enumerate(records):
            draw = _planned_realization(plan, records, position, source, seeded)
            if draw is None:
                return _Decision("no realization")
            input_digests, input_value_digests = position_inputs(
                named, position, keyed=plan.keyed
            )
            entry = read_entry(
                self._storage,
                entry_digest(
                    spec_digest=plan.spec_digest,
                    address=RecordAddress(
                        op=plan.op_name, record=record, round=round_number
                    ),
                    input_digests=input_digests,
                    input_values=input_value_digests,
                    realization=draw,
                    versions=EffectVersions(),
                ),
            )
            reason = _entry_miss_reason(entry)
            found = None if reason is not None else entry
            if found is not None:
                entries.append(found)
                slots.append(found.value)
                continue
            if not served:
                return _Decision(reason)
            if first_reason is None:
                first_reason = reason
            missed.append(position)
            slots.append(python_step.MISSED)
        if not missed:
            return _Decision(entries=entries)
        if len(missed) == len(records):
            # No record of this op survives, so it re-runs whole.
            return _Decision(first_reason)
        return _Decision(
            partial=PartialRecords(keys=list(records), served=slots, missed=missed)
        )


def _reduce_rows(
    plan: RecordPlan,
    decision: ReusePlan,
    stored: Mapping[str, Sequence[RecordEntry]],
) -> list[Any] | None:
    """The rows a reduce groups: its one keyed input's records, served or
    stored, in the order the worker zips them; ``None`` when no input is known."""
    sources = [node for node in input_nodes(plan) if node in stored]
    if sources:
        rows = [entry.value for entry in stored[sources[0]]]
        return rows if rows else None
    # A reduce whose input runs in this batch groups that input's own rows, which its
    # spec read names; those group keys are not known until that input finishes.
    return None


def _expand_parent_keys(
    plan: RecordPlan,
    stored: Mapping[str, Sequence[RecordEntry]],
) -> list[str] | None:
    """A live parent key per row an expand reads, from its served parent's keys;
    ``None`` when that parent runs in this batch, whole or in part, since a
    parent this run computes has no recorded identity to probe its children at."""
    sources = [node for node in input_nodes(plan) if node in stored]
    if not sources:
        return None
    source = sources[0]
    keys = [entry.address.record for entry in stored[source]]
    if not keys or any(key is None for key in keys):
        return None
    return [str(key) for key in keys]


def _content_parent_keys(plan: RecordPlan, decision: ReusePlan) -> list[str] | None:
    """A live parent key per row an expand reads, from an unversioned read the
    plan re-ran and its rows read keyed; ``None`` when that read is not its one
    keyed input."""
    keyed = [
        node
        for node in input_nodes(plan)
        if node in decision.content and decision.content[node].keyed
    ]
    if len(keyed) != 1 or not decision.content[keyed[0]].lines_up:
        return None
    keys = decision.content[keyed[0]].keys
    if not keys or any(key is None for key in keys):
        return None
    return [str(key) for key in keys]


def rebuilt_output(plan: ReusePlan, node_id: str) -> list[str]:
    """A hit output's archived values, shaped the way its output node emits them."""
    entries = plan.hits[node_id]
    shape = plan.output_shapes.get(node_id, OutputShape())
    if entries[0].address.record is not None:
        return shape.archived([entry.value for entry in entries])
    value = entries[0].value
    if not isinstance(value, list):
        return shape.archived([value])
    return shape.archived(value)


def _output_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def served_records(plan: ReusePlan) -> dict[str, NodeRecords]:
    """The records every hit serves, shaped as the writer reads a finished node,
    plus the groups a partial reduce and the children and child counts a partial
    expand already holds, so the run indexes them."""
    served: dict[str, NodeRecords] = {}
    for node_id, entries in plan.hits.items():
        keyed = entries[0].address.record is not None
        records = NodeRecords(
            keyed=keyed,
            values=(
                [entry.value for entry in entries] if keyed else entries[0].value.copy()
            ),
            keys=[entry.address.record for entry in entries],
            entries=entries.copy(),
        )
        for entry in entries:
            records.digests[entry.address.record, entry.address.round] = (
                entry.digest,
                entry.value_digest,
            )
        served[node_id] = records
    for node_id, partial in plan.partial.items():
        entries = partial.stored.copy()
        if partial.children is not None:
            entries = [
                slot.count for slot in partial.children if slot.count is not None
            ] + entries
        if not entries:
            continue
        records = NodeRecords(
            keyed=True,
            values=[entry.value for entry in entries],
            keys=[entry.address.record for entry in entries],
            entries=entries,
        )
        served[node_id] = records
    return served


def writer_records(
    plan: ReusePlan, content_values: Mapping[str, NodeRecords]
) -> dict[str, NodeRecords]:
    """The records a reusing job hands its writer: re-ran reads and every hit."""
    return {**content_values, **served_records(plan)}


_PROMPT_TASKS = frozenset({"inference"})
"""Task types whose output items carry the per-row prompt a history rebuilds
from; a stored value holds the output text alone."""


def output_shape(plan: RecordPlan, path: str | None) -> OutputShape | None:
    """How a hit output archives its records; ``None`` if no stored value holds it."""
    if plan.op.task_type in _PROMPT_TASKS:
        return None
    if path == "items":
        # An ``items`` output archives each whole item, envelope and all, which no
        # stored value holds.
        return None
    spec = plan.op.data_spec
    whole_list = isinstance(spec, PythonData) and spec.mode == "list" and not plan.keyed
    whole_run = (
        isinstance(spec, PythonData) and spec.mode in python_step.WHOLE_RESULT_MODES
    )
    if path is None or path == "":
        return OutputShape(whole_list=whole_list, whole_run=whole_run)
    field = _stored_path(path)
    if field is None:
        return None
    return OutputShape(path=field, whole_list=whole_list, whole_run=whole_run)


def output_runs(shape: OutputShape, entries: Sequence[RecordEntry]) -> bool:
    """Whether a hit output runs: it reads a field or list no stored value holds."""
    if shape.whole_list:
        return entries[0].address.record is not None or not isinstance(
            entries[0].value, list
        )
    return any(
        _value_at_path(value, shape.path) is _UNSET for value in _hit_values(entries)
    )


def _merge_step(task_type: str, partial: PartialRecords) -> bool:
    """Whether a merge step rebuilds a partially served op. Only a model op of
    records gets one: a python step is rewritten in place and keeps its own id and
    output path, and a grouped op emits in one step."""
    return (
        partial.groups is None
        and partial.children is None
        and task_type != PYTHON_TASK_TYPE
    )


def _rewritten(
    op: RuntimeOp, plan: ReusePlan, runtime_graph: RuntimeGraph
) -> RuntimeOp:
    """An op reading its served inputs' values, each entry gaining values; a
    partially served op keeps every column at its live width and gains a slot
    list that tells its step which records it computes."""
    partial = plan.partial.get(op.node_id)
    if partial is not None:
        if op.task_type != PYTHON_TASK_TYPE:
            return _narrowed_model(op, partial)
        return _rewritten_slots(op, partial, plan, runtime_graph)
    served = [node for node in op.dependencies if _served_node(plan, node)]
    if not served:
        return op
    if op.task_type != PYTHON_TASK_TYPE:
        # A model op reads its served producer's rows through a pass-through step, which
        # carries them, so its spec, and its digest, stay a full run's and its address
        # is the one a full run stores.
        return op
    spec = op.data_spec
    if isinstance(spec, PythonData) and spec.plan:
        return replace(op, data_spec=_rewritten_plan(op, spec, plan, runtime_graph))
    raise ValueError(f"op '{op.node_id}' cannot read a served record")


def _rewritten_plan(
    op: RuntimeOp,
    spec: PythonData,
    plan: ReusePlan,
    runtime_graph: RuntimeGraph,
) -> PythonData:
    """A python step whose plan entries read the values of served inputs."""
    rewritten: list[ColumnPlanEntry] = []
    for entry in spec.plan:
        node = entry.node
        if node is None or not _served_node(plan, node):
            rewritten.append(entry)
            continue
        column = _plan_column(plan, node, entry, runtime_graph)
        if column is None:
            raise ValueError(f"op '{op.node_id}' cannot rebuild a served column")
        rewritten.append(entry.model_copy(update={"values": column}))
    return spec.model_copy(update={"plan": rewritten})


def _rewritten_slots(
    op: RuntimeOp,
    partial: PartialRecords,
    plan: ReusePlan,
    runtime_graph: RuntimeGraph,
) -> RuntimeOp:
    """A partially served op as its step reads it: every column keeps its live
    width, so a literal or served column names each record by the position the
    writer keys it at, and the slot list tells the step what it computes and in
    what order to emit it."""
    spec = op.data_spec
    if not isinstance(spec, PythonData) or not spec.plan:
        raise ValueError(f"op '{op.node_id}' cannot rewrite a partial run")
    rewritten = _rewritten_plan(op, spec, plan, runtime_graph)
    slots = _partial_slots(partial)
    return replace(
        op, data_spec=rewritten.model_copy(update={"plan": [*rewritten.plan, slots]})
    )


def _partial_slots(partial: PartialRecords) -> ColumnPlanEntry:
    """The served slot entry for a partial op: one per group or child, else one
    record per live position."""
    if partial.groups is not None:
        return _served_entry(
            [slot.model_dump(exclude_none=True) for slot in partial.groups]
        )
    if partial.children is not None:
        return _served_entry(
            [slot.model_dump(exclude_none=True) for slot in partial.children]
        )
    return _served_entry(
        [[key, value] for key, value in zip(partial.keys, partial.served)]
    )


def _narrowed_model(op: RuntimeOp, partial: PartialRecords) -> RuntimeOp:
    """A partially served model op reading only its missed rows: each literal rows
    list is cut to the missed positions, so it reads exactly the rows this
    run computes. A node reference keeps its whole list, and a pass-through step
    narrows the producer instead."""
    spec = cut_rows(data_spec_dict(op), partial.missed)
    return replace(op, data_spec=parse_data_spec(op.task_type, spec))


def _list_refs(value: Any) -> list[Mapping[str, Any]] | None:
    """Every node reference a data spec makes, or ``None`` when one carries an
    expression, which no stored value can fill in place."""
    refs: list[Mapping[str, Any]] = []

    def walk(item: Any) -> bool:
        if isinstance(item, Mapping):
            node = item.get("node")
            if isinstance(node, str) and "path" in item:
                if item.get("expr") is not None:
                    return False
                refs.append(item)
                return True
            return all(walk(nested) for nested in item.values())
        if isinstance(item, list):
            return all(walk(nested) for nested in item)
        return True

    return refs if walk(value) else None


_OUTPUT_ENVELOPES = ("value.items.output", "items.output")
_VALUE_ENVELOPES = ("value.items.value", "items.value")
_ENVELOPES = (*_OUTPUT_ENVELOPES, *_VALUE_ENVELOPES)


def _model_entry_values(
    entries: Sequence[RecordEntry], path: Any, typed: bool
) -> list[Any]:
    """Every value a model op reads from a hit at ``path``, in record order."""
    field = _stored_path(path)
    if field is None:
        return [_UNSET]
    values = [
        _value_at_path(value, field, decode=False) for value in _hit_values(entries)
    ]
    whole_output = typed and field == "" and path in _OUTPUT_ENVELOPES
    if not whole_output:
        return values
    return [_UNSET if v is _UNSET else _output_text(v) for v in values]


def _stored_path(path: Any) -> str | None:
    """The field an envelope path selects inside a stored value, or ``None``."""
    if not isinstance(path, str) or not path:
        return None
    for prefix in _ENVELOPES:
        if path == prefix:
            return ""
        if path.startswith(prefix + "."):
            return path[len(prefix) + 1 :]
    return None


def _served_node(plan: ReusePlan, node: Any) -> bool:
    """Whether one of the plan's served nodes answers ``node``."""
    return isinstance(node, str) and (node in plan.hits or node in plan.content_served)


def _content_servable(
    runtime_graph: RuntimeGraph, plan: ReusePlan, node_id: str
) -> bool:
    """Whether every reader of a re-ran read takes its values."""
    readers = [
        reader
        for reader in runtime_graph.nodes.values()
        if node_id in reader.dependencies
    ]
    if not readers:
        return True
    return all(_reader_takes_content(plan, node_id, reader) for reader in readers)


def _reader_takes_content(plan: ReusePlan, node_id: str, reader: RuntimeOp) -> bool:
    """Whether one reader can read a re-ran read's values in place of its node."""
    if reader.task_type != PYTHON_TASK_TYPE:
        return False
    spec = reader.data_spec
    if not isinstance(spec, PythonData):
        return False
    if not spec.plan:
        return False
    for entry in spec.plan:
        if entry.node != node_id:
            continue
        column = _content_column(plan, node_id, entry)
        if not column or any(value is _UNSET for value in column):
            return False
    return True


def _content_column(
    plan: ReusePlan, node: str, entry: ColumnPlanEntry
) -> list[Any] | None:
    """The column a python reader takes from a re-ran read: its items, which a
    full run's task hands it one cell per, not one cell per stored record."""
    content = plan.content[node]
    try:
        return items_column(
            entry.model_dump(exclude_none=True, exclude={"values"}),
            content.items,
            _as_is,
            "served",
        )
    except ValueError:
        return None


def _hit_values(entries: Sequence[RecordEntry]) -> list[Any]:
    """Every stored value one hit answers with, one per record or per item."""
    if entries[0].address.record is not None:
        return [entry.value for entry in entries]
    value = entries[0].value
    return value if isinstance(value, list) else [value]


_WRAPPED_TASKS = frozenset({"python", "inference"})
"""Task types whose items carry their value under ``output``, which the writer
stores unwrapped."""


type _Wrapping = Literal["output", "text"]


def _wrap(op: RuntimeOp) -> _Wrapping | None:
    """``text`` for a row step's item, ``output`` for a bare item, else ``None``."""
    if typed_row_step(op):
        return "text"
    return "output" if op.task_type in _WRAPPED_TASKS else None


def _plan_column(
    plan: ReusePlan,
    node: str,
    entry: ColumnPlanEntry,
    runtime_graph: RuntimeGraph,
) -> list[Any] | None:
    """An answered input's column for a python step, built as the worker builds it."""
    if node in plan.content:
        return _content_column(plan, node, entry)
    producer = runtime_graph.nodes[node]
    return served_column(
        plan.hits[node],
        entry,
        _wrap(producer),
        grouped_local_chat(producer),
    )


def served_lists(entries: Sequence[RecordEntry], grouped: bool) -> list[list[Any]]:
    """A hit producer's item outputs: one list per keyed row, one whole-op value,
    or every keyed row folded into the one grouped list a local chat emits."""
    if entries[0].address.record is None:
        return [[output] for output in entries[0].value]
    if grouped:
        return [[[entry.value for entry in entries]]]
    return [[entry.value] for entry in entries]


def served_column(
    entries: Sequence[RecordEntry],
    entry: ColumnPlanEntry,
    wrapping: _Wrapping | None,
    grouped: bool,
) -> list[Any] | None:
    """The column a python step reads from a hit producer, built as its item shape."""
    if entry.grouped:
        return None
    items: list[Any] = []
    for values in served_lists(entries, grouped):
        for item in values:
            if wrapping == "text":
                items.append({"output": _output_text(item), "value": item})
            elif wrapping == "output":
                items.append({"output": item})
            else:
                items.append(item)
    try:
        return items_column(
            entry.model_dump(exclude_none=True, exclude={"values"}),
            items,
            _as_is,
            "served",
        )
    except ValueError:
        return None


def _as_is(value: Any) -> Any:
    return value


def _key_at(keys: Sequence[str | None] | None, position: int) -> str | None:
    """The key of the record at ``position``, or ``None`` when it has none."""
    if keys is None or position >= len(keys):
        return None
    return keys[position]


_UNSET = object()


def _value_at_path(value: Any, path: Any, decode: bool = True) -> Any:
    """One stored value selected at ``path``, or ``_UNSET`` when a part is missing."""
    if not isinstance(path, str) or not path:
        return value
    current = value
    for part in path.split("."):
        if isinstance(current, str):
            if not decode:
                return _UNSET
            try:
                current = json.loads(current)
            except ValueError:
                return _UNSET
        if not isinstance(current, dict) or part not in current:
            return _UNSET
        current = current[part]
    return current


def _keep_seed_nodes_dispatched(runtime_graph: RuntimeGraph, plan: ReusePlan) -> None:
    """Demote a seed node a dispatched reader still reads, so the graph holds it."""
    placed = {d.node: index for index, d in enumerate(plan.decisions)}
    readers = _dispatched_seed_readers(runtime_graph, plan)
    for node_id, reader in readers.items():
        name = runtime_graph.op_names.get(reader, reader)
        decision = plan.decisions[placed[node_id]]
        reason = f"read by dispatched '{name}'"
        update = {"decision": "miss", "reason": reason}
        plan.decisions[placed[node_id]] = decision.model_copy(update=update)
        del plan.hits[node_id]


def _dispatched_seed_readers(
    runtime_graph: RuntimeGraph, plan: ReusePlan
) -> dict[str, str]:
    """Each hit seed node read by a dispatched op, mapped to that reader's id."""
    readers: dict[str, str] = {}
    for reader in runtime_graph.nodes.values():
        if reader.node_id in plan.hits:
            continue
        spec = reader.data_spec
        if isinstance(spec, DataFrame) and isinstance(spec.seeds, NodeRead):
            readers.setdefault(spec.seeds.node, reader.node_id)
    return {node_id: readers[node_id] for node_id in plan.hits if node_id in readers}


def _without_hits(runtime_graph: RuntimeGraph, plan: ReusePlan) -> RuntimeGraph:
    """The graph to dispatch, with every served node reading its value instead
    and every partial op carrying the slots its run has to fill."""
    nodes: dict[str, RuntimeOp] = {}
    for node_id in runtime_graph.node_order:
        if node_id in plan.served_nodes():
            continue
        nodes[node_id] = _rewritten(runtime_graph.nodes[node_id], plan, runtime_graph)
    output_node_map = runtime_graph.output_node_map.copy()
    before, after = _plumbing_nodes(plan, nodes, output_node_map, runtime_graph)
    op_names = runtime_graph.op_names.copy()
    for node_id in runtime_graph.node_order:
        if node_id not in nodes:
            continue
        for step_id in before.get(node_id, ()):
            source = plan.pass_throughs[step_id]
            op_names[step_id] = runtime_graph.op_names.get(source, source)
        for merge in after.get(node_id, ()):
            op_names[merge] = runtime_graph.op_names.get(node_id, node_id)
    order: list[str] = []
    for node_id in runtime_graph.node_order:
        if node_id not in nodes:
            continue
        order.extend(node for node in before.get(node_id, ()) if node in nodes)
        order.append(node_id)
        order.extend(node for node in after.get(node_id, ()) if node in nodes)
    return RuntimeGraph(
        nodes=nodes,
        node_order=order,
        output_node_map={
            node_id: name
            for node_id, name in output_node_map.items()
            if node_id in nodes
        },
        output_paths={
            node_id: path
            for node_id, path in runtime_graph.output_paths.items()
            if node_id in nodes
        },
        dsl_to_runtime=runtime_graph.dsl_to_runtime.copy(),
        record_keys=_dispatch_record_keys(runtime_graph, plan, nodes),
        op_names=op_names,
    )


def _plumbing_nodes(
    plan: ReusePlan,
    nodes: dict[str, RuntimeOp],
    output_node_map: dict[str, str],
    runtime_graph: RuntimeGraph,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Each partial model op's pass-through steps to place before it and its merge
    step to place after it, rewiring readers and the output mapping."""
    before: dict[str, list[str]] = {}
    after: dict[str, list[str]] = {}
    for node_id, partial in plan.partial.items():
        op = nodes.get(node_id)
        if op is None or not _merge_step(op.task_type, partial):
            continue
        rewired = op
        for source in [node for node in op.dependencies if node in plan.hits]:
            step_id = _pass_through(
                nodes, plan, runtime_graph, node_id, source, partial
            )
            before.setdefault(node_id, []).append(step_id)
            rewired = _reference(source, step_id, rewired)
        nodes[node_id] = rewired
        merge = merge_id(node_id)
        nodes[merge] = _merge_op(merge, node_id, partial)
        after.setdefault(node_id, []).append(merge)
        plan.aliases[merge] = node_id
        _point_readers_at(nodes, node_id, merge)
        if node_id in output_node_map:
            output_node_map[merge] = output_node_map.pop(node_id)
    for node_id in [node for node in nodes if node not in plan.partial]:
        op = nodes[node_id]
        if op.task_type == PYTHON_TASK_TYPE:
            continue
        for source in [node for node in op.dependencies if node in plan.hits]:
            step_id = _pass_through(nodes, plan, runtime_graph, node_id, source, None)
            before.setdefault(node_id, []).append(step_id)
            nodes[node_id] = _reference(source, step_id, nodes[node_id])
    return before, after


def _pass_through(
    nodes: dict[str, RuntimeOp],
    plan: ReusePlan,
    runtime_graph: RuntimeGraph,
    node_id: str,
    source: str,
    partial: PartialRecords | None,
) -> str:
    """The pass-through step ``node_id`` reads ``source`` through: every served row
    for an op that computes all of them, or the rows its partial run reads."""
    rows = _served_rows(plan, runtime_graph.nodes[source])
    positions = None
    if rows is not None:
        positions = (
            list(range(len(rows))) if partial is None else _positions(rows, partial)
        )
    if rows is None or positions is None:
        raise ValueError(f"op '{node_id}' cannot read a served row list")
    step_id = pass_id(node_id, source)
    nodes[step_id] = _pass_op(step_id, positions, rows)
    plan.pass_throughs[step_id] = source
    return step_id


def _point_readers_at(nodes: dict[str, RuntimeOp], node_id: str, merge: str) -> None:
    """Every dispatched reader of ``node_id`` reads the merge step instead: its
    dependency edge and the references its data spec names, so it takes every
    live row rather than the computed ones alone."""
    for reader_id, reader in list(nodes.items()):
        if reader_id in (merge, node_id) or node_id not in reader.dependencies:
            continue
        spec = data_spec_dict(reader)
        _rename_refs(spec, node_id, merge)
        nodes[reader_id] = replace(
            reader,
            data_spec=parse_data_spec(reader.task_type, spec),
            dependencies=tuple(
                merge if node == node_id else node for node in reader.dependencies
            ),
        )


def _merge_op(merge: str, node_id: str, partial: PartialRecords) -> RuntimeOp:
    """The merge step: a slot list of every live row's stored value, and the partial
    run's computed rows flattened back to one per row."""
    slots = [
        {"key": key, "value": value} for key, value in zip(partial.keys, partial.served)
    ]
    return _plumbing_op(
        merge,
        PythonData(
            mode="merge",
            plan=[
                ColumnPlanEntry(kind="literal", values=slots),
                ColumnPlanEntry(kind="stage", node=node_id, flatten=True),
            ],
            timeout_s=python_step.DEFAULT_TIMEOUT_SECONDS,
        ),
        dependencies=(node_id,),
    )


def _pass_op(
    step_id: str,
    positions: Sequence[int],
    rows: list[Mapping[str, Any]],
) -> RuntimeOp:
    """The pass-through step: the served producer's rows at ``positions``, in live
    order, as literals, since the store served that producer and no task
    emitted them."""
    return _plumbing_op(
        step_id,
        PythonData(
            mode="pass_through",
            plan=[
                ColumnPlanEntry(kind="literal", values=rows),
                ColumnPlanEntry(kind="literal", values=list(positions)),
            ],
            timeout_s=python_step.DEFAULT_TIMEOUT_SECONDS,
        ),
        dependencies=(),
    )


def _served_rows(plan: ReusePlan, source: RuntimeOp) -> list[Mapping[str, Any]] | None:
    """A served producer's whole item list, each row rebuilt the way its own step emits
    it, or ``None`` when the store holds no entry for it.

    Every producer row is expanded, whether it was stored per record key or as
    one whole-op value, so the count is the row count the partial run reads."""
    entries = plan.hits.get(source.node_id)
    if not entries:
        return None
    bare = _emits_bare_value(source)
    wrapped = _wrap(source)
    return [
        _served_item(item, None if bare else wrapped)
        for entries_of_row in served_lists(entries, grouped_local_chat(source))
        for item in entries_of_row
    ]


def _emits_bare_value(source: RuntimeOp) -> bool:
    """Whether the producer's step puts its value under ``output`` as it stands: an
    aligned or list step emits the value itself, where a row or parse step emits
    its JSON text."""
    spec = source.data_spec
    return (
        source.task_type == PYTHON_TASK_TYPE
        and isinstance(spec, PythonData)
        and spec.mode in ("aligned", "list")
    )


def _positions(
    rows: Sequence[Mapping[str, Any]], partial: PartialRecords
) -> list[int] | None:
    """The producer rows a partial op reads: its missed positions when the column
    carries one value per live record, the single row a broadcasting column
    repeats over every row this run computes, or ``None`` when the column's rows
    match neither the live records nor one value."""
    missed = list(partial.missed)
    if len(rows) == len(partial.keys):
        return missed
    if len(rows) == 1:
        return [0] * len(missed)
    return None


def _served_item(value: Any, wrapped: _Wrapping | None) -> Mapping[str, Any]:
    """One stored value as the item its producer's step emitted."""
    if wrapped == "text":
        return {"output": _output_text(value), "value": value}
    if wrapped == "output":
        return {"output": _output_text(value)}
    return {"output": value}


def _plumbing_op(
    node_id: str,
    spec: PythonData,
    *,
    dependencies: tuple[str, ...],
) -> RuntimeOp:
    """A generated python step a partial run inserts to carry rows between the op
    it narrows and its readers."""
    return RuntimeOp(
        node_id=node_id,
        task_type=PYTHON_TASK_TYPE,
        backend=python_step.BACKEND,
        model="",
        data_spec=spec,
        model_spec={},
        inference_spec={},
        dependencies=dependencies,
    )


def _reference(source: str, step_id: str, op: RuntimeOp) -> RuntimeOp:
    """The op reading ``step_id`` where it named ``source``, its dependency edge and
    paths rewritten so the step's items, which are the producer's own, read
    as they did."""
    spec = data_spec_dict(op)
    _rename_refs(spec, source, step_id)
    return replace(
        op,
        data_spec=parse_data_spec(op.task_type, spec),
        dependencies=tuple(
            step_id if node == source else node for node in op.dependencies
        ),
    )


def _rename_refs(value: Any, old: str, new: str) -> None:
    if isinstance(value, dict):
        if value.get("node") == old:
            value["node"] = new
        for item in value.values():
            _rename_refs(item, old, new)
    elif isinstance(value, list):
        for item in value:
            _rename_refs(item, old, new)


def _dispatch_record_keys(
    runtime_graph: RuntimeGraph, plan: ReusePlan, nodes: Mapping[str, RuntimeOp]
) -> dict[str, RecordKey]:
    """Each dispatched op's declared key, plus one inherited from a served input."""
    keys = {
        node_id: key
        for node_id, key in runtime_graph.record_keys.items()
        if node_id in nodes
    }
    # Only a served input declares a key: a dispatched producer hands its rows' keys
    # down by position, as a full run does, whatever path its op declares.
    for node_id, op in nodes.items():
        inherited = plan.key_paths.get(node_id)
        if node_id in keys or inherited is None:
            continue
        sources = [
            node
            for node in op.dependencies
            if node not in nodes and plan.key_paths.get(node)
        ]
        if len(sources) == 1:
            keys[node_id] = RecordKey(node=sources[0], path=inherited)
    return keys


def _planner_inputs(
    plan: RecordPlan,
    decision: ReusePlan,
    plans: Mapping[str, RecordPlan],
    content: Mapping[str, NodeRecords],
) -> list[_Input]:
    """The inputs one op's records read, named the way the writer names them."""
    inputs: list[_Input] = []
    for node in input_nodes(plan):
        hit = decision.hits.get(node)
        if hit is not None:
            inputs.append(_served_input(hit))
            continue
        if node in decision.content:
            inputs.append(_content_input(decision.content[node]))
            continue
        values = input_values(plan.op, node)
        inputs.append(
            _Input(
                named=RecordInput(
                    keyed=False,
                    records=[
                        (digest_of({"node": node, "values": values}), digest_of(values))
                    ],
                ),
                keys=[None],
            )
        )
    for rows in literal_rows(plan):
        inputs.append(_literal_input(rows, plan.key_path))
    return inputs


def _served_input(records: Sequence[RecordEntry]) -> _Input:
    """A served input, named by the entry each of its records read."""
    return _Input(
        named=RecordInput(
            keyed=records[0].address.record is not None,
            records=[(entry.digest, entry.value_digest or "") for entry in records],
        ),
        keys=[entry.address.record for entry in records],
    )


def _group_members(
    plan: RecordPlan, rows: Sequence[Any] | None
) -> list[tuple[str, list[int]]] | None:
    """A reduce's groups as ``(by value, member positions)`` in first-appearance
    order, from the rows it groups; ``None`` when those rows are not known
    before the op runs, so the group is a whole-op miss.

    Members are positions in the op's rows, so they name the same input records
    the writer zips into each group."""
    spec = plan.op.data_spec
    if not isinstance(spec, PythonData) or not spec.by or rows is None:
        return None
    groups: dict[str, list[int]] = {}
    for position, row in enumerate(rows):
        found = _value_at_path(_decoded(row), spec.by)
        if found is _UNSET:
            return None
        groups.setdefault(key_text(found), []).append(position)
    return list(groups.items())


def _group_inputs(
    inputs: Sequence[_Input], members: Sequence[int]
) -> tuple[list[str], list[str]]:
    """What one reduce group read, named in the order the writer names it: each
    keyed input's record at each member, then every whole input once."""
    keyed, whole = split_inputs([item.named for item in inputs])
    digests: list[str] = []
    values: list[str] = []
    for member in members:
        read_digests, read_values = position_inputs(keyed, member, keyed=True)
        digests.extend(read_digests)
        values.extend(read_values)
    whole_digests, whole_values = position_inputs(whole, 0, keyed=False)
    digests.extend(whole_digests)
    values.extend(whole_values)
    return digests, values


def _content_input(records: ContentRecords) -> _Input:
    """An unversioned read the plan re-ran, named by the writer's own rule: each
    row by the value the read returned, so the two sides share one identity."""
    named = content_input(
        NodeRecords(keyed=records.keyed, values=records.values, keys=records.keys)
    )
    positions = range(len(records.values)) if records.keyed else range(1)
    return _Input(
        named=named,
        keys=(
            [records.keys[position] for position in positions]
            if records.keyed and records.lines_up
            else [None]
        ),
    )


def _literal_input(values: Sequence[Any], key_path: str | None) -> _Input:
    """One literal rows list, named per value and keyed by the op's key path."""
    return _Input(
        named=RecordInput(
            keyed=True,
            records=[
                (digest_of({"values": value}), digest_of(value)) for value in values
            ],
        ),
        keys=[row_key(value, key_path) if key_path else None for value in values],
    )


def _record_keys(plan: RecordPlan, inputs: Sequence[_Input]) -> list[str | None] | None:
    """Every record an op has before it ran, or ``None`` when a keyed op lacks one."""
    if not plan.keyed:
        return [None]
    keys: list[str | None] = []
    seen: set[str] = set()
    for item in inputs:
        if all(key is None for key in item.keys):
            continue
        for key in item.keys:
            if key is None:
                return None
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys or None


def _seed_edges(runtime_graph: RuntimeGraph) -> dict[str, list[str]]:
    """Each seed node paired with the chats reading it, so they sort after it."""
    edges: dict[str, list[str]] = {}
    for node_id, node in runtime_graph.nodes.items():
        spec = node.data_spec
        if isinstance(spec, DataFrame) and isinstance(spec.seeds, NodeRead):
            edges.setdefault(spec.seeds.node, []).append(node_id)
    return edges


def _seeded_records(
    plan: RecordPlan,
    stored: Mapping[str, list[RecordEntry]],
    runtime_graph: RuntimeGraph,
) -> dict[str, NodeRecords]:
    """The planned records a node-bound seed column reads, in the seed node's items."""
    spec = plan.op.data_spec
    if not isinstance(spec, DataFrame) or not isinstance(spec.seeds, NodeRead):
        return {}
    node = spec.seeds.node
    entries = stored.get(node)
    if entries is None:
        return {}
    wrapping = _wrap(runtime_graph.nodes[node])
    if entries[0].address.record is None:
        values = entries[0].value
        if not isinstance(values, list):
            return {}
        return {
            node: NodeRecords(
                keyed=False,
                values=values,
                keys=None,
                items=[_seed_item(v, wrapping) for v in values],
            )
        }
    return {
        node: NodeRecords(
            keyed=True,
            values=[entry.value for entry in entries],
            keys=[entry.address.record for entry in entries],
            items=[_seed_item(entry.value, wrapping) for entry in entries],
        )
    }


def _seed_item(value: Any, wrapping: _Wrapping | None) -> Any:
    """One seed value shaped as the seed node's own item."""
    if wrapping == "text":
        return {"output": _output_text(value), "value": value}
    return {"output": value} if wrapping == "output" else value


def _planned_realization(
    plan: RecordPlan,
    records: list[str | None],
    position: int,
    source: RecordSource,
    seeded: Mapping[str, NodeRecords],
) -> RecordRealization | None:
    """The realization at ``position``, or ``None`` when no seed reproduces it."""
    spec = plan.op.data_spec
    seeds = spec.seeds if isinstance(spec, DataFrame) else None
    if isinstance(seeds, list) and len(seeds) != len(records):
        return None
    node_records = NodeRecords(
        keyed=plan.keyed,
        values=[],
        keys=records if plan.keyed else [],
    )
    item = realization(plan, node_records, seeded, position, source, unversioned=False)
    if item.production is not None:
        return None
    return RecordRealization(seed=item.seed, attempt=item.attempt)


def _counted(
    plan: RecordPlan,
    records: list[str | None] | None,
    inputs: Sequence[_Input],
) -> int:
    """How many records a missing op covers."""
    if records is not None:
        return len(records)
    if not plan.keyed:
        return 1
    keys = {key for item in inputs for key in item.keys}
    return len(keys - {None}) + (1 if None in keys else 0)


def _unserved_reader_reason(
    runtime_graph: RuntimeGraph,
    plan: ReusePlan,
    op: RuntimeOp,
    hit: Sequence[RecordEntry],
) -> str | None:
    """The reason a hit stays dispatched, or ``None`` when it is served."""
    for reader in runtime_graph.nodes.values():
        if op.node_id not in reader.dependencies:
            continue
        spec = reader.data_spec
        if isinstance(spec, DataFrame) and isinstance(spec.seeds, NodeRead):
            if spec.seeds.node == op.node_id:
                continue
        if reader.task_type != PYTHON_TASK_TYPE:
            if not _reads_literal_ref(reader, op, hit):
                return "read by a non-literal op"
            continue
        spec = reader.data_spec
        if not isinstance(spec, PythonData) or not spec.plan:
            return "value not servable"
        for entry in spec.plan:
            if entry.node != op.node_id:
                continue
            if served_column(hit, entry, _wrap(op), grouped_local_chat(op)) is None:
                return "value not servable"
    return None


def _reads_literal_ref(
    reader: RuntimeOp, producer: RuntimeOp, entries: Sequence[RecordEntry]
) -> bool:
    """Whether every reference ``reader`` makes to ``producer`` names a scalar."""
    refs = _list_refs(data_spec_dict(reader))
    if refs is None:
        return False
    to_node = [ref for ref in refs if ref.get("node") == producer.node_id]
    if not to_node:
        return False
    typed = typed_row_step(producer)
    for ref in to_node:
        values = _model_entry_values(entries, ref.get("path"), typed)
        if any(value is _UNSET for value in values):
            return False
        if _stored_path(ref.get("path")) == "" and any(
            isinstance(_decoded(value), list) for value in values
        ):
            return False
    return True


def _entry_miss_reason(entry: RecordEntry | None) -> str | None:
    """Why a stored entry cannot answer, or ``None`` when it can."""
    if entry is None:
        return "no entry"
    if entry.status != "ok" or is_marker(entry.value):
        return "failed entry"
    if digest_of(entry.value) != entry.value_digest:
        return "value not servable"
    return None
