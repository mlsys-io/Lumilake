"""Whole-op revision plan: which ops a reusing job keeps from the store.

A job submitted with ``reuse: true`` asks the server to answer every record an
earlier storing job already computed, and to dispatch only the ops whose
records it cannot answer. The plan is decided before any task runs: the runtime
graph is walked in topological order and each op's records are keyed with the
writer's own helpers, so a hit is an object the writer itself would accept.

Granularity is the whole op. An op hits only when every one of its records has a
trusted success entry, and one miss reruns all of them. An op downstream of a
miss, of an unversioned read or of a record whose draw no seed reproduces
misses too, because its input identity is then unknown before it runs.

A hit is applied as the plan is made. The hit node leaves the graph, and each op
that read it keeps that dependency but carries the stored values in the same
column-plan entry, so the op's spec digest, the names its writer gives its
inputs and the worker's read of the column all stay what they were.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lumilake.log import Logger
from lumilake_server.runtime.record_store import (
    EffectVersions,
    RecordAddress,
    RecordEntry,
    RecordRealization,
    RecordSource,
    digest_of,
    entry_digest,
    is_marker,
)
from lumilake_server.runtime.record_writer import (
    NodeRecords,
    RecordInput,
    RecordPlan,
    _input_nodes,
    _input_values,
    _key_path,
    _literal_entries,
    _logical_names,
    _position_inputs,
    _reads_unversioned,
    _realization,
    read_entry,
)
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp
from lumilake_server.utils.job_storage import JobStorage

_LAMBDA_TASK_TYPE = "python"

_SERVED_TASKS = frozenset({"python", "inference", "classification", "embedding"})
"""Task types whose value a stored entry carries whole. An api, diffusion or
fetch task archives its value as an artifact the entry does not carry, so such
an op always runs."""

_SERVED_OUTPUT_PATHS = frozenset({None})
"""Output paths whose archived value is one stored entry per record in order:
a python step's row output, or a list-mode step's whole-list record."""


class RecordDecision(BaseModel):
    """What a reusing job decided for one op, and the first reason it missed."""

    model_config = ConfigDict(extra="forbid")

    op: str
    decision: Literal["hit", "miss"]
    reason: str | None = None
    records: int = 0
    """How many records the decision covers."""


class ReusePlan(BaseModel):
    """One reusing job's plan: what the store answers, and every decision."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[RecordDecision] = Field(default_factory=list)
    hits: dict[str, list[RecordEntry]] = Field(default_factory=dict)
    """Node id -> the trusted entries that answer its records, in key order."""

    def hit_nodes(self) -> set[str]:
        return set(self.hits)


@dataclass(frozen=True)
class _Input:
    """One of an op's inputs as the planner knows it: the identity the writer
    names it by, and the record key at each of its rows."""

    named: RecordInput
    keys: list[str | None]


class ReusePlanner:
    """Decides, before dispatch, which ops of a reusing job the store answers."""

    def __init__(self, *, storage: JobStorage, logger: Logger) -> None:
        self._storage = storage
        self.logger = logger

    async def plan(
        self, *, runtime_graph: RuntimeGraph, round_number: int, job_id: str
    ) -> tuple[ReusePlan, RuntimeGraph]:
        """Walk the graph in topological order and decide every op, returning
        the plan and the graph to dispatch in its place.

        Every op is keyed from the form the workflow declares, so the key a
        full run wrote answers it. The graph returned from a plan whose misses
        are dispatched is built from the same declared form and served entries,
        so its writer keys those ops exactly as this walk did."""
        names = _logical_names(runtime_graph)
        decision = ReusePlan()
        misses: set[str] = set()
        plans: dict[str, RecordPlan] = {}
        for node_id in runtime_graph.topological_order():
            plans[node_id] = RecordPlan(
                op=runtime_graph.nodes[node_id],
                op_name=names.get(node_id, node_id),
                key_path=_key_path(runtime_graph, node_id, plans),
                logical_names=names,
            )
            plan = plans[node_id]
            inputs = _planner_inputs(plan, decision)
            records = _record_keys(plan, inputs)
            reason, hit = self._decide(
                plan,
                inputs,
                records,
                round_number=round_number,
                job_id=job_id,
                misses=misses,
                runtime_graph=runtime_graph,
                node_id=node_id,
            )
            decision.decisions.append(
                RecordDecision(
                    op=plan.op_name,
                    decision="hit" if hit is not None else "miss",
                    reason=reason,
                    records=len(hit) if hit is not None else len(records),
                )
            )
            if hit is None:
                misses.add(node_id)
                continue
            decision.hits[node_id] = hit
        self.logger.info(
            "Record reuse plan for job %s: %d hit op(s), %d miss op(s)",
            job_id,
            len(decision.hits),
            len(decision.decisions) - len(decision.hits),
        )
        return decision, _without_hits(runtime_graph, decision)

    def _decide(
        self,
        plan: RecordPlan,
        inputs: Sequence[_Input],
        records: Sequence[str | None],
        *,
        round_number: int,
        job_id: str,
        misses: set[str],
        runtime_graph: RuntimeGraph,
        node_id: str,
    ) -> tuple[str | None, list[RecordEntry] | None]:
        """One op's ``(first miss reason, stored entries)``: every record must
        have a trusted success entry under the key the writer would use."""
        if plan.op.task_type not in _SERVED_TASKS:
            return "value not servable", None
        if node_id in runtime_graph.output_node_map and not _servable_output(
            runtime_graph, plan, node_id
        ):
            return "value not servable", None
        if any(node in misses for node in _input_nodes(plan)):
            return "upstream miss", None
        if _reads_unversioned(plan):
            return "unversioned upstream", None
        realization = _planned_realization(
            plan, records, RecordSource(job=job_id, round=round_number, task_id="")
        )
        if realization is None:
            return "no realization", None
        named = [item.named for item in inputs]
        entries: list[RecordEntry] = []
        for position, record in enumerate(records):
            input_digests, input_values = _position_inputs(
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
                    input_values=input_values,
                    realization=realization,
                    versions=EffectVersions(),
                ),
            )
            reason = _entry_miss_reason(entry)
            if reason is not None or entry is None:
                return reason, None
            entries.append(entry)
        return None, entries


def rebuilt_output(plan: ReusePlan, node_id: str) -> list[str]:
    """One hit output node's archived values, from its stored entries.

    A row op archives one value per record; a list-mode op archives one value
    holding its whole list. Both are what the node's own aggregation returns
    for a stored record's value, so a hit's outputs equal a full run's."""
    entries = plan.hits[node_id]
    if entries[0].address.record is None:
        return [json.dumps(entries[0].value)]
    return [json.dumps(entry.value) for entry in entries]


def served_records(plan: ReusePlan) -> dict[str, NodeRecords]:
    """The records every hit serves, shaped as the writer reads a finished
    node's records so a consumer names them identically.

    A keyed op's entry value is one row's output; a whole-op record's value is
    the node's whole item list, which is what the writer reads at position 0."""
    served: dict[str, NodeRecords] = {}
    for node_id, entries in plan.hits.items():
        keyed = entries[0].address.record is not None
        records = NodeRecords(
            keyed=keyed,
            values=(
                [entry.value for entry in entries] if keyed else list(entries[0].value)
            ),
            keys=[entry.address.record for entry in entries],
        )
        for entry in entries:
            records.digests[entry.address.record] = (entry.digest, entry.value_digest)
        served[node_id] = records
    return served


def _servable_output(
    runtime_graph: RuntimeGraph, plan: RecordPlan, node_id: str
) -> bool:
    """Whether a hit at an output node rebuilds that node's archived value.

    A python step's output is one stored value per record, in order. A list-mode
    step is one whole-op record holding the whole list. Any other output path
    reads a field the stored value does not name, so the node runs."""
    if runtime_graph.output_paths.get(node_id) not in _SERVED_OUTPUT_PATHS:
        return False
    return plan.op.task_type == _LAMBDA_TASK_TYPE


def _rewritten(op: RuntimeOp, plan: ReusePlan) -> RuntimeOp:
    """An op whose already-decided hit inputs read their stored values.

    The hit stays in the op's dependencies and its column-plan entry keeps its
    kind, node and path, so the op's spec digest and the names its writer gives
    its inputs do not move. Only the values the entry reads are added."""
    served = [node for node in op.dependencies if node in plan.hits]
    if not served:
        return op
    if op.task_type != _LAMBDA_TASK_TYPE:
        raise ValueError(f"op '{op.node_id}' cannot read a served record")
    entries = op.data_spec.get("plan")
    if not isinstance(entries, list):
        raise ValueError(f"python step '{op.node_id}' carries no column plan")
    rewritten: list[dict[str, Any]] = []
    for entry in entries:
        node = _entry_node(entry)
        if node is None or node not in plan.hits:
            rewritten.append(entry)
            continue
        rewritten.append(
            {
                **entry,
                "values": [
                    _value_at_path(record.value, entry.get("path"))
                    for record in plan.hits[node]
                ],
            }
        )
    return replace(op, data_spec={**op.data_spec, "plan": rewritten})


def _entry_node(entry: Any) -> str | None:
    return entry.get("node") if isinstance(entry, dict) else None


def _value_at_path(value: Any, path: Any) -> Any:
    """One stored value selected at the entry's declared path, as a worker
    reads that field of an upstream item."""
    if not isinstance(path, str) or not path:
        return value
    current = value
    for part in path.split("."):
        if isinstance(current, str):
            try:
                current = json.loads(current)
            except ValueError:
                return current
        if isinstance(current, dict) and part in current:
            current = current[part]
    return current


def _without_hits(runtime_graph: RuntimeGraph, plan: ReusePlan) -> RuntimeGraph:
    """The graph to dispatch: every hit node gone, and every miss whose input a
    hit answers reading that hit's stored values in its column plan.

    A dispatched op's column-plan entry keeps its kind, node and path and gains
    the values, so its spec digest is what the workflow declares and its writer
    keys it exactly as this plan did."""
    nodes: dict[str, RuntimeOp] = {}
    for node_id in runtime_graph.node_order:
        if node_id in plan.hits:
            continue
        nodes[node_id] = _rewritten(runtime_graph.nodes[node_id], plan)
    return RuntimeGraph(
        nodes=nodes,
        node_order=[
            node_id for node_id in runtime_graph.node_order if node_id in nodes
        ],
        output_node_map={
            node_id: name
            for node_id, name in runtime_graph.output_node_map.items()
            if node_id in nodes
        },
        output_paths={
            node_id: path
            for node_id, path in runtime_graph.output_paths.items()
            if node_id in nodes
        },
        dsl_to_runtime={
            op_id: [node_id for node_id in runtime_ids]
            for op_id, runtime_ids in runtime_graph.dsl_to_runtime.items()
        },
        record_keys={
            node_id: key
            for node_id, key in runtime_graph.record_keys.items()
            if node_id in nodes
        },
        op_names=dict(runtime_graph.op_names),
    )


def _planner_inputs(plan: RecordPlan, decision: ReusePlan) -> list[_Input]:
    """The inputs one op's records read, named the way the writer names them.

    A hit input is named by the entry the store answered for each of its
    records, exactly as a full run names the entry its batch wrote; every other
    node is named by its declaration, as the writer names a dependency the
    batch did not store."""
    inputs: list[_Input] = []
    for node in _input_nodes(plan):
        hit = decision.hits.get(node)
        if hit is not None:
            inputs.append(_served_input(hit))
            continue
        values = _input_values(plan.op, node)
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
    for entry in _literal_entries(plan):
        inputs.append(_literal_input(entry, plan.key_path))
    return inputs


def _served_input(records: Sequence[RecordEntry]) -> _Input:
    """A served input, named by the entry each of its records read."""
    return _Input(
        named=RecordInput(
            keyed=True,
            records=[(entry.digest, entry.value_digest or "") for entry in records],
        ),
        keys=[entry.address.record for entry in records],
    )


def _literal_input(entry: Any, key_path: str | None) -> _Input:
    """A literal entry, named and keyed per value."""
    values = entry.get("values", [])
    return _Input(
        named=RecordInput(
            keyed=True,
            records=[
                (digest_of({"values": value}), digest_of(value)) for value in values
            ],
        ),
        keys=[_value_key(value, key_path) if key_path else None for value in values],
    )


def _value_key(value: Any, key_path: str) -> str | None:
    """The record key a row carries at ``key_path``, as the writer spells it."""
    current = value
    for part in [part for part in key_path.split(".") if part]:
        if isinstance(current, str):
            try:
                current = json.loads(current)
            except ValueError:
                return None
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if current is None:
        return None
    return current if isinstance(current, str) else json.dumps(current, sort_keys=True)


def _record_keys(plan: RecordPlan, inputs: Sequence[_Input]) -> list[str | None]:
    """Every record an op has before it ran: one per key of a keyed input, or
    one whole-op record for an op that carries no key."""
    if not plan.keyed:
        return [None]
    return [key for item in inputs for key in item.keys if key is not None]


def _planned_realization(
    plan: RecordPlan, records: Sequence[str | None], source: RecordSource
) -> RecordRealization | None:
    """The realization every one of an op's records keys with, or ``None`` when
    the records disagree or a draw no seed reproduces enters it, so no stored
    key can answer the op before it runs."""
    keys = list(records) if plan.keyed else []
    seeds = plan.op.data_spec.get("seeds")
    if isinstance(seeds, list) and len(seeds) != len(keys):
        return None
    node_records = NodeRecords(keyed=plan.keyed, values=[], keys=keys)
    shared = {
        (item.seed, item.attempt, item.production)
        for item in (
            _realization(plan, node_records, position, source, unversioned=False)
            for position in range(len(records))
        )
    }
    if len(shared) != 1:
        return None
    seed, attempt, production = shared.pop()
    if production is not None:
        return None
    return RecordRealization(seed=seed, attempt=attempt)


def _entry_miss_reason(entry: RecordEntry | None) -> str | None:
    """Why a stored entry cannot answer, or ``None`` when it can."""
    if entry is None:
        return "no entry"
    if entry.status != "ok" or is_marker(entry.value):
        return "failed entry"
    if digest_of(entry.value) != entry.value_digest:
        return "value not servable"
    return None
