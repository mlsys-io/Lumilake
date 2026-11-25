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
    NodeRead,
    PythonData,
)
from lumilake_server.ops import RecordKey
from lumilake_server.runtime.lambda_task_helpers import items_column
from lumilake_server.runtime.record_store import (
    EffectVersions,
    RecordAddress,
    RecordEntry,
    RecordRealization,
    RecordSource,
    digest_of,
    entry_digest,
    is_marker,
    row_key,
)
from lumilake_server.runtime.record_writer import (
    NodeRecords,
    RecordInput,
    RecordPlan,
    grouped_local_chat,
    input_nodes,
    input_values,
    key_path,
    literal_rows,
    logical_names,
    position_inputs,
    read_entry,
    reads_unversioned,
    realization,
    typed_row_step,
)
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp
from lumilake_server.utils.job_storage import JobStorage

_SERVED_TASKS = frozenset({"python", "inference", "classification", "embedding"})
"""Task types whose stored value a served op carries whole; an api, diffusion
or fetch task archives its value as an artifact instead."""

_SERVED_OUTPUT_PATHS = frozenset({None, "items.output"})
"""Output paths whose archived value is one stored value per record in order."""


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


@dataclass(frozen=True)
class _Input:
    """One input's writer identity, and the record key at each of its rows."""

    named: RecordInput
    keys: list[str | None]


class ReusePlanner:
    """Decides, before dispatch, which ops of a reusing job the store answers."""

    def __init__(self, *, storage: JobStorage, logger: Logger) -> None:
        self._storage = storage
        self.logger = logger

    def plan(
        self, *, runtime_graph: RuntimeGraph, round_number: int, job_id: str
    ) -> tuple[ReusePlan, RuntimeGraph]:
        """Decide every op in topological order and return the graph to dispatch."""
        names = logical_names(runtime_graph)
        decision = ReusePlan(outputs=runtime_graph.output_node_map.copy())
        misses: set[str] = set()
        stored: dict[str, list[RecordEntry]] = {}
        plans: dict[str, RecordPlan] = {}
        for node_id in runtime_graph.topological_order(_seed_edges(runtime_graph)):
            plans[node_id] = RecordPlan(
                op=runtime_graph.nodes[node_id],
                op_name=names.get(node_id, node_id),
                key_path=key_path(runtime_graph, node_id, plans),
                logical_names=names,
            )
            plan = plans[node_id]
            if plan.key_path is not None:
                decision.key_paths[node_id] = plan.key_path
            inputs = _planner_inputs(plan, decision)
            records = _record_keys(plan, inputs)
            reason, hit = self._decide(
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
            unserved = _unserved_reader_reason(runtime_graph, plan.op, hit)
            if unserved is not None:
                decision.decisions[-1] = decision.decisions[-1].model_copy(
                    update={"decision": "miss", "reason": unserved}
                )
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
    ) -> tuple[str | None, list[RecordEntry] | None]:
        """One op's ``(first miss reason, stored entries)``."""
        if plan.op.task_type not in _SERVED_TASKS:
            return "value not servable", None
        if node_id in runtime_graph.output_node_map and not _servable_output(
            runtime_graph, plan, node_id
        ):
            return "value not servable", None
        if any(node in misses for node in input_nodes(plan)):
            return "upstream miss", None
        if reads_unversioned(plan):
            return "unversioned upstream", None
        if records is None:
            return "unkeyed row", None
        named = [item.named for item in inputs]
        entries: list[RecordEntry] = []
        source = RecordSource(job=job_id, round=round_number, task_id="")
        seeded = _seeded_records(plan, stored, runtime_graph)
        for position, record in enumerate(records):
            draw = _planned_realization(plan, records, position, source, seeded)
            if draw is None:
                return "no realization", None
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
            if reason is not None or entry is None:
                return reason, None
            entries.append(entry)
        return None, entries


def rebuilt_output(plan: ReusePlan, node_id: str) -> list[str]:
    """A hit output node's archived values, as the aggregator emits them."""
    entries = plan.hits[node_id]
    if entries[0].address.record is None:
        return [_output_text(entries[0].value)]
    return [_output_text(entry.value) for entry in entries]


def _output_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def served_records(plan: ReusePlan) -> dict[str, NodeRecords]:
    """The records every hit serves, shaped as the writer reads a finished node."""
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
    return served


def _servable_output(
    runtime_graph: RuntimeGraph, plan: RecordPlan, node_id: str
) -> bool:
    """Whether a hit at an output node rebuilds its archived value."""
    if plan.op.task_type not in _SERVED_TASKS:
        return False
    path = runtime_graph.output_paths.get(node_id)
    if path not in _SERVED_OUTPUT_PATHS:
        return False
    return path is None if plan.keyed else _list_mode(plan.op)


def _list_mode(op: RuntimeOp) -> bool:
    """Whether the op runs its whole input as one list-mode python step."""
    spec = op.data_spec
    return (
        op.task_type == PYTHON_TASK_TYPE
        and isinstance(spec, PythonData)
        and spec.mode == "list"
    )


def _rewritten(
    op: RuntimeOp, plan: ReusePlan, runtime_graph: RuntimeGraph
) -> RuntimeOp:
    """An op whose hit inputs read their stored values, with its spec digest kept."""
    served = [node for node in op.dependencies if node in plan.hits]
    if not served:
        return op
    if op.task_type != PYTHON_TASK_TYPE:
        raise ValueError(f"op '{op.node_id}' cannot read a served record")
    spec = op.data_spec
    if not isinstance(spec, PythonData):
        raise ValueError(f"op '{op.node_id}' cannot read a served record")
    entries = spec.plan
    if not entries:
        raise ValueError(f"python step '{op.node_id}' carries no column plan")
    rewritten: list[ColumnPlanEntry] = []
    for entry in entries:
        node = entry.node
        if node is None or node not in plan.hits:
            rewritten.append(entry)
            continue
        producer = runtime_graph.nodes[node]
        column = served_column(
            plan.hits[node],
            entry,
            _wrap(producer),
            grouped_local_chat(producer),
        )
        if column is None:
            raise ValueError(f"op '{op.node_id}' cannot rebuild a served column")
        rewritten.append(entry.model_copy(update={"values": column}))
    return replace(op, data_spec=spec.model_copy(update={"plan": rewritten}))


_WRAPPED_TASKS = frozenset({"python", "inference"})
"""Task types whose items carry their value under ``output``, which the writer
stores unwrapped."""


type _Wrapping = Literal["output", "text"]


def _wrap(op: RuntimeOp) -> _Wrapping | None:
    """``text`` for a row step's item, ``output`` for a bare item, else ``None``."""
    if typed_row_step(op):
        return "text"
    return "output" if op.task_type in _WRAPPED_TASKS else None


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
    """The graph to dispatch: every hit gone, every miss reading stored values."""
    nodes: dict[str, RuntimeOp] = {}
    for node_id in runtime_graph.node_order:
        if node_id in plan.hits:
            continue
        nodes[node_id] = _rewritten(runtime_graph.nodes[node_id], plan, runtime_graph)
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
        dsl_to_runtime=runtime_graph.dsl_to_runtime.copy(),
        record_keys=_dispatch_record_keys(runtime_graph, plan, nodes),
        op_names=runtime_graph.op_names.copy(),
    )


def _dispatch_record_keys(
    runtime_graph: RuntimeGraph, plan: ReusePlan, nodes: Mapping[str, RuntimeOp]
) -> dict[str, RecordKey]:
    """Each dispatched op's declared key, plus one inherited from a served input."""
    keys = {
        node_id: key
        for node_id, key in runtime_graph.record_keys.items()
        if node_id in nodes
    }
    for node_id, op in nodes.items():
        inherited = plan.key_paths.get(node_id)
        if node_id in keys or inherited is None:
            continue
        served = [
            node
            for node in op.dependencies
            if node in plan.hits and plan.key_paths.get(node) is not None
        ]
        if len(served) == 1:
            keys[node_id] = RecordKey(node=served[0], path=inherited)
    return keys


def _planner_inputs(plan: RecordPlan, decision: ReusePlan) -> list[_Input]:
    """The inputs one op's records read, named the way the writer names them."""
    inputs: list[_Input] = []
    for node in input_nodes(plan):
        hit = decision.hits.get(node)
        if hit is not None:
            inputs.append(_served_input(hit))
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
            keyed=True,
            records=[(entry.digest, entry.value_digest or "") for entry in records],
        ),
        keys=[entry.address.record for entry in records],
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
    """Every record an op has before it ran: one per distinct key, or one whole-op."""
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
    runtime_graph: RuntimeGraph, op: RuntimeOp, hit: Sequence[RecordEntry]
) -> str | None:
    """Why a hit is not served: a non-python reader, or an unrebuildable column."""
    for reader in runtime_graph.nodes.values():
        if op.node_id not in reader.dependencies:
            continue
        spec = reader.data_spec
        if isinstance(spec, DataFrame) and isinstance(spec.seeds, NodeRead):
            if spec.seeds.node == op.node_id:
                continue
        if reader.task_type != PYTHON_TASK_TYPE:
            return "read by a non-literal op"
        if not isinstance(spec, PythonData):
            return "value not servable"
        entries = spec.plan
        if not entries:
            return "value not servable"
        for entry in entries:
            if (
                entry.node == op.node_id
                and served_column(hit, entry, _wrap(op), grouped_local_chat(op)) is None
            ):
                return "value not servable"
    return None


def _entry_miss_reason(entry: RecordEntry | None) -> str | None:
    """Why a stored entry cannot answer, or ``None`` when it can."""
    if entry is None:
        return "no entry"
    if entry.status != "ok" or is_marker(entry.value):
        return "failed entry"
    if digest_of(entry.value) != entry.value_digest:
        return "value not servable"
    return None
