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
    parse_data_spec,
)
from lumilake_server.ops import RecordKey
from lumilake_server.runtime.lambda_task_helpers import items_column
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
    row_key,
)
from lumilake_server.runtime.record_writer import (
    NodeRecords,
    RecordInput,
    RecordPlan,
    _decoded,
    content_value,
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


@dataclass(frozen=True)
class OutputShape:
    """How an output node archives its records."""

    path: str | None = None
    """The stored value's own field to archive, when the output names one."""
    whole_list: bool = False

    def archived(self, values: Sequence[Any]) -> list[str]:
        selected = [_value_at_path(value, self.path) for value in values]
        if self.whole_list:
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

    def value_at(self, position: int) -> Any:
        return self.values[position] if self.keyed else self.values


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

    def served_nodes(self) -> set[str]:
        return {*self.hits, *self.content_served}


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
        self,
        *,
        runtime_graph: RuntimeGraph,
        round_number: int,
        job_id: str,
        content_values: Mapping[str, NodeRecords] | None = None,
    ) -> tuple[ReusePlan, RuntimeGraph]:
        """Decide every op in topological order and return the graph to dispatch."""
        names = logical_names(runtime_graph)
        decision = ReusePlan(outputs=runtime_graph.output_node_map.copy())
        misses: set[str] = set()
        stored: dict[str, list[RecordEntry]] = {}
        plans: dict[str, RecordPlan] = {}
        content = {} if content_values is None else content_values
        content_keys = set(content)
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
            if node_id in content_keys:
                self._serve_content(decision, plan, content[node_id], runtime_graph)
                continue
            inputs = _planner_inputs(plan, decision, plans, content)
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
            unserved = _unserved_reader_reason(runtime_graph, decision, plan.op, hit)
            if unserved is not None:
                decision.decisions[-1] = decision.decisions[-1].model_copy(
                    update={"decision": "miss", "reason": unserved}
                )
                misses.add(node_id)
                continue
            if node_id in runtime_graph.output_node_map:
                shape = output_shape(plan, runtime_graph.output_paths.get(node_id))
                if shape is None or output_runs(shape, hit):
                    decision.decisions[-1] = decision.decisions[-1].model_copy(
                        update={"decision": "miss", "reason": "value not servable"}
                    )
                    misses.add(node_id)
                    continue
                decision.output_shapes[node_id] = shape
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
        )
        if _content_servable(runtime_graph, decision, plan.op.node_id):
            decision.content_served.add(plan.op.node_id)

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
        if reads_unversioned(plan.op):
            return "unversioned upstream", None
        if plan.op.task_type not in _SERVED_TASKS:
            return "value not servable", None
        if any(node in misses for node in input_nodes(plan)):
            return "upstream miss", None
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
    spec = plan.op.data_spec
    whole_list = isinstance(spec, PythonData) and spec.mode == "list" and not plan.keyed
    if path is None or path == "items" or path == "":
        return OutputShape(whole_list=whole_list)
    field = _stored_path(path)
    if field is None:
        return None
    return OutputShape(path=field, whole_list=whole_list)


def output_runs(shape: OutputShape, entries: Sequence[RecordEntry]) -> bool:
    """Whether a hit output runs: it reads a field or list no stored value holds."""
    if shape.whole_list:
        return entries[0].address.record is not None or not isinstance(
            entries[0].value, list
        )
    return any(
        _value_at_path(value, shape.path) is _UNSET for value in _hit_values(entries)
    )


def _rewritten(
    op: RuntimeOp, plan: ReusePlan, runtime_graph: RuntimeGraph
) -> RuntimeOp:
    """An op reading its served inputs' values, each entry gaining values."""
    served = [node for node in op.dependencies if _served_node(plan, node)]
    if not served:
        return op
    spec = op.data_spec
    if isinstance(spec, PythonData) and spec.plan:
        return _rewritten_plan(op, spec, plan, runtime_graph)
    if _reads_literal_lists(op):
        literals = _replace_with_literals(data_spec_dict(op), plan, runtime_graph)
        return replace(op, data_spec=parse_data_spec(op.task_type, literals))
    raise ValueError(f"op '{op.node_id}' cannot read a served record")


def _rewritten_plan(
    op: RuntimeOp,
    spec: PythonData,
    plan: ReusePlan,
    runtime_graph: RuntimeGraph,
) -> RuntimeOp:
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
    return replace(op, data_spec=spec.model_copy(update={"plan": rewritten}))


def _reads_literal_lists(op: RuntimeOp) -> bool:
    """Whether the op reads its rows in positions a literal list fills."""
    return (
        op.task_type != PYTHON_TASK_TYPE and _list_refs(data_spec_dict(op)) is not None
    )


def _list_refs(value: Any) -> list[Mapping[str, Any]] | None:
    """Every reference a literal list can replace, or ``None`` for one it cannot."""
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


def _replace_with_literals(
    value: Any, plan: ReusePlan, runtime_graph: RuntimeGraph
) -> Any:
    """One data-spec side with every served reference replaced by its values."""
    if isinstance(value, Mapping):
        node = value.get("node")
        if isinstance(node, str) and _served_node(plan, node):
            typed = typed_row_step(runtime_graph.nodes[node])
            items = _model_entry_values(plan.hits[node], value.get("path"), typed)
            if any(item is _UNSET for item in items):
                raise ValueError("op reads a field the stored value does not hold")
            if value.get("type") == "list":
                return {**value, "items": items}
            if "label" in value:
                return {
                    "label": value["label"],
                    "data": {"type": "list", "items": items},
                }
            return {"type": "list", "items": items}
        return {
            key: _replace_with_literals(item, plan, runtime_graph)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_with_literals(item, plan, runtime_graph) for item in value]
    return value


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
    """The column a python reader takes from a re-ran read."""
    try:
        return items_column(
            entry.model_dump(exclude_none=True, exclude={"values"}),
            plan.content[node].values,
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
    """The graph to dispatch, with every served node reading its value instead."""
    nodes: dict[str, RuntimeOp] = {}
    for node_id in runtime_graph.node_order:
        if node_id in plan.served_nodes():
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
            keyed=True,
            records=[(entry.digest, entry.value_digest or "") for entry in records],
        ),
        keys=[entry.address.record for entry in records],
    )


def _content_input(records: ContentRecords) -> _Input:
    """An unversioned read the plan re-ran, named by the value it returned."""
    positions = range(len(records.values)) if records.keyed else range(1)
    named: list[tuple[str, str] | None] = []
    for position in positions:
        value = content_value(records.value_at(position))
        named.append((digest_of({"value": value}), digest_of(value)))
    return _Input(
        named=RecordInput(keyed=records.keyed, records=named),
        keys=(
            [records.keys[position] for position in positions]
            if records.keyed
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
