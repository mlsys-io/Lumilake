"""Per-record store writer and lineage read: one entry per record under
``records/<digest>.json``, then one index object per member job."""

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any
from weakref import WeakValueDictionary

from lumilake.log import Logger
from lumilake_server.data_specs import DataFrame, PythonData
from lumilake_server.ops import RecordKey
from lumilake_server.runtime.record_store import (
    EffectVersions,
    RecordAddress,
    RecordEntry,
    RecordIndex,
    RecordIndexEntry,
    RecordRealization,
    RecordSource,
    data_spec_dict,
    declared_seed,
    digest_of,
    entry_digest,
    entry_object_name,
    failed_record_digest,
    index_object_name,
    is_marker,
    literal_lists,
    op_spec_digest,
    path_value,
    row_key,
    samples,
)
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_graph.outputs import item_relative_path
from lumilake_server.runtime.runtime_ops import RuntimeOp
from lumilake_server.runtime.sensitive import redact_sensitive
from lumilake_server.utils.job_storage import ArchiveNotFound, JobStorage

_LAMBDA_TASK_TYPE = "python"
"""The task type a Lambda op lowers to."""

_UNVERSIONED_TASKS = frozenset({"data_retrieval", "api"})
"""Task types that read a source no snapshot records."""

_INDEX_LOCKS: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
"""One lock per job id, held across a batch's load-merge-save of its index."""


class RecordInput:
    """One input node as a consumer reads it: whether its records are keyed,
    and at each row position of the task the identity of the record there
    (``None`` for a row that carries no record)."""

    def __init__(
        self,
        *,
        keyed: bool,
        records: Sequence[tuple[str, str] | None],
    ) -> None:
        self.keyed = keyed
        self.records = records


class NodeRecords:
    """One node's records: its values, the key of each, the raw result item
    each value came from, and (once written) each one's digest and value
    digest, per round it was stored at."""

    def __init__(
        self,
        *,
        keyed: bool,
        values: list[Any],
        keys: list[str | None] | None,
        items: list[Any] | None = None,
        entries: list[RecordEntry] | None = None,
    ) -> None:
        self.keyed = keyed
        self.values = values
        self.keys = keys
        self.items: list[Any] = [] if items is None else items
        self.digests: dict[tuple[str | None, int], tuple[str, str | None]] = {}
        self.entries: list[RecordEntry] = entries if entries is not None else []
        """The entries this node's records were read from, when no task wrote
        them: a reuse plan's served entries, listed in the job's index."""

    def value_at(self, position: int) -> Any:
        """The value of the record at ``position``; a node's unkeyed record is
        its whole value list."""
        return self.values[position] if self.keyed else self.values

    def identity_at(self, position: int, round_number: int) -> tuple[str, str] | None:
        """The record at ``position`` as a consumer at ``round_number`` names
        it: its two digests, or ``None`` when the row has no record. A node
        stored only at another round is read at the lowest round it has."""
        if not self.keyed:
            found = self._digest_at(None, round_number)
            if found is not None:
                digest, value = found
                return digest, value or ""
            value_digest = digest_of(self.values)
            return digest_of({"values": self.values}), value_digest
        key = record_key_at(self.keys, position)
        if key is None:
            return None
        found = self._digest_at(key, round_number)
        if found is not None:
            digest, value = found
            return digest, value or ""
        raise ValueError(
            f"record {key!r} at position {position} has no signature yet: every keyed"
            " node is written before its consumers"
        )

    def _digest_at(
        self, record: str | None, round_number: int
    ) -> tuple[str, str | None] | None:
        """One record's digest and value digest at a round, else at the lowest
        round it was stored at."""
        if (record, round_number) in self.digests:
            return self.digests[record, round_number]
        rounds = sorted(round_ for key, round_ in self.digests if key == record)
        if not rounds:
            return None
        return self.digests[record, rounds[0]]


class RecordPlan:
    """The per-node facts a signature needs, resolved once per batch."""

    def __init__(
        self,
        *,
        op: RuntimeOp,
        op_name: str,
        key_path: str | Sequence[str] | None,
        logical_names: Mapping[str, str],
    ) -> None:
        self.op = op
        self.op_name = op_name
        self.key_path = key_path
        self.spec_digest = op_spec_digest(op, logical_names)

    @property
    def keyed(self) -> bool:
        return self.key_path is not None


class RecordWriter:
    """Writes one batch's records once."""

    def __init__(
        self,
        *,
        storage: JobStorage,
        enabled: bool,
        logger: Logger,
    ) -> None:
        self._storage = storage
        self._enabled = enabled
        self.logger = logger

    async def write(
        self,
        *,
        job_id: str,
        member_job_ids: list[str],
        round_number: int,
        runtime_graph: RuntimeGraph,
        node_task_map: Mapping[str, str],
        results: Mapping[str, Any],
        member_store: Mapping[str, bool] | None = None,
        member_rounds: Mapping[str, int] | None = None,
        node_owners: Mapping[str, set[str]] | None = None,
        served: Mapping[str, NodeRecords] | None = None,
    ) -> int:
        """Write the records of every node this batch produced and return how many.

        ``served`` holds the records a reuse plan answered without a task; they
        enter each storing member's index too."""
        if not self._enabled:
            return 0
        order = runtime_graph.topological_order()
        plans = _resolve_plans(runtime_graph, order)
        records = _collect_node_records(
            runtime_graph, plans, order, node_task_map, results, served, self.logger
        )
        store = {} if member_store is None else member_store
        rounds_of = {} if member_rounds is None else member_rounds
        owners_of = {} if node_owners is None else node_owners
        unversioned: set[str] = set()
        written: list[tuple[RecordEntry, list[tuple[str, int]]]] = []
        for node_id in order:
            node_records = records.get(node_id)
            if node_records is None:
                continue
            task_id = node_task_map.get(node_id)
            if task_id is None:
                continue
            plan = plans[node_id]
            storing = _storing_owners(
                node_id, member_job_ids, store=store, owners=owners_of
            )
            if not storing:
                continue
            if reads_unversioned(plan) or any(
                node in unversioned for node in input_nodes(plan)
            ):
                unversioned.add(node_id)
            versions = EffectVersions(unversioned=node_id in unversioned)
            targets = sorted(
                (member, rounds_of.get(member, round_number)) for member in storing
            )
            for member_round, members in _members_by_round(targets):
                source = RecordSource(
                    job=job_id,
                    round=member_round,
                    task_id=task_id,
                    member_jobs=member_job_ids,
                )
                written.extend(
                    (entry, [(member, member_round) for member in members])
                    for entry in self._node_entries(
                        plan=plan,
                        node_records=node_records,
                        records=records,
                        inputs=_record_inputs(plan, records, member_round),
                        versions=versions,
                        source=source,
                    )
                )
        entries = [entry for entry, _ in written]
        if entries:
            await asyncio.gather(
                *(asyncio.to_thread(self._save_entry_once, entry) for entry in entries)
            )
            indexed = {member for _, targets in written for member, _ in targets}
            for member in sorted(indexed):
                await self._write_index(
                    member,
                    [
                        (entry, member_round)
                        for entry, targets in written
                        for target_member, member_round in targets
                        if target_member == member
                    ],
                )
            self.logger.info(
                "Record store wrote %d record(s) for job(s) %s round %d",
                len(entries),
                ", ".join(sorted(indexed)),
                round_number,
            )
        await self._index_served(
            member_job_ids,
            served or {},
            store=store,
            rounds=rounds_of,
            owners=owners_of,
        )
        return len(entries)

    async def _index_served(
        self,
        member_job_ids: list[str],
        served: Mapping[str, NodeRecords],
        *,
        store: Mapping[str, bool],
        rounds: Mapping[str, int],
        owners: Mapping[str, set[str]],
    ) -> None:
        """List the entries a reuse plan served in each storing member's index."""
        for node_id, node_records in served.items():
            if not node_records.entries:
                continue
            members = _storing_owners(
                node_id, member_job_ids, store=store, owners=owners
            )
            for member in members:
                await self._write_index(
                    member,
                    [
                        (entry, rounds.get(member, entry.address.round))
                        for entry in node_records.entries
                    ],
                )

    def _save_entry_once(self, entry: RecordEntry) -> None:
        """Write an entry unless its object already exists. Best effort: a
        concurrent writer of the same digest writes the same body."""
        if read_entry(self._storage, entry.digest) is not None:
            return
        self._storage.save_blob(
            entry_object_name(entry.digest),
            entry.body(),
            "application/json",
        )

    def _node_entries(
        self,
        *,
        plan: RecordPlan,
        node_records: NodeRecords,
        records: Mapping[str, NodeRecords],
        inputs: Sequence[RecordInput],
        versions: EffectVersions,
        source: RecordSource,
    ) -> list[RecordEntry]:
        """One entry per record of this node: one per key, or one whole-op."""
        positions = range(len(node_records.keys or [])) if plan.keyed else range(1)
        entries: list[RecordEntry] = []
        for position in positions:
            record = record_key_at(node_records.keys, position) if plan.keyed else None
            if plan.keyed and record is None:
                continue
            address = RecordAddress(op=plan.op_name, record=record, round=source.round)
            value = node_records.value_at(position)
            failed = is_marker(value) or is_marker(_decoded(value))
            if failed and not is_marker(value):
                value = _decoded(value)
            value_hash = None if failed else digest_of(value)
            input_digests, input_value_digests = position_inputs(
                inputs, position, keyed=plan.keyed
            )
            draw = realization(
                plan,
                node_records,
                records,
                position,
                source,
                unversioned=versions.unversioned,
            )
            signature = failed_record_digest if failed else entry_digest
            digest = signature(
                spec_digest=plan.spec_digest,
                address=address,
                input_digests=input_digests,
                input_values=input_value_digests,
                realization=draw,
                versions=versions,
            )
            node_records.digests[record, source.round] = (digest, value_hash)
            entries.append(
                RecordEntry(
                    digest=digest,
                    address=address,
                    status="failed" if failed else "ok",
                    value=redact_sensitive(value),
                    value_digest=value_hash,
                    input_digests=input_digests,
                    input_values=input_value_digests,
                    realization=draw,
                    versions=versions,
                    source=source,
                    written_at=time.time(),
                )
            )
        return entries

    async def _write_index(
        self, job_id: str, additions: Sequence[tuple[RecordEntry, int]]
    ) -> None:
        """Add one batch's entries to one job's index object in one save, one
        batch of the job at a time."""
        async with _INDEX_LOCKS.setdefault(job_id, asyncio.Lock()):
            known = await asyncio.to_thread(read_index, self._storage, job_id)
            listed: dict[tuple[str, str | None, int], str] = (
                {
                    (known_entry.op, known_entry.record, known_entry.round): (
                        known_entry.digest
                    )
                    for known_entry in known.entries
                }
                if known is not None
                else {}
            )
            for entry, member_round in additions:
                listed[(entry.address.op, entry.address.record, member_round)] = (
                    entry.digest
                )
            index = RecordIndex(
                job=job_id,
                entries=[
                    RecordIndexEntry(op=op, record=record, round=round_, digest=digest)
                    for (op, record, round_), digest in sorted(listed.items())
                ],
            )
            await asyncio.to_thread(
                self._storage.save_blob,
                index_object_name(job_id),
                index.body(),
                "application/json",
            )


def _decoded(value: Any) -> Any:
    """A task's output can reach the writer as its JSON text; a marker written
    that way is still a marker."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def record_key_at(keys: list[str | None] | None, position: int) -> str | None:
    """The key of the record at ``position``."""
    if keys is None:
        return None
    return keys[position]


def _storing_owners(
    node_id: str,
    member_job_ids: list[str],
    *,
    store: Mapping[str, bool],
    owners: Mapping[str, set[str]],
) -> list[str]:
    """The members that own ``node_id`` and asked for the store; an unknown
    owner stores for every member."""
    owning = owners.get(node_id)
    candidates = (
        member_job_ids
        if owning is None
        else [member for member in member_job_ids if member in owning]
    )
    if not store:
        return candidates
    return [member for member in candidates if store.get(member, False)]


def _members_by_round(
    targets: Sequence[tuple[str, int]],
) -> list[tuple[int, list[str]]]:
    """The members that store a node, grouped by the round each stores it at,
    in round order. Rounds are part of a record's address, so members that
    run the node at different rounds get one entry each."""
    grouped: dict[int, list[str]] = {}
    for member, member_round in targets:
        grouped.setdefault(member_round, []).append(member)
    return sorted((member_round, members) for member_round, members in grouped.items())


def position_inputs(
    inputs: Sequence[RecordInput], position: int, *, keyed: bool
) -> tuple[list[str], list[str]]:
    """What one record read, in its inputs' declared order: at its position for
    a keyed input, or every record of one too short for it."""
    digests: list[str] = []
    values: list[str] = []

    def take(found: tuple[str, str] | None) -> None:
        if found is None:
            return
        digests.append(found[0])
        values.append(found[1])

    for input_node in inputs:
        if keyed and input_node.keyed and position < len(input_node.records):
            take(input_node.records[position])
            continue
        for found in input_node.records:
            take(found)
    return digests, values


def _record_inputs(
    plan: RecordPlan, records: Mapping[str, NodeRecords], round_number: int
) -> list[RecordInput]:
    """The inputs a node reads, in its declared order, at ``round_number``, each
    literal rows list included at the row's position."""
    inputs: list[RecordInput] = []
    for node in input_nodes(plan):
        produced = records.get(node)
        if produced is not None:
            positions = range(len(produced.values)) if produced.keyed else range(1)
            inputs.append(
                RecordInput(
                    keyed=produced.keyed,
                    records=[
                        produced.identity_at(position, round_number)
                        for position in positions
                    ],
                )
            )
            continue
        values = input_values(plan.op, node)
        inputs.append(
            RecordInput(
                keyed=False,
                records=[
                    (digest_of({"node": node, "values": values}), digest_of(values))
                ],
            )
        )
    for rows in literal_rows(plan):
        inputs.append(
            RecordInput(
                keyed=True,
                records=[
                    (digest_of({"values": value}), digest_of(value)) for value in rows
                ],
            )
        )
    return inputs


def literal_rows(plan: RecordPlan) -> list[list[Any]]:
    """Every literal rows list the op carries: its python plan's literal
    entries, then the lists its data spec holds outside a plan."""
    return [entry["values"] for entry in _literal_entries(plan)] + literal_lists(
        plan.op
    )


def _literal_entries(plan: RecordPlan) -> list[Mapping[str, Any]]:
    """The plan entries that inline literal values a python step reads."""
    entries = data_spec_dict(plan.op).get("plan")
    if not isinstance(entries, list):
        return []
    return [
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("kind") == "literal"
        and isinstance(entry.get("values"), list)
    ]


def input_nodes(plan: RecordPlan) -> list[str]:
    """The nodes a node reads, in its declared order."""
    return list(plan.op.dependencies)


def reads_unversioned(plan: RecordPlan) -> bool:
    """A retrieval or fetch reads a source that records no snapshot, so its
    value is not reusable and neither is anything computed from it."""
    return plan.op.task_type in _UNVERSIONED_TASKS


def realization(
    plan: RecordPlan,
    node_records: NodeRecords,
    records: Mapping[str, NodeRecords],
    position: int,
    source: RecordSource,
    *,
    unversioned: bool,
) -> RecordRealization:
    """The draw a record used: its per-row seed row, else its op's seed. A
    record no snapshot fixes names the production that drew it."""
    seed = _seed_at(plan, node_records, records, position)
    if unversioned:
        return _draw_unique_to(source, seed)
    if seed is not None:
        return RecordRealization(seed=seed)
    if samples(plan.op):
        return _draw_unique_to(source)
    return RecordRealization()


def _seed_at(
    plan: RecordPlan,
    node_records: NodeRecords,
    records: Mapping[str, NodeRecords],
    position: int,
) -> int | None:
    """The seed the record at ``position`` was drawn with, if the op binds one;
    a node-bound column is read from that node's item at the same position."""
    seeds = data_spec_dict(plan.op).get("seeds")
    if isinstance(seeds, Mapping) and plan.keyed:
        node = seeds.get("node")
        path = seeds.get("path", "")
        if not isinstance(node, str) or not isinstance(path, str):
            return None
        source = records.get(node)
        if source is None or position >= len(source.items):
            return None
        seed = path_value(source.items[position], item_relative_path(path))
        if isinstance(seed, bool) or not isinstance(seed, int):
            return None
        return seed
    if isinstance(seeds, list) and plan.keyed:
        rows = len(node_records.keys or [])
        if len(seeds) != rows:
            raise ValueError(
                f"op '{plan.op_name}' declares {len(seeds)} per-row seed(s) for {rows}"
                " row(s)"
            )
        seed = seeds[position]
        if isinstance(seed, bool) or not isinstance(seed, int):
            return None
        return seed
    return declared_seed(plan.op)


def _draw_unique_to(source: RecordSource, seed: int | None = None) -> RecordRealization:
    """A draw nothing reproduces: it names the production that made it, and
    keeps the seed it was asked for as information only."""
    return RecordRealization(seed=seed, production=f"{source.job}/{source.task_id}")


def _resolve_plans(
    runtime_graph: RuntimeGraph, order: Sequence[str]
) -> dict[str, RecordPlan]:
    """Resolve every node's plan, deriving a 1:1 node's key from its keyed
    input in dependency order."""
    names = logical_names(runtime_graph)
    plans: dict[str, RecordPlan] = {}
    for node_id in order:
        op = runtime_graph.nodes[node_id]
        plans[node_id] = RecordPlan(
            op=op,
            op_name=names.get(node_id, node_id),
            key_path=key_path(runtime_graph, node_id, plans),
            logical_names=names,
        )
    return plans


def logical_names(runtime_graph: RuntimeGraph) -> dict[str, str]:
    """Each runtime node's op name, so no record address or spec digest carries
    a job id, a prefix, a slice suffix or a graph scope.

    A node takes the name its workflow declares; a node whose op the workflow
    did not name (an implicit step, a lowering helper) takes its DSL op id."""
    names: dict[str, str] = {}
    for op_name, node_ids in runtime_graph.dsl_to_runtime.items():
        for node_id in node_ids:
            names[node_id] = runtime_graph.op_names.get(node_id, op_name)
    return names


def key_path(
    runtime_graph: RuntimeGraph, node_id: str, plans: Mapping[str, RecordPlan]
) -> str | Sequence[str] | None:
    """A node's record-key path or paths: the ones its op declares, else the
    path of the single keyed input a 1:1 node reads by position, else ``None``
    (whole-op)."""
    declared = runtime_graph.record_keys.get(node_id)
    if declared is not None:
        return declared.path
    source = _keyed_input(runtime_graph, node_id, plans)
    if source is None:
        return None
    return plans[source].key_path


def _one_to_one(op: RuntimeOp) -> bool:
    """Whether the op maps each input row to exactly one output row."""
    mode = data_spec_dict(op).get("mode")
    if op.task_type == _LAMBDA_TASK_TYPE:
        return mode in (None, "row", "parse")
    return mode not in {"list", "reduce", "pairs", "regroup"}


def _keyed_input(
    runtime_graph: RuntimeGraph, node_id: str, plans: Mapping[str, RecordPlan]
) -> str | None:
    """The single keyed node a 1:1 node reads, if it reads exactly one; a
    dependency this batch did not plan is a served record."""
    op = runtime_graph.nodes[node_id]
    if not _one_to_one(op):
        return None
    keyed = [node for node in op.dependencies if node in plans and plans[node].keyed]
    if len(keyed) != 1:
        return None
    return keyed[0]


def input_values(op: RuntimeOp, node: str) -> Any:
    """The spec side of an input read from a node that produced no stored
    records (a literal, an input op, a node outside the stored graph)."""
    spec = data_spec_dict(op)
    plan = spec.get("plan")
    if isinstance(plan, list):
        for entry in plan:
            if isinstance(entry, Mapping) and entry.get("node") == node:
                return entry.get("values", entry.get("path"))
    for key in ("items", "seeds"):
        value = spec.get(key)
        if isinstance(value, Mapping) and value.get("node") == node:
            return value.get("path")
    return None


def _collect_node_records(
    runtime_graph: RuntimeGraph,
    plans: Mapping[str, RecordPlan],
    order: Sequence[str],
    node_task_map: Mapping[str, str],
    results: Mapping[str, Any],
    served: Mapping[str, NodeRecords] | None,
    logger: Logger,
) -> dict[str, NodeRecords]:
    """Read every finished node's records in the order it produced them, and the
    records a reuse plan served, so a keyed node reads a served input's keys."""
    produced: dict[str, list[Any]] = {}
    raw_items: dict[str, list[Any]] = {}
    for node_id, task_id in node_task_map.items():
        response = results.get(task_id)
        if response is None or plans.get(node_id) is None:
            continue
        found = _record_values(response, plans[node_id])
        produced[node_id] = [value for value, _ in found]
        raw_items[node_id] = [item for _, item in found]
    records: dict[str, NodeRecords] = {
        node_id: node_records
        for node_id, node_records in (served or {}).items()
        if node_id not in produced
    }
    for node_id, values in produced.items():
        plan = plans[node_id]
        key_path = plan.key_path
        keys = (
            None
            if key_path is None
            else _record_keys(
                runtime_graph, plan, key_path, values, produced, records, logger
            )
        )
        records[node_id] = NodeRecords(
            keyed=plan.keyed,
            values=values,
            keys=keys,
            items=raw_items[node_id],
        )
    _inherit_input_keys(order, plans, runtime_graph, records)
    return records


def _record_keys(
    runtime_graph: RuntimeGraph,
    plan: RecordPlan,
    key_path: str | Sequence[str],
    values: Sequence[Any],
    produced: Mapping[str, Sequence[Any]],
    records: Mapping[str, NodeRecords],
    logger: Logger,
) -> list[str | None]:
    """One key per row of one keyed node, in row order; a one-to-one op reads a
    missing key from the node it declares, at the same position."""
    own = [row_key(value, key_path) for value in values]
    declared = runtime_graph.record_keys.get(plan.op.node_id)
    if declared is None or declared.node == plan.op.node_id:
        return own
    if not _one_to_one(plan.op):
        return own
    source = _declared_keys(plan, declared, len(values), produced, records, logger)
    if source is None:
        return own
    return [
        source[position] if key is None and position < len(source) else key
        for position, key in enumerate(own)
    ]


def _declared_keys(
    plan: RecordPlan,
    declared: RecordKey,
    count: int,
    produced: Mapping[str, Sequence[Any]],
    records: Mapping[str, NodeRecords],
    logger: Logger,
) -> Sequence[str | None] | None:
    """The key each row of a declared input carries at the path the op declares,
    read from its rows: the ones it produced, the values a reuse plan served for
    it, or its literal rows. A served node answers with its own keys only when
    the op declares no path, so a node keyed by several paths keeps its keys."""
    rows = produced.get(declared.node)
    if rows is not None:
        return [row_key(row, declared.path) for row in rows]
    served = records.get(declared.node)
    if served is not None:
        if not declared.path:
            return served.keys
        return [row_key(value, declared.path) for value in served.values]
    literal = _declared_rows(plan, declared.path, count, logger)
    return None if literal is None else [row_key(row, declared.path) for row in literal]


def _declared_rows(
    plan: RecordPlan, path: str | Sequence[str], count: int, logger: Logger
) -> Sequence[Any] | None:
    """The literal list the op carries for a declared input, when exactly one of
    its lists holds ``count`` rows that all carry a key at ``path``."""
    matching = [
        rows
        for rows in literal_rows(plan)
        if len(rows) == count and all(row_key(row, path) is not None for row in rows)
    ]
    if len(matching) > 1:
        logger.warning(
            "Op '%s' carries %d literal lists that could hold the rows its key names;"
            " the record takes no key from any of them",
            plan.op_name,
            len(matching),
        )
    return matching[0] if len(matching) == 1 else None


def _inherit_input_keys(
    order: Sequence[str],
    plans: Mapping[str, RecordPlan],
    runtime_graph: RuntimeGraph,
    records: dict[str, NodeRecords],
) -> None:
    """Fill each None key of a 1:1 node from its keyed input at the same position."""
    for node_id in order:
        node_records = records.get(node_id)
        source = _keyed_input(runtime_graph, node_id, plans)
        if node_records is None or node_records.keys is None or source is None:
            continue
        inherited = records.get(source)
        if inherited is None or inherited.keys is None:
            continue
        if len(inherited.keys) != len(node_records.keys):
            continue
        declared = runtime_graph.record_keys.get(node_id) is not None
        for position, key in enumerate(node_records.keys):
            value = node_records.values[position]
            marker = is_marker(value) or is_marker(_decoded(value))
            if key is None and (marker or not declared):
                node_records.keys[position] = inherited.keys[position]


def _record_values(response: Any, plan: RecordPlan) -> list[tuple[Any, Any]]:
    """Each value one task produced, in order, with its own result item."""
    payload = response
    if isinstance(payload, Mapping) and isinstance(payload.get("value"), Mapping):
        payload = payload["value"]
    if not isinstance(payload, Mapping):
        return []
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    typed = typed_row_step(plan.op)
    found: list[tuple[Any, Any]] = []
    for item in items:
        rows = item.get("rows") if isinstance(item, Mapping) else None
        if isinstance(rows, list):
            found.extend((_row_value(row), row) for row in rows)
        elif _grouped_model_output(item, plan):
            found.extend((text, item) for text in _item_output(item, typed))
        else:
            found.append((_item_output(item, typed), item))
    return found


def _grouped_model_output(item: Any, plan: RecordPlan) -> bool:
    """Whether an item holds a rowwise model task's per-row outputs as one list."""
    return (
        grouped_local_chat(plan.op)
        and isinstance(item, Mapping)
        and isinstance(item.get("output"), list)
    )


def _row_value(row: Any) -> Any:
    """The value one grouped API row stands for: its message content (even
    when null), else its text, else the row itself."""
    if not isinstance(row, Mapping):
        return row
    message = None
    try:
        message = row["json"]["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        message = None
    if isinstance(message, Mapping):
        return message.get("content")
    text = row.get("text")
    return row if text is None else text


def typed_row_step(op: RuntimeOp) -> bool:
    """Whether ``op`` emits one item per row with the typed value under ``value``."""
    spec = op.data_spec
    return (
        op.task_type == _LAMBDA_TASK_TYPE
        and isinstance(spec, PythonData)
        and spec.mode in (None, "row", "parse")
    )


def grouped_local_chat(op: RuntimeOp) -> bool:
    """Whether ``op`` is a local rowwise chat that groups every row's reply into
    one item."""
    return op.task_type == "inference" and isinstance(op.data_spec, DataFrame)


def _item_output(item: Any, typed: bool) -> Any:
    """The record value an item carries: a row step's typed value, else its output."""
    if typed and isinstance(item, Mapping) and "value" in item:
        return item["value"]
    if isinstance(item, Mapping) and "output" in item:
        return item["output"]
    return item


def read_index(storage: JobStorage, job_id: str) -> RecordIndex | None:
    """A job's record index, or ``None`` when the job wrote no records."""
    try:
        body, _ = storage.load_blob(index_object_name(job_id))
    except (ArchiveNotFound, KeyError):
        return None

    try:
        return RecordIndex.from_body(body)
    except ValueError:
        return None


def read_entry(storage: JobStorage, digest: str) -> RecordEntry | None:
    """An entry object, trusted only when its recorded digest equals its key."""
    try:
        body, _ = storage.load_blob(entry_object_name(digest))
    except (ArchiveNotFound, KeyError):
        return None
    try:
        entry = RecordEntry.from_body(body)
    except ValueError:
        return None
    return entry if entry.digest == digest else None


def lineage(
    storage: JobStorage,
    *,
    index: RecordIndex,
    op: str,
    record: str | None,
    round_number: int,
) -> list[RecordEntry]:
    """A record's chain: itself, then each input digest it names, level by
    level, in their defined order."""
    found = index.find(op=op, record=record, round_number=round_number)
    chain: list[RecordEntry] = []
    seen: set[str] = set()

    def visit(digest: str) -> None:
        if digest in seen:
            return
        seen.add(digest)
        entry = read_entry(storage, digest)
        if entry is None:
            return
        chain.append(entry)
        for parent in entry.input_digests:
            visit(parent)

    if found is not None:
        visit(found.digest)
    return chain
