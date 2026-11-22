"""Per-record store writer and lineage read.

The writer runs once per execution batch, after the batch's tasks finished. It
reads each task's raw FlowMesh response and, for every node it stores, writes
one entry per record: one per keyed row for an op that declares or inherits a
record key, or one entry covering the whole op otherwise. Entries are written
once, each as a single object, under ``records/<digest>.json``; the batch then
writes one index object listing the job's entries, so a later read needs only
the archive.

A record key is declared once, on the op or input that carries it
(``key: {node, path}``), and derived downstream: a row-mode Lambda and a 1:1
model op inherit their single keyed input's key by position. A list-mode op, a
reduce or an op with no keyed input is one whole-op record.

An entry names the signature digest of every record it read: a keyed record
names the record at its own position, and a whole-op record names every record
of every input in order. Lineage then walks a final record back through every
record and value it depended on. Per-row data enters as those inputs, and a
per-row seed enters as that record's realization; the op spec digest covers only
the computation, so two slices of one op store the same records under the same
keys.

Writing is off unless the job asked for it, and an archive failure never fails a
job whose values were already computed and returned. A stored value is redacted
the way the per-task artifact is, while its value digest stays over the raw
value the job computed.
"""

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

from lumilake.log import Logger
from lumilake_server.runtime.record_store import (
    EffectVersions,
    RecordAddress,
    RecordEntry,
    RecordIndex,
    RecordIndexEntry,
    RecordRealization,
    RecordSource,
    declared_seed,
    digest_of,
    entry_digest,
    entry_object_name,
    failed_record_digest,
    index_object_name,
    is_marker,
    op_spec_digest,
    row_key,
    samples,
)
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp
from lumilake_server.runtime.sensitive import redact_sensitive
from lumilake_server.utils.job_storage import ArchiveNotFound, JobStorage

_LAMBDA_TASK_TYPE = "python"
"""The task type a Lambda op lowers to."""

_UNVERSIONED_TASKS = frozenset({"data_retrieval", "api"})
"""Task types that read a source no snapshot records."""


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
    """One node's records: its values, the key of each, and (once written)
    each one's digest and value digest."""

    def __init__(
        self,
        *,
        keyed: bool,
        values: list[Any],
        keys: list[str | None] | None,
    ) -> None:
        self.keyed = keyed
        self.values = values
        self.keys = keys
        self.digests: dict[str | None, tuple[str, str | None]] = {}

    def value_at(self, position: int) -> Any:
        """The value of the record at ``position``; a node's unkeyed record is
        its whole value list."""
        return self.values[position] if self.keyed else self.values

    def identity_at(self, position: int) -> tuple[str, str] | None:
        """The record at ``position`` as a consumer names it: its signature
        digest and its value digest, or ``None`` when the row has no record."""
        if not self.keyed:
            if None in self.digests:
                digest, value = self.digests[None]
                return digest, value or ""
            value_digest = digest_of(self.values)
            return digest_of({"values": self.values}), value_digest
        key = record_key_at(self.keys, position)
        if key is None:
            return None
        if key in self.digests:
            digest, value = self.digests[key]
            return digest, value or ""
        raise ValueError(
            f"record {key!r} at position {position} has no signature yet: every"
            " keyed node is written before its consumers"
        )


class RecordPlan:
    """The per-node facts a signature needs, resolved once per batch."""

    def __init__(
        self,
        *,
        op: RuntimeOp,
        op_name: str,
        key_path: str | None,
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
        member_job_ids: Sequence[str],
        round_number: int,
        runtime_graph: RuntimeGraph,
        node_task_map: Mapping[str, str],
        task_worker: Mapping[str, str | None],
        results: Mapping[str, Any],
        member_store: Mapping[str, bool] | None = None,
        member_rounds: Mapping[str, int] | None = None,
        node_owners: Mapping[str, set[str]] | None = None,
    ) -> int:
        """Write the records of every node this batch produced.

        ``results`` maps a task id to its fetched FlowMesh response. Returns the
        number of entries written.

        A batch may hold several jobs. Each entry is written under the round of
        the member jobs that own its node, and only a job that asked for the
        store takes the entry into its index: a batch's first job's flag never
        speaks for the others."""
        if not self._enabled or not results:
            return 0
        order = runtime_graph.topological_order()
        plans = _resolve_plans(runtime_graph, order)
        records = _collect_node_records(
            runtime_graph, plans, order, node_task_map, results
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
            inputs = _record_inputs(plan, records)
            if _reads_unversioned(plan) or any(
                node in unversioned for node in _input_nodes(plan)
            ):
                unversioned.add(node_id)
            versions = EffectVersions(unversioned=node_id in unversioned)
            source = RecordSource(
                job=job_id,
                round=max(rounds_of.get(member, round_number) for member in storing),
                task_id=task_id,
                worker=task_worker.get(task_id),
                member_jobs=list(member_job_ids),
            )
            targets = sorted(
                (member, rounds_of.get(member, round_number)) for member in storing
            )
            written.extend(
                (entry, targets)
                for entry in self._node_entries(
                    plan=plan,
                    node_records=node_records,
                    inputs=inputs,
                    versions=versions,
                    source=source,
                )
            )
        entries = [entry for entry, _ in written]
        for entry in entries:
            await asyncio.to_thread(
                self._save_entry_once,
                entry,
            )
        if entries:
            indexed = {member for _, targets in written for member, _ in targets}
            for entry, targets in written:
                for member, member_round in targets:
                    await self._write_index(member, entry, member_round)
            self.logger.info(
                "Record store wrote %d record(s) for job(s) %s round %d",
                len(entries),
                ", ".join(sorted(indexed)),
                round_number,
            )
        return len(entries)

    def _save_entry_once(self, entry: RecordEntry) -> None:
        """Write an entry unless its object already exists.

        Objects are content-addressed by the decisions that produce the value,
        so an object already there is the same record, and the first job to
        produce it keeps the stored source. Written once is best effort: the
        read-then-write is not atomic, so two writers of one digest can both
        write and the last wins. Both bodies carry the same digest, so a reader
        stays correct either way."""
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
            input_digests, input_values = _position_inputs(
                inputs, position, keyed=plan.keyed
            )
            realization = _realization(
                plan,
                node_records,
                position,
                source,
                unversioned=versions.unversioned,
            )
            signature = failed_record_digest if failed else entry_digest
            digest = signature(
                spec_digest=plan.spec_digest,
                address=address,
                input_digests=input_digests,
                input_values=input_values,
                realization=realization,
                versions=versions,
            )
            node_records.digests[record] = (digest, value_hash)
            entries.append(
                RecordEntry(
                    digest=digest,
                    address=address,
                    status="failed" if failed else "ok",
                    value=redact_sensitive(value),
                    value_digest=value_hash,
                    input_digests=input_digests,
                    input_values=input_values,
                    realization=realization,
                    versions=versions,
                    source=source,
                    written_at=time.time(),
                )
            )
        return entries

    async def _write_index(
        self, job_id: str, entry: RecordEntry, member_round: int
    ) -> None:
        """Add one entry to one job's index object, at the round that job
        runs the record's op in."""
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
        listed[(entry.address.op, entry.address.record, member_round)] = entry.digest
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
    member_job_ids: Sequence[str],
    *,
    store: Mapping[str, bool],
    owners: Mapping[str, set[str]],
) -> list[str]:
    """The members that own ``node_id`` and asked for the store.

    A node whose owning members are unknown belongs to every member, so a caller
    that names no owners stores for all of them."""
    owning = owners.get(node_id)
    candidates = (
        list(member_job_ids)
        if owning is None
        else [member for member in member_job_ids if member in owning]
    )
    if not store:
        return candidates
    return [member for member in candidates if store.get(member, False)]


def _position_inputs(
    inputs: Sequence[RecordInput], position: int, *, keyed: bool
) -> tuple[list[str], list[str]]:
    """What one record read, in its inputs' declared order.

    A keyed record reads one record of a keyed input, the one at its own
    position, and every record of an input that carries none. An input too
    short for that position (a broadcast one-row literal, a shorter keyed
    input) contributes its whole list rather than failing the write. A whole-op
    record reads every record of every input, keyed or not, in order, so a
    change in any input row moves its signature. The digests are ordered by the
    node's declared inputs, then by position, so a reader that probed before
    running computes the same signature."""
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
    plan: RecordPlan, records: Mapping[str, NodeRecords]
) -> list[RecordInput]:
    """The inputs a node reads, in its declared order.

    An input this batch did not store (a node outside the stored graph) is
    named by the spec side it declares. A literal input is one dependency the
    op never declares, because it reaches the task through the column plan
    instead, and it is named by its own value at this row's position, so a
    changed literal moves that record's signature."""
    inputs: list[RecordInput] = []
    for node in _input_nodes(plan):
        produced = records.get(node)
        if produced is not None:
            positions = range(len(produced.values)) if produced.keyed else range(1)
            inputs.append(
                RecordInput(
                    keyed=produced.keyed,
                    records=[produced.identity_at(position) for position in positions],
                )
            )
            continue
        values = _input_values(plan.op, node)
        inputs.append(
            RecordInput(
                keyed=False,
                records=[
                    (digest_of({"node": node, "values": values}), digest_of(values))
                ],
            )
        )
    for entry in _literal_entries(plan):
        inputs.append(
            RecordInput(
                keyed=True,
                records=[
                    (digest_of({"values": value}), digest_of(value))
                    for value in entry.get("values", [])
                ],
            )
        )
    return inputs


def _literal_entries(plan: RecordPlan) -> list[Mapping[str, Any]]:
    """The plan entries that inline literal values a python step reads."""
    entries = plan.op.data_spec.get("plan")
    if not isinstance(entries, list):
        return []
    return [
        entry
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("kind") == "literal"
    ]


def _input_nodes(plan: RecordPlan) -> list[str]:
    """The nodes a node reads, in its declared order."""
    return list(plan.op.dependencies)


def _reads_unversioned(plan: RecordPlan) -> bool:
    """A retrieval or fetch reads a source that records no snapshot, so its
    value is not reusable and neither is anything computed from it."""
    return plan.op.task_type in _UNVERSIONED_TASKS


def _realization(
    plan: RecordPlan,
    node_records: NodeRecords,
    position: int,
    source: RecordSource,
    *,
    unversioned: bool,
) -> RecordRealization:
    """The draw a record used: its own seed row when the op declares per-row
    seeds, else the op's declared seed.

    A record whose value no snapshot fixes (an unversioned read) names its own
    production instead, so it never shares a key with another run's draw: the
    source does not promise the same value twice even under one seed. Only a
    versioned op's seed decides the realization."""
    seed = _seed_at(plan, node_records, position)
    if unversioned:
        return _draw_unique_to(source, seed)
    if seed is not None:
        return RecordRealization(seed=seed)
    if samples(plan.op):
        return _draw_unique_to(source)
    return RecordRealization()


def _seed_at(plan: RecordPlan, node_records: NodeRecords, position: int) -> int | None:
    """The seed the record at ``position`` was drawn with, if the op binds one."""
    seeds = plan.op.data_spec.get("seeds")
    if isinstance(seeds, list) and plan.keyed:
        keyed_rows = sum(1 for key in node_records.keys or [] if key is not None)
        if len(seeds) != keyed_rows:
            raise ValueError(
                f"op '{plan.op_name}' declares {len(seeds)} per-row seed(s) for"
                f" {keyed_rows} keyed row(s)"
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
    names = _logical_names(runtime_graph)
    plans: dict[str, RecordPlan] = {}
    for node_id in order:
        op = runtime_graph.nodes[node_id]
        plans[node_id] = RecordPlan(
            op=op,
            op_name=names.get(node_id, node_id),
            key_path=_key_path(runtime_graph, node_id, plans),
            logical_names=names,
        )
    return plans


def _logical_names(runtime_graph: RuntimeGraph) -> dict[str, str]:
    """Each runtime node's op name, so no record address or spec digest carries
    a job id, a prefix, a slice suffix or a graph scope.

    A node takes the name its workflow declares; a node whose op the workflow
    did not name (an implicit step, a lowering helper) takes its DSL op id."""
    names: dict[str, str] = {}
    for op_name, node_ids in runtime_graph.dsl_to_runtime.items():
        for node_id in node_ids:
            names[node_id] = runtime_graph.op_names.get(node_id, op_name)
    return names


def _key_path(
    runtime_graph: RuntimeGraph, node_id: str, plans: Mapping[str, RecordPlan]
) -> str | None:
    """A node's record-key path: the one its op declares, else the path of the
    single keyed input a 1:1 node reads by position, else ``None`` (whole-op)."""
    declared = runtime_graph.record_keys.get(node_id)
    if declared is not None:
        return declared["path"]
    source = _keyed_input(runtime_graph, node_id, plans)
    if source is None:
        return None
    return plans[source].key_path


def _one_to_one(op: RuntimeOp) -> bool:
    """Whether the op maps each input row to exactly one output row: a
    row-mode Lambda, or a scalar model op. A list-mode or reduce op consumes
    its whole input and is one whole-op record."""
    if op.task_type == _LAMBDA_TASK_TYPE:
        return op.data_spec.get("mode", "row") == "row"
    return op.data_spec.get("mode") not in {"list", "reduce", "pairs", "regroup"}


def _keyed_input(
    runtime_graph: RuntimeGraph, node_id: str, plans: Mapping[str, RecordPlan]
) -> str | None:
    """The single keyed node a 1:1 node reads, if it reads exactly one."""
    op = runtime_graph.nodes[node_id]
    if not _one_to_one(op):
        return None
    keyed = [node for node in op.dependencies if plans[node].keyed]
    if len(keyed) != 1:
        return None
    return keyed[0]


def _input_values(op: RuntimeOp, node: str) -> Any:
    """The spec side of an input read from a node that produced no stored
    records (a literal, an input op, a node outside the stored graph)."""
    plan = op.data_spec.get("plan")
    if isinstance(plan, list):
        for entry in plan:
            if isinstance(entry, Mapping) and entry.get("node") == node:
                return entry.get("values", entry.get("path"))
    for key in ("items", "seeds"):
        value = op.data_spec.get(key)
        if isinstance(value, Mapping) and value.get("node") == node:
            return value.get("path")
    return None


def _collect_node_records(
    runtime_graph: RuntimeGraph,
    plans: Mapping[str, RecordPlan],
    order: Sequence[str],
    node_task_map: Mapping[str, str],
    results: Mapping[str, Any],
) -> dict[str, NodeRecords]:
    """Read every finished node's records in the order it produced them."""
    records: dict[str, NodeRecords] = {}
    for node_id, task_id in node_task_map.items():
        response = results.get(task_id)
        plan = plans.get(node_id)
        if response is None or plan is None:
            continue
        values = _record_values(response)
        keys = None
        if plan.key_path is not None:
            keys = [row_key(value, plan.key_path) for value in values]
        records[node_id] = NodeRecords(
            keyed=plan.keyed,
            values=values,
            keys=keys,
        )
    _inherit_marker_keys(order, plans, runtime_graph, records)
    return records


def _inherit_marker_keys(
    order: Sequence[str],
    plans: Mapping[str, RecordPlan],
    runtime_graph: RuntimeGraph,
    records: dict[str, NodeRecords],
) -> None:
    """A failed record keeps its address: a marker carries no key of its own,
    so a keyed node's marker takes the key of the record it replaces, at the
    same position of the node's one keyed input."""
    for node_id in order:
        node_records = records.get(node_id)
        source = _keyed_input(runtime_graph, node_id, plans)
        inherited = None if source is None else records.get(source)
        if node_records is None or node_records.keys is None or inherited is None:
            continue
        for position, value in enumerate(node_records.values):
            if node_records.keys[position] is not None:
                continue
            if not (is_marker(value) or is_marker(_decoded(value))):
                continue
            if inherited.keys is not None and position < len(inherited.keys):
                node_records.keys[position] = inherited.keys[position]


def _record_values(response: Any) -> list[Any]:
    """The values one task produced, in order."""
    payload = response
    if isinstance(payload, Mapping) and isinstance(payload.get("value"), Mapping):
        payload = payload["value"]
    if isinstance(payload, Mapping):
        items = payload.get("items")
        if isinstance(items, list):
            return [_item_output(item) for item in items]
    return []


def _item_output(item: Any) -> Any:
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
