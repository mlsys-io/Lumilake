"""Per-record store writer and lineage read.

The writer runs once per execution batch, after the batch's tasks finished. It
reads each task's raw FlowMesh response and, for every node it stores, writes
one entry per record: one per keyed row for an op that declares a record key,
or one entry covering the whole op otherwise. Entries are written once, each as
a single object, under ``records/<digest>.json``.

An entry names the signature digests of the records it read, so lineage walks a
final record back through every record and value it depended on. A keyed input
contributes one digest per record, ordered by key; an unkeyed input contributes
the digest of its single whole-op entry. Physical node ids, batch ids, workers
and row positions never enter a digest, so slicing and reordering a job's rows
leaves its records unchanged.

Writing is off unless the job asked for it, and a write failure never fails a
job whose values were already computed and returned.
"""

import asyncio
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

from lumilake.log import Logger, LogLevel, init_child_logger
from lumilake_server.runtime.record_store import (
    EffectVersions,
    RecordAddress,
    RecordEntry,
    RecordRealization,
    RecordSource,
    declared_seed,
    digest_of,
    entry_digest,
    entry_object_name,
    failed_record_digest,
    is_marker,
    op_spec_digest,
    row_key,
)
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp
from lumilake_server.utils.job_storage import ArchiveNotFound, JobStorage


class RecordStore:
    """The entries this server wrote, kept for a job's lineage read."""

    def __init__(self, logger: Logger | None = None, log_level: LogLevel | None = None):
        self.logger = init_child_logger("RecordStore", logger, log_level)
        self._lock = threading.Lock()
        self._records: dict[str, list[RecordEntry]] = {}

    def register(self, entry: RecordEntry) -> None:
        """Keep an entry under every job that shared the task, and the one that
        owns it, so a lineage read finds it from either."""
        jobs = [entry.source.job, *entry.source.member_jobs]
        with self._lock:
            for job_id in dict.fromkeys(jobs):
                self._records.setdefault(job_id, []).append(entry)

    def entries_for(self, job_id: str) -> list[RecordEntry]:
        with self._lock:
            return list(self._records.get(job_id, []))

    def job_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._records)

    def rounds_for(self, job_id: str) -> list[int]:
        with self._lock:
            rounds = {entry.address.round for entry in self._records.get(job_id, [])}
        return sorted(rounds)

    def forget(self, job_id: str) -> None:
        with self._lock:
            self._records.pop(job_id, None)


class NodeRecords:
    """One node's stored records: the values a consumer can read, and their
    signature digests once written."""

    def __init__(
        self,
        *,
        op_name: str,
        keyed: bool,
        values: list[Any],
        keys: list[str | None] | None,
    ) -> None:
        self.op_name = op_name
        self.keyed = keyed
        self.values = values
        self.keys = keys
        self.digests: list[tuple[str | None, str, str | None]] = []

    def record(self, digest: str, record: str | None, value_digest: str | None) -> None:
        self.digests.append((record, digest, value_digest))

    def inputs(self) -> list[tuple[str, str]]:
        """What a consumer names for this node, in its defined order: a keyed
        node's records in key order, an unkeyed node's single entry."""
        order = sorted(self.digests, key=lambda item: item[0] or "")
        return [(digest, value or "") for _, digest, value in order]


class RecordPlan:
    """The per-node facts a signature needs, resolved once per batch."""

    def __init__(
        self,
        *,
        op: RuntimeOp,
        op_name: str,
        input_nodes: Sequence[str],
        input_names: Sequence[str],
        key_path: str | None,
    ) -> None:
        self.op = op
        self.op_name = op_name
        self.input_nodes = list(input_nodes)
        self.input_names = list(input_names)
        self.key_path = key_path
        self.spec_digest = op_spec_digest(
            op, dict(zip(self.input_nodes, self.input_names))
        )

    @property
    def keyed(self) -> bool:
        return self.key_path is not None


class RecordWriter:
    """Writes one batch's records once."""

    def __init__(
        self,
        *,
        storage: JobStorage,
        store: RecordStore,
        enabled: bool,
        logger: Logger,
    ) -> None:
        self._storage = storage
        self._store = store
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
    ) -> int:
        """Write the records of every finished keyed-descendant node.

        ``results`` maps a task id to its fetched FlowMesh response. Returns the
        number of entries written.
        """
        if not self._enabled or not results:
            return 0
        plans = _resolve_plans(runtime_graph)
        records = _collect_node_records(runtime_graph, plans, node_task_map, results)
        unversioned: set[str] = set()
        written = 0
        for node_id in runtime_graph.topological_order():
            node_records = records.get(node_id)
            plan = plans.get(node_id)
            if node_records is None or plan is None:
                continue
            task_id = node_task_map.get(node_id)
            if task_id is None:
                continue
            source = RecordSource(
                job=job_id,
                round=round_number,
                task_id=task_id,
                worker=task_worker.get(task_id),
                member_jobs=list(member_job_ids),
            )
            inputs = _read_inputs(plan, records)
            if _reads_unversioned(plan) or any(
                node in unversioned for node in plan.input_nodes
            ):
                unversioned.add(node_id)
            written += await self._write_node(
                plan=plan,
                node_records=node_records,
                inputs=inputs,
                realization=RecordRealization(seed=declared_seed(plan.op)),
                versions=EffectVersions(unversioned=node_id in unversioned),
                source=source,
            )
        if written:
            self.logger.info(
                "Record store wrote %d record(s) for job %s round %d",
                written,
                job_id,
                round_number,
            )
        return written

    async def _write_node(
        self,
        *,
        plan: RecordPlan,
        node_records: NodeRecords,
        inputs: list[tuple[str, str]],
        realization: RecordRealization,
        versions: EffectVersions,
        source: RecordSource,
    ) -> int:
        pairs: list[tuple[str | None, Any]]
        if plan.keyed:
            pairs = [
                (record, value)
                for record, value in zip(node_records.keys or [], node_records.values)
                if record is not None
            ]
        else:
            pairs = [(None, list(node_records.values))]
        for record, value in pairs:
            address = RecordAddress(op=plan.op_name, record=record, round=source.round)
            failed = is_marker(value)
            value_hash = None if failed else digest_of(value)
            digest = (
                failed_record_digest(spec_digest=plan.spec_digest, address=address)
                if failed
                else entry_digest(
                    spec_digest=plan.spec_digest,
                    address=address,
                    input_digests=[digest for digest, _ in inputs],
                    input_values=[value for _, value in inputs],
                    realization=realization,
                    versions=versions,
                )
            )
            entry = RecordEntry(
                digest=digest,
                address=address,
                status="failed" if failed else "ok",
                value=value,
                value_digest=value_hash,
                input_digests=[digest for digest, _ in inputs],
                input_values=[value for _, value in inputs],
                realization=realization,
                versions=versions,
                source=source,
                written_at=time.time(),
            )
            await asyncio.to_thread(
                self._storage.save_blob,
                entry_object_name(digest),
                entry.body(),
                "application/json",
            )
            self._store.register(entry)
            node_records.record(digest, record, value_hash)
        return len(pairs)


def _collect_node_records(
    runtime_graph: RuntimeGraph,
    plans: Mapping[str, RecordPlan],
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
        keys = (
            [row_key(value, plan.key_path) for value in values]
            if plan.key_path is not None
            else None
        )
        records[node_id] = NodeRecords(
            op_name=plan.op_name,
            keyed=plan.keyed,
            values=values,
            keys=keys,
        )
    return records


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


def _read_inputs(
    plan: RecordPlan, records: Mapping[str, NodeRecords]
) -> list[tuple[str, str]]:
    """What this node read, in its defined order: each input's signature digest
    and the value digest it returned.

    A node whose records this batch did not store contributes its spec and the
    values it read instead, so a changed literal still moves the signature."""
    inputs: list[tuple[str, str]] = []
    for node in plan.input_nodes:
        produced = records.get(node)
        if produced is not None:
            inputs.extend(produced.inputs())
            continue
        values = _input_values(plan.op, node)
        inputs.append(
            (
                digest_of({"node": node}),
                digest_of({"node": node, "values": values}),
            )
        )
    for name, values in _literal_inputs(plan.op):
        inputs.append((digest_of({"literal": name}), digest_of({name: values})))
    return inputs


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


def _literal_inputs(op: RuntimeOp) -> list[tuple[str, list[Any]]]:
    """The op's literal input lists, one per key, in a stable order."""
    literals: list[tuple[str, list[Any]]] = []
    plan = op.data_spec.get("plan")
    if isinstance(plan, list):
        for index, entry in enumerate(plan):
            if not isinstance(entry, Mapping) or entry.get("kind") != "literal":
                continue
            values = entry.get("values")
            if isinstance(values, list):
                literals.append((f"plan[{index}]", list(values)))
    for key in ("items", "seeds"):
        values = op.data_spec.get(key)
        if isinstance(values, list):
            literals.append((key, list(values)))
    return sorted(literals, key=lambda item: item[0])


def _reads_unversioned(plan: RecordPlan) -> bool:
    """A retrieval or fetch reads a source that records no snapshot, so its
    value is not reusable and neither is anything computed from it."""
    return plan.op.task_type in {"data_retrieval", "api"}


def _resolve_plans(runtime_graph: RuntimeGraph) -> dict[str, RecordPlan]:
    names = _logical_names(runtime_graph)
    plans: dict[str, RecordPlan] = {}
    for node_id in runtime_graph.node_order:
        node = runtime_graph.nodes[node_id]
        key = runtime_graph.record_keys.get(node_id)
        plans[node_id] = RecordPlan(
            op=node,
            op_name=_op_name(node_id, names),
            input_nodes=node.dependencies,
            input_names=[_op_name(dep, names) for dep in node.dependencies],
            key_path=None if key is None else key["path"],
        )
    return plans


def _logical_names(runtime_graph: RuntimeGraph) -> dict[str, str]:
    names: dict[str, str] = {}
    for op_name, node_ids in runtime_graph.dsl_to_runtime.items():
        for node_id in node_ids:
            names[node_id] = op_name
    return names


def _op_name(node_id: str, names: Mapping[str, str]) -> str:
    return names.get(node_id, node_id)


def lineage(
    storage: JobStorage,
    *,
    job_id: str,
    op: str,
    record: str | None,
    round_number: int,
    entries: Sequence[RecordEntry],
) -> list[RecordEntry]:
    """A record's chain: itself, then each input digest it names, level by
    level, in their defined order."""
    head = _find_entry(
        entries, job_id=job_id, op=op, record=record, round_number=round_number
    )
    chain: list[RecordEntry] = []
    seen: set[str] = set()

    def visit(entry: RecordEntry) -> None:
        if entry.digest in seen:
            return
        seen.add(entry.digest)
        chain.append(entry)
        for digest in entry.input_digests:
            parent = _load(storage, digest)
            if parent is not None:
                visit(parent)

    if head is not None:
        visit(head)
    return chain


def _find_entry(
    entries: Sequence[RecordEntry],
    *,
    job_id: str,
    op: str,
    record: str | None,
    round_number: int,
) -> RecordEntry | None:
    for entry in entries:
        if (
            entry.source.job == job_id
            and entry.address.op == op
            and entry.address.record == record
            and entry.address.round == round_number
        ):
            return entry
    return None


def _load(storage: JobStorage, digest: str) -> RecordEntry | None:
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
