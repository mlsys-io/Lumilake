"""Per-record store writer and lineage read.

The writer runs once per execution batch and, for every node it stores, writes
one entry per record: one per keyed row, or one covering the whole op, each
under ``records/<digest>.json``, then one index object per member job. An entry
names the signature digest of every record it read, so lineage walks a final
record back through what it depended on. An archive failure never fails a job
whose values were returned, and a stored value is redacted.
"""

import asyncio
import json
import time
from collections.abc import Collection, Mapping, Sequence
from typing import Any
from weakref import WeakValueDictionary

from lumilake.log import Logger
from lumilake_server.data_specs import (
    ColumnPlanEntry,
    JoinDescription,
    PythonData,
)
from lumilake_server.runtime.lambda_task_helpers import serialized_table, table_rows
from lumilake_server.runtime.record_keys import (
    child_key,
    composite_key,
    join_key_paths,
    join_row_key_paths,
)
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
    key_text,
    literal_lists,
    op_spec_digest,
    path_value,
    samples,
)
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_graph.outputs import item_relative_path
from lumilake_server.runtime.runtime_ops import RuntimeOp
from lumilake_server.runtime.sensitive import redact_sensitive
from lumilake_server.utils.job_storage import ArchiveNotFound, JobStorage
from lumilake_server.utils.retrieval_key import retrieval_key_column

_LAMBDA_TASK_TYPE = "python"
"""The task type a Lambda, join or parse op lowers to."""

_UNVERSIONED_TASKS = frozenset({"data_retrieval", "api"})
"""Task types that read a source no snapshot records."""

_CONTENT_ADDRESSED_TASKS = frozenset({"data_retrieval"})
"""Unversioned task types whose value is a read rather than a draw."""


def reads_unversioned(op: RuntimeOp) -> bool:
    """Whether the op reads a source no snapshot records."""
    return op.task_type in _UNVERSIONED_TASKS


def content_addressed(op: RuntimeOp) -> bool:
    """Whether the op's unversioned value can be re-read and keyed on."""
    return op.task_type in _CONTENT_ADDRESSED_TASKS


_KEYED_MODES = frozenset({"expand", "reduce", "join"})
"""Python modes that key their own records: expand children, reduce groups and
join rows."""

_JOIN_MODE = "join"

_INDEX_LOCKS: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
"""One lock per job index, so batches that write one job's index take turns."""


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
    each one's digest and value digest, per round it was stored at."""

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
        """The entries a reuse plan served, listed in the job's index."""

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

    def identity_of(self, key: str | None, round_number: int) -> tuple[str, str] | None:
        """The record a consumer names by key, whatever position it was stored at."""
        found = self._digest_at(key, round_number)
        if found is not None:
            digest, value = found
            return digest, value or ""
        return None

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
        key_path: str | None,
        key_paths: list[str],
        row_key_paths: list[str],
        join_inputs: list[str | None],
        join_attaches: list[tuple[str | None, str | None, bool]],
        column: str | None,
        groups_keyed: bool,
        expand_source: str | None,
        inherits_key: str | None,
        logical_names: Mapping[str, str],
    ) -> None:
        self.op = op
        self.op_name = op_name
        self.key_path = key_path
        self.key_paths = key_paths
        self.row_key_paths = row_key_paths
        self.join_inputs = join_inputs
        self.join_attaches = join_attaches
        self.column = column
        self.groups_keyed = groups_keyed
        self.expand_source = expand_source
        self.inherits_key = inherits_key
        self.spec_digest = op_spec_digest(op, logical_names)

    @property
    def keyed(self) -> bool:
        return (
            self.key_path is not None
            or self.column is not None
            or bool(self.key_paths)
            or bool(self.row_key_paths)
            or self.groups_keyed
            or self.expand_source is not None
            or self.inherits_key is not None
        )

    @property
    def mode(self) -> str:
        """The python step's mode; a model op or an unset mode runs row by row."""
        spec = self.op.data_spec
        return spec.mode or "row" if isinstance(spec, PythonData) else "row"


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
        task_worker: Mapping[str, str | None],
        results: Mapping[str, Any],
        member_store: Mapping[str, bool] | None = None,
        member_rounds: Mapping[str, int] | None = None,
        node_owners: Mapping[str, set[str]] | None = None,
        served: Mapping[str, NodeRecords] | None = None,
        content_nodes: Collection[str] = (),
    ) -> int:
        """Write the records of every node this batch produced and return how many.

        ``served`` holds the records a reuse plan answered without a task;
        they enter each storing member's index too.
        """
        if not self._enabled:
            return 0
        order = runtime_graph.topological_order()
        plans = _resolve_plans(runtime_graph, order)
        records, items_of = _collect_node_records(
            runtime_graph, plans, order, node_task_map, results
        )
        for node_id, served_records_of_node in (served or {}).items():
            records.setdefault(node_id, served_records_of_node)
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
            if reads_unversioned(plan.op):
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
                    worker=task_worker.get(task_id),
                    member_jobs=list(member_job_ids),
                )
                written.extend(
                    (entry, [(member, member_round) for member in members])
                    for entry in self._node_entries(
                        plan=plan,
                        node_records=node_records,
                        records=records,
                        inputs=record_inputs(
                            plan, records, member_round, plans, content_nodes
                        ),
                        items=items_of.get(node_id, []),
                        versions=versions,
                        source=source,
                        round_number=member_round,
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
        items: Sequence[Mapping[str, Any]],
        versions: EffectVersions,
        source: RecordSource,
        round_number: int,
    ) -> list[RecordEntry]:
        """One entry per record of this node: one per key, or one whole-op."""
        if plan.mode == "expand" and (expand_source := plan.expand_source) is not None:
            return self._child_entries(
                plan=plan,
                node_records=node_records,
                records=records,
                inputs=inputs,
                items=items,
                parent=records.get(expand_source),
                versions=versions,
                source=source,
            )
        if plan.mode == "reduce":
            return self._group_entries(
                plan=plan,
                node_records=node_records,
                items=items,
                groups=_first_input(plan, records),
                inputs=inputs,
                versions=versions,
                source=source,
            )
        if plan.mode == _JOIN_MODE:
            return self._join_entries(
                plan=plan,
                node_records=node_records,
                records=records,
                versions=versions,
                source=source,
                round_number=round_number,
            )
        positions = range(len(node_records.keys or [])) if plan.keyed else range(1)
        entries: list[RecordEntry] = []
        for position in positions:
            record = record_key_at(node_records.keys, position) if plan.keyed else None
            if plan.keyed and record is None:
                continue
            value = node_records.value_at(position)
            value, failed = _record_value(value)
            value_hash = None if failed else digest_of(value)
            row_versions = _row_versions(
                versions,
                value_hash=value_hash,
                failed=failed,
            )
            input_digests, input_values = _position_inputs(
                inputs, position, keyed=plan.keyed
            )
            entries.append(
                self._entry(
                    plan=plan,
                    record=record,
                    value=value,
                    value_hash=value_hash,
                    failed=failed,
                    input_digests=input_digests,
                    input_values=input_values,
                    realization=_realization(
                        plan,
                        node_records,
                        records,
                        position,
                        source,
                        unversioned=row_versions.unversioned,
                    ),
                    versions=row_versions,
                    source=source,
                    node_records=node_records,
                )
            )
        return entries

    def _child_entries(
        self,
        *,
        plan: RecordPlan,
        node_records: NodeRecords,
        records: Mapping[str, NodeRecords],
        inputs: Sequence[RecordInput],
        items: Sequence[Mapping[str, Any]],
        parent: NodeRecords | None,
        versions: EffectVersions,
        source: RecordSource,
    ) -> list[RecordEntry]:
        """One entry per expand child, keyed by its parent's key and its index
        among that parent's children, as the step reports them."""
        entries: list[RecordEntry] = []
        for item in items:
            parent_index = _index_field(item, "parent")
            parent_key = (
                None if parent is None else record_key_at(parent.keys, parent_index)
            )
            if parent_key is None:
                continue
            value = item.get("output")
            value, failed = _record_value(value)
            value_hash = None if failed else digest_of(value)
            row_versions = _row_versions(
                versions,
                value_hash=value_hash,
                failed=failed,
            )
            input_digests, input_values = _position_inputs(
                inputs, parent_index, keyed=True
            )
            entries.append(
                self._entry(
                    plan=plan,
                    record=child_key(parent_key, _index_field(item, "child")),
                    value=value,
                    value_hash=value_hash,
                    failed=failed,
                    input_digests=input_digests,
                    input_values=input_values,
                    realization=_realization(
                        plan,
                        node_records,
                        records,
                        parent_index,
                        source,
                        unversioned=row_versions.unversioned,
                    ),
                    versions=row_versions,
                    source=source,
                    node_records=node_records,
                )
            )
        return entries

    def _group_entries(
        self,
        *,
        plan: RecordPlan,
        node_records: NodeRecords,
        items: Sequence[Mapping[str, Any]],
        groups: NodeRecords | None,
        inputs: Sequence[RecordInput],
        versions: EffectVersions,
        source: RecordSource,
    ) -> list[RecordEntry]:
        """One entry per reduce group, keyed by its ``by`` value and listing
        the ordered keys and digests of the members it read."""
        keyed, whole = _split_inputs(inputs)
        entries: list[RecordEntry] = []
        for item in items:
            record = _group_key(item, groups)
            if record is None:
                continue
            wanted = item.get("members")
            member_keys: list[str] = []
            input_digests: list[str] = []
            input_values: list[str] = []
            for member in wanted if isinstance(wanted, list) else []:
                if not isinstance(member, int):
                    continue
                if groups is not None:
                    key = record_key_at(groups.keys, member)
                    member_keys.append(
                        key if key is not None else digest_of(groups.values[member])
                    )
                read_digests, read_values = _position_inputs(keyed, member, keyed=True)
                input_digests.extend(read_digests)
                input_values.extend(read_values)
            whole_digests, whole_values = _position_inputs(whole, 0, keyed=False)
            input_digests.extend(whole_digests)
            input_values.extend(whole_values)
            value = item.get("output")
            value, failed = _record_value(value)
            value_hash = None if failed else digest_of(value)
            row_versions = _row_versions(
                versions,
                value_hash=value_hash,
                failed=failed,
            )
            entries.append(
                self._entry(
                    plan=plan,
                    record=record,
                    value=value,
                    value_hash=value_hash,
                    failed=failed,
                    input_digests=input_digests,
                    input_values=input_values,
                    realization=RecordRealization(),
                    versions=row_versions,
                    source=source,
                    member_keys=member_keys,
                    node_records=node_records,
                )
            )
        return entries

    def _join_entries(
        self,
        *,
        plan: RecordPlan,
        node_records: NodeRecords,
        records: Mapping[str, NodeRecords],
        versions: EffectVersions,
        source: RecordSource,
        round_number: int,
    ) -> list[RecordEntry]:
        """One entry per join output row, under the key it was given when the
        node's records were keyed."""
        indexes = _join_indexes(plan, records)
        entries: list[RecordEntry] = []
        for position, unfailed in enumerate(node_records.values):
            record = None if node_records.keys is None else node_records.keys[position]
            if record is None:
                continue
            value, failed = _record_value(unfailed)
            value_hash = None if failed else digest_of(value)
            row_versions = _row_versions(
                versions,
                value_hash=value_hash,
                failed=failed,
            )
            input_digests, input_values = _join_inputs(
                plan, records, indexes, position, round_number, value
            )
            entries.append(
                self._entry(
                    plan=plan,
                    record=record,
                    value=value,
                    value_hash=value_hash,
                    failed=failed,
                    input_digests=input_digests,
                    input_values=input_values,
                    realization=RecordRealization(),
                    versions=row_versions,
                    source=source,
                    node_records=node_records,
                )
            )
        return entries

    def _entry(
        self,
        *,
        plan: RecordPlan,
        record: str | None,
        value: Any,
        value_hash: str | None,
        failed: bool,
        input_digests: list[str],
        input_values: list[str],
        realization: RecordRealization,
        versions: EffectVersions,
        source: RecordSource,
        node_records: NodeRecords | None = None,
        member_keys: list[str] | None = None,
    ) -> RecordEntry:
        """One record's entry, at the signature its inputs decide."""
        address = RecordAddress(op=plan.op_name, record=record, round=source.round)
        signature = failed_record_digest if failed else entry_digest
        digest = signature(
            spec_digest=plan.spec_digest,
            address=address,
            input_digests=input_digests,
            input_values=input_values,
            realization=realization,
            versions=versions,
        )
        if node_records is not None:
            _register(node_records, plan, record, digest, value_hash, source.round)
        return RecordEntry(
            digest=digest,
            address=address,
            status="failed" if failed else "ok",
            value=redact_sensitive(value),
            value_digest=value_hash,
            input_digests=input_digests,
            input_values=input_values,
            member_keys=member_keys or [],
            realization=realization,
            versions=versions,
            source=source,
            written_at=time.time(),
        )

    async def _write_index(
        self, job_id: str, additions: Sequence[tuple[RecordEntry, int]]
    ) -> None:
        """Add one batch's entries to one job's index object in one save."""
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


def _register(
    node_records: NodeRecords,
    plan: RecordPlan,
    record: str | None,
    digest: str,
    value_hash: str | None,
    round_number: int,
) -> None:
    """Register one record's digest in this batch, refusing two values that
    share an address: keys ``1`` and ``"1"`` write the same key text."""
    known = node_records.digests.get((record, round_number))
    if known is not None and known[0] != digest:
        raise ValueError(
            f"op '{plan.op_name}' has two records under key {record!r} at round"
            f" {round_number}"
        )
    node_records.digests[record, round_number] = (digest, value_hash)


def _record_value(value: Any) -> tuple[Any, bool]:
    """One record's value and whether it is a failed record's marker, decoded
    when the task delivered it as JSON text."""
    if is_marker(value):
        return value, True
    decoded = _decoded(value)
    if is_marker(decoded):
        return decoded, True
    return value, False


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
    """The members that own ``node_id`` and asked for the store; an unknown
    owner stores for every member."""
    owning = owners.get(node_id)
    candidates = (
        list(member_job_ids)
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


def _node_items(response: Any) -> list[Mapping[str, Any]]:
    """The items one task returned, as objects, with each one's own fields
    (an expand's ``parent`` and ``child``, a reduce's ``members``) kept."""
    payload = response
    if isinstance(payload, Mapping) and isinstance(payload.get("value"), Mapping):
        payload = payload["value"]
    if not isinstance(payload, Mapping):
        return []
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, Mapping)]


def _index_field(item: Mapping[str, Any], field: str) -> int:
    """An expand item's ``parent`` or ``child`` index; a missing one raises."""
    value = item.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"expand item has no {field} index: {item!r}")
    return value


def _group_key(item: Mapping[str, Any], groups: NodeRecords | None) -> str | None:
    """A reduce item's group key: its ``by`` value, or, for the group of one row
    whose ``by`` could not be read, that row's key or its value digest."""
    value = item.get("key")
    if value is not None:
        return key_text(value)
    members = item.get("members")
    if groups is None or not isinstance(members, list) or len(members) != 1:
        return None
    member = members[0]
    if not isinstance(member, int):
        return None
    key = record_key_at(groups.keys, member)
    return key if key is not None else digest_of(groups.values[member])


def _first_input(
    plan: RecordPlan, records: Mapping[str, NodeRecords]
) -> NodeRecords | None:
    """The op's first input's records, or ``None`` with none stored."""
    nodes = plan.join_inputs or _input_nodes(plan)
    first = nodes[0] if nodes else None
    return None if first is None else records.get(first)


def _entry_node(entries: Sequence[ColumnPlanEntry], index: int) -> str | None:
    """The node one plan index reads, or ``None`` for a literal entry."""
    return entries[index].node if 0 <= index < len(entries) else None


def _join(op: RuntimeOp) -> JoinDescription | None:
    """The join description a python step's spec carries, when it is a join."""
    spec = op.data_spec
    return spec.join if isinstance(spec, PythonData) else None


def join_input_nodes(op: RuntimeOp) -> list[str | None]:
    """A join's inputs in ``JoinDescription.inputs`` order, ``None`` for a literal."""
    spec = op.data_spec
    if not isinstance(spec, PythonData) or spec.join is None:
        return []
    return [_entry_node(spec.plan, index) for index in spec.join.inputs]


def join_attach_nodes(op: RuntimeOp) -> list[tuple[str | None, str | None, bool]]:
    """Each join attach's value and source node, and whether it is the first input."""
    spec = op.data_spec
    if not isinstance(spec, PythonData) or spec.join is None:
        return []
    return [
        (
            _entry_node(spec.plan, value_index),
            _entry_node(spec.plan, source_index),
            first,
        )
        for _, value_index, source_index, first in spec.join.attach
    ]


def position_inputs(
    inputs: Sequence[RecordInput], position: int, *, keyed: bool
) -> tuple[list[str], list[str]]:
    """What one record read, in its inputs' declared order."""
    return _position_inputs(inputs, position, keyed=keyed)


def input_nodes(plan: RecordPlan) -> Sequence[str]:
    """The nodes a node reads, in its declared order."""
    return _input_nodes(plan)


def input_values(op: RuntimeOp, node: str) -> Any:
    """The spec side of an input read from a node that produced no stored records."""
    return _input_values(op, node)


def literal_rows(plan: RecordPlan) -> list[list[Any]]:
    """Every literal rows list the op carries."""
    return _literal_rows(plan)


def logical_names(runtime_graph: RuntimeGraph) -> dict[str, str]:
    """Each runtime node's op name."""
    return _logical_names(runtime_graph)


def key_path(
    runtime_graph: RuntimeGraph, node_id: str, plans: Mapping[str, RecordPlan]
) -> str | None:
    """A node's record-key path."""
    return _key_path(runtime_graph, node_id, plans)


def realization(
    plan: RecordPlan,
    node_records: NodeRecords,
    records: Mapping[str, NodeRecords],
    position: int,
    source: RecordSource,
    *,
    unversioned: bool,
) -> RecordRealization:
    """The draw a record used."""
    return _realization(
        plan, node_records, records, position, source, unversioned=unversioned
    )


def _position_inputs(
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


def _split_inputs(
    inputs: Sequence[RecordInput],
) -> tuple[list[RecordInput], list[RecordInput]]:
    """A reduce's inputs split into the ones it reads one row of per member and
    the ones it reads whole."""
    keyed = [input_node for input_node in inputs if input_node.keyed]
    whole = [input_node for input_node in inputs if not input_node.keyed]
    return keyed, whole


def content_value(value: Any) -> Any:
    """What an unversioned read's record is addressed by: the returned values."""
    if isinstance(value, Mapping):
        table = value.get("table")
        if isinstance(table, str):
            return {"rows": _decoded_table(table)}
        if isinstance(table, Mapping) and isinstance(table.get("df"), str):
            return {"rows": _decoded_table(table["df"])}
    if isinstance(value, list):
        return [content_value(item) for item in value]
    return value


def _decoded_table(table: str) -> Any:
    try:
        return json.loads(table)
    except ValueError:
        return table


def content_input(produced: NodeRecords) -> RecordInput:
    """One input named by each record's value, in record order."""
    positions = range(len(produced.values)) if produced.keyed else range(1)
    records: list[tuple[str, str] | None] = []
    for position in positions:
        value = content_value(produced.value_at(position))
        records.append((digest_of({"value": value}), digest_of(value)))
    return RecordInput(keyed=produced.keyed, records=records)


def record_inputs(
    plan: RecordPlan,
    records: Mapping[str, NodeRecords],
    round_number: int,
    plans: Mapping[str, RecordPlan],
    content_nodes: Collection[str] = (),
) -> list[RecordInput]:
    """The inputs a node reads at ``round_number``, in its declared order."""
    return _record_inputs(plan, records, round_number, plans, content_nodes)


def _record_inputs(
    plan: RecordPlan,
    records: Mapping[str, NodeRecords],
    round_number: int,
    plans: Mapping[str, RecordPlan],
    content_nodes: Collection[str] = (),
) -> list[RecordInput]:
    """The inputs a node reads, in its declared order, at ``round_number``, each
    literal rows list included at the row's position."""
    inputs: list[RecordInput] = []
    for node in _input_nodes(plan):
        produced = records.get(node)
        if produced is not None:
            addressing = (
                content_addressed(plans[node].op)
                if node in plans
                else node in content_nodes
            )
            if addressing:
                inputs.append(content_input(produced))
                continue
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
        values = _input_values(plan.op, node)
        inputs.append(
            RecordInput(
                keyed=False,
                records=[
                    (digest_of({"node": node, "values": values}), digest_of(values))
                ],
            )
        )
    for rows in _literal_rows(plan):
        inputs.append(
            RecordInput(
                keyed=True,
                records=[
                    (digest_of({"values": value}), digest_of(value)) for value in rows
                ],
            )
        )
    return inputs


def _join_indexes(
    plan: RecordPlan, records: Mapping[str, NodeRecords]
) -> dict[str, dict[str, int]]:
    """For each keyed further input, the position of the first row carrying each
    ``by`` key."""
    indexes: dict[str, dict[str, int]] = {}
    for node in _join_looked_up_nodes(plan):
        produced = records.get(node)
        if produced is None or not produced.keyed:
            continue
        found: dict[str, int] = {}
        for position, row in enumerate(produced.values):
            key = composite_key(row, plan.key_paths)
            if key is not None and key not in found:
                found[key] = position
        indexes[node] = found
    return indexes


def _join_looked_up_nodes(plan: RecordPlan) -> list[str]:
    """The keyed inputs a join looks a row up in: further inputs and attach sources."""
    nodes = plan.join_inputs or _input_nodes(plan)
    sources = [source for _, source, _ in plan.join_attaches]
    return [
        node
        for node in [*(nodes[1:]), *sources]
        if node is not None and node not in nodes[:1]
    ]


def _join_inputs(
    plan: RecordPlan,
    records: Mapping[str, NodeRecords],
    indexes: Mapping[str, Mapping[str, int]],
    position: int,
    round_number: int,
    value: Any,
) -> tuple[list[str], list[str]]:
    """What one join row read, its first input by position, the rest by their own."""
    nodes = plan.join_inputs or _input_nodes(plan)
    key = composite_key(value, plan.key_paths) if plan.key_paths else None
    digests: list[str] = []
    values: list[str] = []
    found = _join_input_records(
        plan, records, indexes, nodes, position, key, round_number
    )
    for record in found:
        digests.append(record[0])
        values.append(record[1])
    for pair in _attach_records(plan, records, indexes, position, key, round_number):
        digests.append(pair[0])
        values.append(pair[1])
    for row in _literal_rows_at(plan, key):
        digests.append(digest_of({"values": [row]}))
        values.append(digest_of(row))
    return digests, values


def _join_input_records(
    plan: RecordPlan,
    records: Mapping[str, NodeRecords],
    indexes: Mapping[str, Mapping[str, int]],
    nodes: Sequence[str | None],
    position: int,
    key: str | None,
    round_number: int,
) -> list[tuple[str, str]]:
    """The records one join row read from its inputs, in the join's own order."""
    found: list[tuple[str, str]] = []
    for offset, node in enumerate(nodes):
        if node is None:
            # A literal column's rows are named once, by its own list.
            continue
        produced = records.get(node)
        if produced is None:
            side = _input_values(plan.op, node)
            found.append((digest_of({"node": node, "values": side}), digest_of(side)))
            continue
        records_at = (
            [produced.identity_at(position, round_number)]
            if offset == 0 or not produced.keyed
            else _joined_records(produced, indexes.get(node, {}), key, round_number)
        )
        found.extend(record for record in records_at if record is not None)
    return found


def _attach_records(
    plan: RecordPlan,
    records: Mapping[str, NodeRecords],
    indexes: Mapping[str, Mapping[str, int]],
    position: int,
    key: str | None,
    round_number: int,
) -> list[tuple[str, str]]:
    """What one join row read through its attaches, at the source row's record."""
    found: list[tuple[str, str]] = []
    for value_node, source_node, first_source in plan.join_attaches:
        source = None if source_node is None else records.get(source_node)
        values = None if value_node is None else records.get(value_node)
        source_position = _attach_source_position(
            source, indexes.get(source_node or "", {}), position, key, first_source
        )
        if source is not None and source_position is not None:
            record = source.identity_at(source_position, round_number)
            if record is not None:
                found.append(record)
        value_position = _attach_value_position(source, values, source_position)
        if values is None or value_position is None:
            continue
        record = values.identity_at(value_position, round_number)
        if record is not None:
            found.append(record)
    return found


def _attach_source_position(
    source: NodeRecords | None,
    index: Mapping[str, int],
    position: int,
    key: str | None,
    first_source: bool,
) -> int | None:
    """The position of the source row one output row's attach came from."""
    if source is None:
        return None
    if not source.keyed:
        return 0
    if first_source:
        return position if position < len(source.values) else None
    return None if key is None else index.get(key)


def _attach_value_position(
    source: NodeRecords | None,
    values: NodeRecords | None,
    source_position: int | None,
) -> int | None:
    """The attach value paired with one source row, by position or among live rows."""
    if source is None or values is None or source_position is None:
        return None
    if not source.keyed:
        return 0
    rows = source.values
    if len(values.values) == len(rows):
        return source_position
    live = [index for index, row in enumerate(rows) if not _is_marker_row(row)]
    return None if source_position not in live else live.index(source_position)


def _is_marker_row(row: Any) -> bool:
    """Whether a stored row is a failed record's marker."""
    return is_marker(row) or is_marker(_decoded(row))


def _joined_records(
    produced: NodeRecords,
    index: Mapping[str, int],
    key: str | None,
    round_number: int,
) -> list[tuple[str, str] | None]:
    """The records a further join input contributes: the row its index holds for
    the join key, nothing for a key no row carries, or every record of an input
    whose rows do not carry those paths at all."""
    if key is None:
        return [None]
    position = index.get(key)
    if position is not None:
        return [produced.identity_at(position, round_number)]
    if index:
        return [None]
    return [
        produced.identity_at(position, round_number)
        for position in range(len(produced.values))
    ]


def _literal_rows_at(plan: RecordPlan, key: str | None) -> list[Any]:
    """The literal row each of the op's literal lists carries at ``key``."""
    if key is None:
        return []
    return [
        row
        for rows in _literal_rows(plan)
        for row in rows
        if composite_key(row, plan.key_paths) == key
    ]


def _literal_rows(plan: RecordPlan) -> list[list[Any]]:
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


def _input_nodes(plan: RecordPlan) -> Sequence[str]:
    """The nodes a node reads, in its declared order."""
    return plan.op.dependencies


def _reads_unversioned(plan: RecordPlan) -> bool:
    """A retrieval or fetch reads a source that records no snapshot, so its
    value is not reusable and neither is anything computed from it."""
    return plan.op.task_type in _UNVERSIONED_TASKS


def _row_versions(
    versions: EffectVersions, *, value_hash: str | None, failed: bool
) -> EffectVersions:
    """An unversioned op's record names its own value digest as its version."""
    if not versions.unversioned:
        return versions
    return EffectVersions(
        unversioned=True,
        value_version=None if failed else value_hash,
    )


def _realization(
    plan: RecordPlan,
    node_records: NodeRecords,
    records: Mapping[str, NodeRecords],
    position: int,
    source: RecordSource,
    *,
    unversioned: bool,
) -> RecordRealization:
    """The draw a record used: its per-row seed row, else its op's seed. A
    record no snapshot fixes names its own production, so it never shares a key
    with another run's draw."""
    seed = _seed_at(plan, node_records, records, position)
    if unversioned:
        if plan.column is not None:
            return RecordRealization()
        if seed is not None or samples(plan.op):
            return _draw_unique_to(source, seed)
        return RecordRealization()
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
    a node-bound column is read from that node's records at the same position."""
    seeds = data_spec_dict(plan.op).get("seeds")
    if isinstance(seeds, Mapping):
        node = seeds.get("node")
        if not isinstance(node, str):
            return None
        source = records.get(node)
        if source is None or position >= len(source.items):
            return None
        seed = path_value(
            source.items[position], item_relative_path(seeds.get("path", ""))
        )
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


def resolve_plans(
    runtime_graph: RuntimeGraph, order: Sequence[str]
) -> dict[str, RecordPlan]:
    """Every node's plan, in the given order."""
    return _resolve_plans(runtime_graph, order)


def _resolve_plans(
    runtime_graph: RuntimeGraph, order: Sequence[str]
) -> dict[str, RecordPlan]:
    """Resolve every node's plan, deriving a 1:1 node's key from its keyed
    input in dependency order."""
    names = _logical_names(runtime_graph)
    plans: dict[str, RecordPlan] = {}
    for node_id in order:
        op = runtime_graph.nodes[node_id]
        join = _join(op)
        plans[node_id] = RecordPlan(
            op=op,
            op_name=names.get(node_id, node_id),
            key_path=_key_path(runtime_graph, node_id, plans),
            key_paths=[] if join is None else join_key_paths(join),
            row_key_paths=[] if join is None else join_row_key_paths(join),
            join_inputs=join_input_nodes(op),
            join_attaches=join_attach_nodes(op),
            column=retrieval_key_column(op.data_spec),
            groups_keyed=_groups_keyed(op),
            expand_source=_mode_source(runtime_graph, node_id, plans),
            inherits_key=_inherits_input(runtime_graph, node_id, plans),
            logical_names=names,
        )
    return plans


def _inherits_input(
    runtime_graph: RuntimeGraph, node_id: str, plans: Mapping[str, RecordPlan]
) -> str | None:
    """The keyed input a node takes its rows' keys from by position."""
    if runtime_graph.record_keys.get(node_id) is not None:
        return None
    return _keyed_input(runtime_graph, node_id, plans)


def _groups_keyed(op: RuntimeOp) -> bool:
    """Whether a reduce groups its rows: a group is its ``by`` value, keyed
    whether or not the input it reads carries record keys."""
    return _mode_of(op) == "reduce" and bool(op.dependencies)


def _mode_of(op: RuntimeOp) -> str | None:
    """A python step's declared mode, or ``None`` for a model op."""
    spec = op.data_spec
    return spec.mode if isinstance(spec, PythonData) else None


def _logical_names(runtime_graph: RuntimeGraph) -> dict[str, str]:
    """Each runtime node's op name, so no address or spec digest carries a job
    id, a prefix, a slice suffix or a graph scope."""
    names: dict[str, str] = {}
    for op_name, node_ids in runtime_graph.dsl_to_runtime.items():
        for node_id in node_ids:
            names[node_id] = runtime_graph.op_names.get(node_id, op_name)
    return names


def _key_path(
    runtime_graph: RuntimeGraph, node_id: str, plans: Mapping[str, RecordPlan]
) -> str | None:
    """A node's record-key path: the one its op declares, else the path of the
    single keyed input a 1:1 node reads by position, else ``None`` (whole-op).
    An input keyed by a mode passes no path: its consumer takes those keys by
    position."""
    declared = runtime_graph.record_keys.get(node_id)
    if declared is not None:
        return declared.path
    source = _keyed_input(runtime_graph, node_id, plans)
    if source is None:
        return None
    if plans[source].mode in _KEYED_MODES:
        return None
    return plans[source].key_path


def _one_to_one(op: RuntimeOp) -> bool:
    """Whether the op maps each input row to exactly one output row."""
    if op.task_type == _LAMBDA_TASK_TYPE:
        return _mode_of(op) in ("row", "parse", None)
    return _mode_of(op) not in {"list", "reduce", "pairs", "regroup"}


def _keyed_input(
    runtime_graph: RuntimeGraph, node_id: str, plans: Mapping[str, RecordPlan]
) -> str | None:
    """The one keyed node a 1:1 node reads; an unplanned dependency is served."""
    op = runtime_graph.nodes[node_id]
    if not _one_to_one(op):
        return None
    keyed = [node for node in op.dependencies if node in plans and plans[node].keyed]
    if len(keyed) != 1:
        return None
    return keyed[0]


def _mode_source(
    runtime_graph: RuntimeGraph, node_id: str, plans: Mapping[str, RecordPlan]
) -> str | None:
    """The single keyed node an expand reads its input rows from."""
    op = runtime_graph.nodes[node_id]
    if _mode_of(op) != "expand":
        return None
    keyed = [node for node in op.dependencies if node in plans and plans[node].keyed]
    if len(keyed) != 1:
        return None
    return keyed[0]


def _input_values(op: RuntimeOp, node: str) -> Any:
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


def read_back_records(
    runtime_graph: RuntimeGraph,
    node_task_map: Mapping[str, str],
    results: Mapping[str, Any],
) -> dict[str, NodeRecords]:
    """Finished values keyed the way the writer keys the store's own records."""
    order = runtime_graph.topological_order()
    plans = _resolve_plans(runtime_graph, order)
    records, _ = _collect_node_records(
        runtime_graph, plans, order, node_task_map, results
    )
    return records


def _collect_node_records(
    runtime_graph: RuntimeGraph,
    plans: Mapping[str, RecordPlan],
    order: Sequence[str],
    node_task_map: Mapping[str, str],
    results: Mapping[str, Any],
) -> tuple[dict[str, NodeRecords], dict[str, list[Mapping[str, Any]]]]:
    """Read every finished node's records and items in the order it produced them."""
    produced: dict[str, list[Any]] = {}
    items_of: dict[str, list[Mapping[str, Any]]] = {}
    for node_id, task_id in node_task_map.items():
        plan = plans.get(node_id)
        response = results.get(task_id)
        if response is None or plan is None:
            continue
        items = _node_items(response)
        items_of[node_id] = items
        produced[node_id] = _record_values(response, plan, items)
    records: dict[str, NodeRecords] = {}
    for node_id in order:
        values = produced.get(node_id)
        if values is None:
            continue
        plan = plans[node_id]
        records[node_id] = NodeRecords(
            keyed=plan.keyed,
            values=values,
            keys=None,
            items=items_of.get(node_id) or [],
        )
    _keyed_mode_records(runtime_graph, plans, order, items_of, produced, records)
    _inherit_marker_keys(order, plans, runtime_graph, records)
    return records, items_of


def _keyed_mode_records(
    runtime_graph: RuntimeGraph,
    plans: Mapping[str, RecordPlan],
    order: Sequence[str],
    items_of: Mapping[str, list[Mapping[str, Any]]],
    produced: Mapping[str, Sequence[Any]],
    records: dict[str, NodeRecords],
) -> None:
    """Key every node's records in dependency order: a whole-op or 1:1 node from
    its rows, an expand child from its parent, a reduce group from its ``by``
    value and a join row from the first input's row."""
    for node_id in order:
        node_records = records.get(node_id)
        plan = plans[node_id]
        if node_records is None:
            continue
        if plan.mode == "expand":
            parent = (
                None if plan.expand_source is None else records.get(plan.expand_source)
            )
            node_records.keys = [
                _child_record_key(item, parent) for item in items_of[node_id]
            ]
            continue
        if plan.mode == "join":
            key_paths = _join_row_paths(plan)
            node_records.keys = [
                (composite_key(_item_output(item), key_paths) if key_paths else None)
                for item in items_of[node_id]
            ]
            continue
        if plan.mode == "reduce":
            groups = _first_input(plan, records)
            node_records.keys = [_group_key(item, groups) for item in items_of[node_id]]
            continue
        node_records.keys = _node_keys(
            runtime_graph, plans, plan, node_records.values, produced, records
        )
    _key_marked_join_rows(records, plans)


def _child_record_key(
    item: Mapping[str, Any], parent: NodeRecords | None
) -> str | None:
    """One expand child's record key: its parent's key and its index among that
    parent's children, or ``None`` when the parent row carries no key."""
    if parent is None:
        return None
    parent_key = record_key_at(parent.keys, _index_field(item, "parent"))
    if parent_key is None:
        return None
    return child_key(parent_key, _index_field(item, "child"))


def _key_marked_join_rows(
    records: dict[str, NodeRecords],
    plans: Mapping[str, RecordPlan],
) -> None:
    """Give a join's rows without a key the row's own ``by`` values or, failing
    that, the first input's key at its position; a marked output row carries no
    fields of its own but keeps its address."""
    for node_id, node_records in records.items():
        plan = plans[node_id]
        if plan.mode != _JOIN_MODE or node_records.keys is None:
            continue
        first = _first_input(plan, records)
        node_records.keys = [
            _join_row_key(
                key,
                node_records.values[position],
                _join_row_paths(plan),
                None if first is None else record_key_at(first.keys, position),
            )
            for position, key in enumerate(node_records.keys)
        ]


def _join_row_paths(plan: RecordPlan) -> list[str]:
    """The paths that address a join's rows: its ``row_key``, else its ``by``."""
    return plan.row_key_paths or plan.key_paths


def _join_row_key(
    key: str | None,
    value: Any,
    key_paths: Sequence[str],
    fallback: str | None,
) -> str | None:
    """A join row's key: the one it already has, else its own ``by`` values,
    else the first input's key at its position."""
    if key is not None:
        return key
    found = composite_key(value, key_paths) if key_paths else None
    return found if found is not None else fallback


def _node_keys(
    runtime_graph: RuntimeGraph,
    plans: Mapping[str, RecordPlan],
    plan: RecordPlan,
    values: Sequence[Any],
    produced: Mapping[str, Sequence[Any]],
    records: Mapping[str, NodeRecords],
) -> list[str | None] | None:
    """The key of each record of one node, in row order, read from its rows.

    A 1:1 op whose own row carries no key reads it from the node it declares, at
    the same position, else from the literal list the op carries for that
    input; every other keyed node (a keyed retrieval, an op inheriting its
    single keyed input's key) reads its own rows."""
    if plan.column is not None:
        return [
            None if value.get(plan.column) is None else key_text(value[plan.column])
            for value in values
        ]
    own = (
        None
        if plan.key_path is None
        else [composite_key(value, [plan.key_path]) for value in values]
    )
    declared = runtime_graph.record_keys.get(plan.op.node_id)
    if declared is not None and declared.node != plan.op.node_id:
        if _one_to_one(plan.op):
            rows = produced.get(declared.node)
            if rows is None:
                rows = _declared_rows(plan, declared.path, len(values))
            if rows is not None:
                return [
                    (
                        composite_key(rows[position], [declared.path])
                        if key is None and position < len(rows)
                        else key
                    )
                    for position, key in enumerate(own or [None] * len(values))
                ]
        return own
    if own is not None:
        return own
    inherited = None if plan.inherits_key is None else records.get(plan.inherits_key)
    if inherited is None:
        return None
    return [record_key_at(inherited.keys, position) for position in range(len(values))]


def _declared_rows(plan: RecordPlan, path: str, count: int) -> Sequence[Any] | None:
    """The literal list the op carries for a declared input, when exactly one of
    its lists holds ``count`` rows that all carry a key at ``path``."""
    matching = [
        rows
        for rows in _literal_rows(plan)
        if len(rows) == count
        and all(composite_key(row, [path]) is not None for row in rows)
    ]
    return matching[0] if len(matching) == 1 else None


def _table_rows(table: Any) -> list[dict[str, Any]]:
    """A serialized column table as one row object per index, in index order."""
    found = serialized_table(table)
    return [] if found is None else table_rows(found)


def _inherit_marker_keys(
    order: Sequence[str],
    plans: Mapping[str, RecordPlan],
    runtime_graph: RuntimeGraph,
    records: dict[str, NodeRecords],
) -> None:
    """A failed record keeps its address: a marker carries no key of its own,
    so a keyed 1:1 node's marker takes the key of the record it replaces at the
    same position of its single keyed input."""
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


def record_values(response: Any) -> list[Any]:
    """The values one task produced, in order: ``items`` outputs, or grouped rows."""
    values: list[Any] = []
    payload = response
    if isinstance(payload, Mapping) and isinstance(payload.get("value"), Mapping):
        payload = payload["value"]
    if not isinstance(payload, Mapping):
        return []
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    for item in items:
        rows = item.get("rows") if isinstance(item, Mapping) else None
        if isinstance(rows, list):
            values.extend(_row_value(row) for row in rows)
        else:
            values.append(_item_output(item))
    return values


def _record_values(
    response: Any, plan: RecordPlan, items: Sequence[Mapping[str, Any]]
) -> list[Any]:
    """The values one task produced, in order, under ``items``: one output per
    item, a retrieval's rows, or a rowwise API task's grouped rows."""
    values: list[Any] = []
    if plan.column is not None:
        return [row for item in items for row in _table_rows(item.get("table"))]
    for item in items:
        rows = item.get("rows")
        if isinstance(rows, list):
            values.extend(_row_value(row) for row in rows)
        else:
            values.append(_item_output(item))
    return values


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
