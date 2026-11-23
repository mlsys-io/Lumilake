"""Per-record store: entry format and signature digest; an entry's object key
is the SHA-256 digest of the decisions that produce its value, and an object is
trusted only when the digest inside it equals its key."""

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from lumilake_server.runtime.runtime_ops import RuntimeOp

RECORD_PREFIX = "records"
"""Archive key prefix every entry object lives under."""

DIGEST_VERSION = "lumilake-record-v1"

_MARKER_KEY = "_fail"


def canonical_json(value: Any) -> str:
    """One spelling for a JSON value, so equal values hash equally."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )


def digest_of(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def data_spec_dict(op: RuntimeOp) -> dict[str, Any]:
    """The op's data spec as the dict a worker receives."""
    return op.data_spec.model_dump(exclude_none=True)


def op_spec_digest(op: RuntimeOp, logical_names: Mapping[str, str]) -> str:
    """The op's computation, with node ids as their logical names and per-row
    data left out."""
    spec = {
        "task_type": op.task_type,
        "backend": op.backend,
        "model": op.model,
        "model_spec": _without_data(op.model_spec, logical_names),
        "inference_spec": _without_data(op.inference_spec, logical_names),
        "api_spec": _without_data(op.api_spec, logical_names),
        "output_enforcement": _without_data(op.output_enforcement, logical_names),
        "data_spec": _without_data_data_spec(data_spec_dict(op), logical_names),
    }
    return digest_of({"digest_version": DIGEST_VERSION, "spec": spec})


_DATA_KEYS = frozenset({"items", "values"})
"""Fields that hold per-row data at the positions where a spec keeps them."""

_SECRET_KEYS = frozenset({"seed", "lumid_cfg"})
"""Fields that hold a seed or a credential wherever they sit."""

_LIST_TYPE = "list"
"""The data spec payload whose ``items`` are literal rows one per position."""

_PYTHON_TASK_TYPE = "python"
"""The task type that reads its rows through a plan."""

_SAMPLING_TASKS = frozenset({"inference", "api"})
"""Task types whose decoding may draw tokens; classification takes an argmax
and every other task computes without a model."""


def literal_lists(op: RuntimeOp) -> list[list[Any]]:
    """Every literal rows list the op's data spec carries outside a python plan:
    each ``type: list`` payload's ``items`` and a top-level ``seeds`` list."""
    if op.task_type == _PYTHON_TASK_TYPE:
        return []
    spec = data_spec_dict(op)
    found: list[list[Any]] = []
    _collect_literal_rows(spec, found)
    seeds = spec.get("seeds")
    if isinstance(seeds, list):
        found.append(seeds)
    return found


def _collect_literal_rows(value: Any, found: list[list[Any]]) -> None:
    """Collect every literal rows list in a data spec side, in spec order."""
    if isinstance(value, Mapping):
        items = value.get("items")
        if value.get("type") == _LIST_TYPE and isinstance(items, list):
            found.append(items)
            return
        for item in value.values():
            _collect_literal_rows(item, found)
        return
    if isinstance(value, list):
        for item in value:
            _collect_literal_rows(item, found)


def _drop_literal_rows(value: Any) -> Any:
    """A data spec side without the literal rows it carries, in the same walk
    ``literal_lists`` collects them by."""
    if isinstance(value, Mapping):
        kept: dict[str, Any] = {}
        for key, item in value.items():
            if (
                key == "items"
                and value.get("type") == _LIST_TYPE
                and isinstance(item, list)
            ):
                continue
            kept[key] = _drop_literal_rows(item)
        return kept
    if isinstance(value, list):
        return [_drop_literal_rows(item) for item in value]
    return value


def _without_data_data_spec(
    data_spec: Mapping[str, Any], logical_names: Mapping[str, str]
) -> Any:
    """A data spec without the rows the task reads: a python step's ``items``
    and plan ``values``, every literal rows list and a top-level ``seeds`` list.
    A schema naming ``items`` elsewhere stays, because it changes the value."""
    dropped = frozenset({*_DATA_KEYS, "seeds"})
    kept: dict[str, Any] = {}
    for key, item in data_spec.items():
        if key in dropped:
            continue
        if key == "plan" and isinstance(item, list):
            kept[key] = [
                (
                    _without_values(entry, logical_names)
                    if isinstance(entry, Mapping)
                    else _without_data(entry, logical_names)
                )
                for entry in item
            ]
            continue
        kept[key] = _drop_literal_rows(_without_data(item, logical_names))
    return kept


def _without_values(entry: Mapping[str, Any], logical_names: Mapping[str, str]) -> Any:
    """A plan entry without the rows it reads, its node reference named by the
    logical op."""
    kept: dict[str, Any] = {}
    for key, item in entry.items():
        if key == "values":
            continue
        if key == "node" and isinstance(item, str):
            kept[key] = logical_names.get(item, item)
            continue
        kept[key] = _without_data(item, logical_names)
    return kept


def _without_data(value: Any, logical_names: Mapping[str, str]) -> Any:
    """Replace every node reference with its logical op name and drop every
    secret, recursively; every other field stays, a schema's ``items`` too."""
    if isinstance(value, Mapping):
        kept: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or key in _SECRET_KEYS:
                continue
            if key == "node" and isinstance(item, str):
                kept[key] = logical_names.get(item, item)
                continue
            kept[key] = _without_data(item, logical_names)
        return kept
    if isinstance(value, list):
        return [_without_data(item, logical_names) for item in value]
    return value


def declared_seed(op: RuntimeOp) -> int | None:
    """The seed the op itself declares, if any."""
    candidates: list[Any] = [
        op.inference_spec.get("seed"),
        data_spec_dict(op).get("seed"),
    ]
    body = op.api_spec.get("json")
    if isinstance(body, Mapping):
        candidates.append(body.get("seed"))
    for seed in candidates:
        if isinstance(seed, int) and not isinstance(seed, bool):
            return seed
    return None


def has_declared_seed(op: RuntimeOp) -> bool:
    """Whether the op names a seed; a sampling op reproduces its draws only
    when it does."""
    return declared_seed(op) is not None


def samples(op: RuntimeOp) -> bool:
    """Whether the op draws tokens freely; beam search or a zero temperature
    settles the decoder, and a hosted chat's decoding is read from its request
    body as well."""
    if op.task_type not in _SAMPLING_TASKS:
        return False
    specs: list[Mapping[str, Any]] = [op.inference_spec]
    body = op.api_spec.get("json")
    if isinstance(body, Mapping):
        specs.append(body)
    return not any(_decodes_deterministically(spec) for spec in specs)


def _decodes_deterministically(spec: Mapping[str, Any]) -> bool:
    """Whether a spec's decoder is fixed: beam search, or a zero temperature."""
    beams = spec.get("num_beams")
    if isinstance(beams, int) and not isinstance(beams, bool) and beams > 1:
        return True
    temperature = spec.get("temperature")
    return (
        isinstance(temperature, (int, float))
        and not isinstance(temperature, bool)
        and temperature == 0
    )


def is_marker(value: Any) -> bool:
    """Whether a value is a failed record's marker: the object whose only key is
    ``_fail``, as the parse-op branch defines it."""
    return isinstance(value, dict) and list(value) == [_MARKER_KEY]


def marker_reason(value: Any) -> Any:
    return value.get(_MARKER_KEY) if isinstance(value, dict) else value


class RecordAddress(BaseModel):
    """Where a record is: its op, its key at that op, and the semantic round."""

    model_config = ConfigDict(extra="forbid")

    op: str
    record: str | None = None
    """``None`` for an op whose rows carry no logical key: one whole-op entry."""
    round: int = 0


class RecordRealization(BaseModel):
    """Which draw produced a record; absent for a deterministic op."""

    model_config = ConfigDict(extra="forbid")

    seed: int | None = None
    attempt: int | None = None
    production: str | None = None
    """Set beside the seed on a record no seed reproduces: the draw is unique to
    its production."""

    def is_absent(self) -> bool:
        return self.seed is None and self.attempt is None and self.production is None


class EffectVersions(BaseModel):
    """What the record depends on beyond its inputs and code."""

    model_config = ConfigDict(extra="forbid")

    unversioned: bool = False
    """Reads its source live, so its own record names a production unique to the
    run that made it; its consumers name the value it returned instead."""


class RecordSource(BaseModel):
    """The physical production of an entry: which job, round and task, where."""

    model_config = ConfigDict(extra="forbid")

    job: str
    round: int = 0
    task_id: str
    worker: str | None = None
    member_jobs: list[str] = Field(default_factory=list)
    """Every job id that shared the physical task; a record answered for each
    of them reads back from this entry."""


class RecordEntry(BaseModel):
    """One stored record, written once to ``records/<digest>.json``."""

    model_config = ConfigDict(extra="forbid")

    digest: str
    address: RecordAddress
    status: Literal["ok", "failed"] = "ok"
    value: Any = None
    value_digest: str | None = None
    input_digests: list[str] = Field(default_factory=list)
    """The signature digests of the records this one read, in their defined
    order; lineage walks them back."""
    input_values: list[str] = Field(default_factory=list)
    """The value digests of those same records, in the same order. They enter
    the signature so a changed input value changes it."""
    realization: RecordRealization = Field(default_factory=RecordRealization)
    versions: EffectVersions = Field(default_factory=EffectVersions)
    source: RecordSource
    written_at: float

    def body(self) -> bytes:
        return canonical_json(self.model_dump(mode="json")).encode("utf-8")

    @classmethod
    def from_body(cls, body: bytes) -> Self:
        return cls.model_validate(json.loads(body.decode("utf-8")))


def _record_body(
    *,
    spec_digest: str,
    address: RecordAddress,
    input_digests: list[str],
    input_values: list[str],
    realization: RecordRealization,
    versions: EffectVersions,
) -> dict[str, Any]:
    """What a record's key covers, before the status tells a success from a
    failure."""
    return {
        "digest_version": DIGEST_VERSION,
        "spec": spec_digest,
        "address": address.model_dump(mode="json"),
        "inputs": input_digests,
        "input_values": input_values,
        "realization": realization.model_dump(mode="json"),
        "versions": versions.model_dump(mode="json"),
    }


def entry_digest(
    *,
    spec_digest: str,
    address: RecordAddress,
    input_digests: list[str],
    input_values: list[str],
    realization: RecordRealization,
    versions: EffectVersions,
) -> str:
    """The record's key: its op spec, address, input signatures and values,
    realization and effect versions. An input's own signature names what it
    read, so the chain cuts off at a shared signature."""
    return digest_of(
        _record_body(
            spec_digest=spec_digest,
            address=address,
            input_digests=input_digests,
            input_values=input_values,
            realization=realization,
            versions=versions,
        )
    )


def failed_record_digest(
    *,
    spec_digest: str,
    address: RecordAddress,
    input_digests: list[str],
    input_values: list[str],
    realization: RecordRealization,
    versions: EffectVersions,
) -> str:
    """A failed attempt's key: the success key's own inputs, realization and
    versions, with the status added, so a failure answers only its own
    request and two failures on different inputs get separate keys."""
    return digest_of(
        {
            **_record_body(
                spec_digest=spec_digest,
                address=address,
                input_digests=input_digests,
                input_values=input_values,
                realization=realization,
                versions=versions,
            ),
            "status": "failed",
        }
    )


def entry_object_name(digest: str) -> str:
    return f"{RECORD_PREFIX}/{digest}.json"


def index_object_name(job_id: str) -> str:
    return f"{RECORD_PREFIX}/jobs/{job_id}.json"


class RecordIndexEntry(BaseModel):
    """One entry a job's index lists."""

    model_config = ConfigDict(extra="forbid")

    op: str
    record: str | None = None
    round: int = 0
    digest: str


class RecordIndex(BaseModel):
    """A job's entries, by op, key and round, so a lineage read needs no scan;
    the writer serialises each job's load-merge-saves."""

    model_config = ConfigDict(extra="forbid")

    job: str
    entries: list[RecordIndexEntry] = Field(default_factory=list)

    def body(self) -> bytes:
        return canonical_json(self.model_dump(mode="json")).encode("utf-8")

    @classmethod
    def from_body(cls, body: bytes) -> Self:
        return cls.model_validate(json.loads(body.decode("utf-8")))

    def rounds(self) -> list[int]:
        return sorted({entry.round for entry in self.entries})

    def find(
        self, *, op: str, record: str | None, round_number: int
    ) -> RecordIndexEntry | None:
        for entry in self.entries:
            if (
                entry.op == op
                and entry.record == record
                and entry.round == round_number
            ):
                return entry
        return None


def row_key(value: Any, path: str) -> str | None:
    """The record key a row carries at ``path``, as text; a JSON string row is
    decoded first and a path that does not resolve gives ``None``."""
    found = path_value(value, path)
    if found is None or isinstance(found, str):
        return found
    return canonical_json(found)


def path_value(value: Any, path: str) -> Any:
    """The value a row carries at ``path``: a JSON string row is decoded first,
    and a path that does not resolve gives ``None``."""
    current: Any = value
    for part in [part for part in path.split(".") if part]:
        if isinstance(current, str):
            try:
                current = json.loads(current)
            except ValueError:
                return None
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current
