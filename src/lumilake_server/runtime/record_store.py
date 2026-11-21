"""Per-record store: entry format and signature digest.

An entry is one computed value at one logical address. Its object key is the
SHA-256 digest of the decisions that produce the value, so an object is trusted
only when the digest recorded inside it equals its key. Physical node ids, batch
ids, workers and row positions never enter the digest.
"""

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Literal

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


def op_spec_digest(op: RuntimeOp) -> str:
    """The op's computation: task, code, prompt, config, model and decoding.

    Only the computation enters: runtime node ids, a declared sampling seed and
    the per-row data the op reads (literal rows and per-row seeds) are left out,
    because physical ids belong nowhere in a digest, and a seed belongs to the
    record's realization while per-row data belongs to that record's inputs. The
    digest is therefore the same however the graph was sliced, prefixed or
    ordered, and however many rows a slice holds.
    """
    spec = {
        "task_type": op.task_type,
        "backend": op.backend,
        "model": op.model,
        "model_spec": _without_data(op.model_spec),
        "inference_spec": _without_data(op.inference_spec),
        "api_spec": _without_data(op.api_spec),
        "output_enforcement": _without_data(op.output_enforcement),
        "data_spec": _without_data(op.data_spec),
    }
    return digest_of({"digest_version": DIGEST_VERSION, "spec": spec})


_DATA_KEYS = frozenset(
    {
        "node",
        "path",
        "items",
        "seeds",
        "plan",
        "values",
        "columns",
        "rows",
        "seed",
        "lumid_cfg",
    }
)
"""Fields that name per-row data, a physical node or a secret, never the
computation."""


def _without_data(value: Any) -> Any:
    """Drop every data-carrying and physical-node field, recursively.

    A spec entry that is only data (a node reference or a per-row cell) becomes
    a marker instead of vanishing, so a column that reads different rows stays
    distinct from one that reads none."""
    if isinstance(value, Mapping):
        kept: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or key in _DATA_KEYS:
                continue
            kept[key] = _without_data(item)
        return kept
    if isinstance(value, list):
        return [_without_data(item) for item in value]
    return value


def declared_seed(op: RuntimeOp) -> int | None:
    """The seed the op itself declares, if any."""
    for spec in (op.inference_spec, op.data_spec):
        seed = spec.get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool):
            return seed
    return None


def is_marker(value: Any) -> bool:
    """Whether a value is a failed record's marker rather than a result."""
    return isinstance(value, dict) and _MARKER_KEY in value


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

    def is_absent(self) -> bool:
        return self.seed is None and self.attempt is None


class EffectVersions(BaseModel):
    """What the record depends on beyond its inputs and code."""

    model_config = ConfigDict(extra="forbid")

    unversioned: bool = False
    """Reads a source with no recorded snapshot upstream: never reusable."""


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

    def for_job(self, job_id: str) -> bool:
        return job_id == self.source.job or job_id in self.source.member_jobs

    @classmethod
    def from_body(cls, body: bytes) -> "RecordEntry":
        return cls.model_validate(json.loads(body.decode("utf-8")))


def entry_digest(
    *,
    spec_digest: str,
    address: RecordAddress,
    input_digests: list[str],
    input_values: list[str],
    realization: RecordRealization,
    versions: EffectVersions,
) -> str:
    """The signature digest: the key of the record these inputs decide.

    It covers the op spec, the record address, what each input record was and
    what it returned, the realization and the effect versions. A consumer that
    probed before running can compute it, and a recomputed input whose value is
    unchanged leaves its consumers' signatures unchanged."""
    return digest_of(
        {
            "digest_version": DIGEST_VERSION,
            "spec": spec_digest,
            "address": address.model_dump(mode="json"),
            "inputs": input_digests,
            "input_values": input_values,
            "realization": realization.model_dump(mode="json"),
            "versions": versions.model_dump(mode="json"),
        }
    )


def failed_record_digest(*, spec_digest: str, address: RecordAddress) -> str:
    """A failed attempt has its own key, so it never answers a request for a
    success."""
    return digest_of(
        {
            "digest_version": DIGEST_VERSION,
            "spec": spec_digest,
            "address": address.model_dump(mode="json"),
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
    """A job's entries, by op, key and round, so a lineage read needs no scan.

    Each batch read-modify-writes the object at
    ``records/jobs/<job id>.json``.
    """

    model_config = ConfigDict(extra="forbid")

    job: str
    entries: list[RecordIndexEntry] = Field(default_factory=list)

    def body(self) -> bytes:
        return canonical_json(self.model_dump(mode="json")).encode("utf-8")

    @classmethod
    def from_body(cls, body: bytes) -> "RecordIndex":
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
    """The record key a row carries at ``path``, as text.

    A row that is a JSON string is decoded first; a path that does not resolve
    leaves the row without a key, so it gets no per-record entry."""
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
    if current is None:
        return None
    return current if isinstance(current, str) else canonical_json(current)
