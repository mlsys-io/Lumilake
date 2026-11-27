"""Payload of a stored record's lineage read, walked from the archive alone."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from lumilake_server.runtime.record_writer import lineage, read_index
from lumilake_server.utils.job_storage import JobStorage


class JobLineageEntry(BaseModel):
    digest: str = Field(description="Record signature digest, its store key.")
    op: str = Field(description="Logical op the record belongs to.")
    record: str | None = Field(
        default=None, description="The record's key at that op, when it has one."
    )
    round: int = Field(description="Semantic round the record belongs to.")
    status: Literal["ok", "failed"] = Field(
        description="Whether the record is a result or a failure marker."
    )
    job_id: str = Field(description="Job that produced the record.")
    task_id: str = Field(description="Physical FlowMesh task that computed it.")
    input_digests: list[str] = Field(
        default_factory=list, description="Signature digests the record read."
    )


class JobLineagePayload(BaseModel):
    job_id: str = Field(description="Job identifier.")
    op: str = Field(description="Logical op the requested record belongs to.")
    record: str | None = Field(
        default=None, description="Requested record key, when the op has keys."
    )
    entries: list[JobLineageEntry] = Field(
        default_factory=list,
        description=(
            "The record's chain: the record itself, then each input it read, level by"
            " level."
        ),
    )


class JobLineageResponse(BaseModel):
    ok: bool = Field(description="Whether the request succeeded.")
    data: JobLineagePayload = Field(description="Lineage payload.")


def stored_rounds(storage: JobStorage, job_id: str) -> list[int]:
    """The rounds the job's index lists, oldest first."""
    index = read_index(storage, job_id)
    return [] if index is None else index.rounds()


def build_lineage_payload(
    storage: JobStorage,
    *,
    job_id: str,
    op: str,
    record: str | None,
    round_number: int | None,
) -> dict[str, Any] | None:
    """The chain of one stored record, or ``None`` when the store has none."""
    index = read_index(storage, job_id)
    if index is None:
        return None
    rounds = index.rounds()
    target = round_number if round_number is not None else (rounds[-1] if rounds else 0)
    entries = lineage(
        storage,
        index=index,
        op=op,
        record=record,
        round_number=target,
    )
    if not entries:
        return None
    return JobLineagePayload(
        job_id=job_id,
        op=op,
        record=record,
        entries=[
            JobLineageEntry(
                digest=entry.digest,
                op=entry.address.op,
                record=entry.address.record,
                round=entry.address.round,
                status=entry.status,
                job_id=entry.source.job,
                task_id=entry.source.task_id,
                input_digests=entry.input_digests,
            )
            for entry in entries
        ],
    ).model_dump(mode="json")
