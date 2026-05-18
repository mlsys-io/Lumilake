import asyncio
import base64
import binascii
import datetime as dt
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from io import BytesIO
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.responses import Response
from lumid_hooks import PrincipalContext
from lumilake import envs
from lumilake.log import Logger, init_child_logger, set_trace_id
from lumilake_hook import ResourceAction, ResourceKind, UsageRow
from minio import Minio
from minio.error import S3Error
from psycopg import sql
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator

from lumilake_server.hooks.security import (
    authenticate_request,
    emit_usage,
    register_resource,
    require_permission,
    resolve_accessible_ids,
    run_submission_guards,
)
from lumilake_server.parser import parse_n8n_payload, parse_yaml_payload
from lumilake_server.runtime.protocol import (
    LumilakeRequestConfig,
    LumilakeResponse,
    Priority,
    RequestCancelledError,
)
from lumilake_server.runtime.request import WorkflowSliceMeta
from lumilake_server.runtime.server import LumilakeServer
from lumilake_server.schemas.io import DBLocation, IOLocation, S3Location
from lumilake_server.schemas.progress import JobProgress
from lumilake_server.utils.data_profile_offload import (
    build_request_data_profile_tasks,
    data_profile_registry,
    run_data_profile_task,
)
from lumilake_server.utils.io_locations import normalize_s3_literal
from lumilake_server.utils.job_storage import JobStorage, get_job_storage
from lumilake_server.utils.parsing import split_bucket_prefix
from lumilake_server.utils.s3 import create_minio_client
from lumilake_server.utils.utils import unique_id

JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
JOB_STATUS_VALUES: tuple[JobStatus, ...] = (
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
)
JOB_STATUS_DESCRIPTION = (
    "Job lifecycle status: `pending` (queued), `running` (executing), "
    "`completed` (finished successfully), `failed` (finished with error), "
    "`cancelled` (cancelled before completion)."
)


def _chunk_inputs(
    inputs: dict[str, list[str]],
    input_batch_size: int,
) -> list[dict[str, list[str]]]:
    lengths = [len(v) for v in inputs.values()]
    max_len = max(lengths) if lengths else 0
    for vals in inputs.values():
        if len(vals) not in {1, max_len}:
            raise ValueError(
                "workflow inputs list lengths must match or be a single value"
            )
    if max_len <= input_batch_size:
        return [inputs]
    batches: list[dict[str, list[str]]] = []
    for start in range(0, max_len, input_batch_size):
        batch_inputs: dict[str, list[str]] = {}
        for key, vals in inputs.items():
            if len(vals) == 1:
                batch_inputs[key] = vals
            else:
                batch_inputs[key] = vals[start : start + input_batch_size]
        batches.append(batch_inputs)
    return batches


def _input_shape(inputs: dict[str, list[str]]) -> tuple[int, tuple[str, ...]]:
    lengths = [len(v) for v in inputs.values()]
    max_len = max(lengths) if lengths else 0
    for vals in inputs.values():
        if len(vals) not in {1, max_len}:
            raise ValueError(
                "workflow inputs list lengths must match or be a single value"
            )
    varying = tuple(sorted(key for key, vals in inputs.items() if len(vals) > 1))
    return max_len, varying


def _workflow_template_hash(workflow_payload: Any, workflow_format: str) -> str:
    payload = {
        "format": workflow_format,
        "workflow": workflow_payload,
    }
    # `default=str` is a defensive fallback: YAML (and n8n workflows that
    # were loaded through a permissive parser) can contain non-JSON-native
    # scalars like ``datetime.date`` from unquoted ``YYYY-MM-DD`` fields.
    # Without this, a template hash computation would raise ``TypeError``
    # and surface as a 500. The str() representation is stable-enough for
    # hashing purposes since the same input always produces the same str.
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _decode_workflow_body(raw: str, workflow_format: str, idx: int) -> Any:
    """Decode the raw ``entry.workflow`` string for the declared format.

    ``native``/``n8n`` are JSON-encoded; ``yaml`` is a YAML document. The
    resulting Python object is stored as ``workflow_payload`` and flows into
    the template-hash + parser-dispatch paths below.
    """
    if workflow_format == "yaml":
        try:
            return yaml.safe_load(raw)
        except yaml.MarkedYAMLError as exc:
            mark = exc.problem_mark
            if mark is not None:
                line = mark.line + 1
                column = mark.column + 1
                detail = (
                    f"Invalid workflow YAML for index {idx} at "
                    f"line {line}, column {column}: {exc}"
                )
            else:
                detail = f"Invalid workflow YAML for index {idx}: {exc}"
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail,
            ) from exc
        except yaml.YAMLError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid workflow YAML for index {idx}: {exc}",
            ) from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid workflow JSON for index {idx}: {exc}",
        ) from exc


def _dispatch_workflow_to_graph_specs(
    *,
    workflow_format: str,
    workflow_payload: Any,
    batch_inputs: dict[str, list[str]],
    graph_name: str,
    graph_specs: dict[str, dict[str, Any]],
    idx: int,
) -> None:
    """Parse one workflow slice and merge it into ``graph_specs`` in place.

    Mirrors the three formats accepted by the submit/preview endpoints:

    * ``native`` — ``workflow_payload`` already contains a compiled Lumilake
      graph (optionally wrapped under a ``graph`` key); stored verbatim.
    * ``n8n`` — wrap into ``{"graphs": [...]}`` and delegate to
      :func:`parse_n8n_payload`.
    * ``yaml`` — the YAML document parsed by :func:`parse_yaml_payload`
      (Lumilake-native op-shape only; users with an n8n workflow should
      submit it via ``Workflow-Format: n8n``). The endpoint overrides the
      YAML's top-level ``name``/``inputs`` with the per-batch values so
      slicing produces unique graph ids just like the ``n8n`` branch does.
    """
    if workflow_format == "n8n":
        payload = {
            "graphs": [
                {
                    "workflow": workflow_payload,
                    "inputs": batch_inputs,
                    "name": graph_name,
                }
            ]
        }
        try:
            parsed_graphs = parse_n8n_payload(payload)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        overlap = set(parsed_graphs).intersection(graph_specs)
        if overlap:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"duplicate graph names after parsing: {sorted(overlap)}",
            )
        graph_specs.update(parsed_graphs)
        return
    if workflow_format == "yaml":
        if not isinstance(workflow_payload, dict):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"YAML workflow at index {idx} must be a mapping at the top level"
                ),
            )
        # Override the YAML document's top-level ``name``/``inputs`` with the
        # endpoint-chosen batch values so slicing produces distinct graph ids,
        # mirroring how the ``n8n`` branch's wrapper payload supplies them.
        yaml_dict = dict(workflow_payload)
        yaml_dict["name"] = graph_name
        yaml_dict["inputs"] = batch_inputs
        try:
            parsed_graphs = parse_yaml_payload(yaml_dict)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        overlap = set(parsed_graphs).intersection(graph_specs)
        if overlap:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"duplicate graph names after parsing: {sorted(overlap)}",
            )
        graph_specs.update(parsed_graphs)
        return
    # native
    if not isinstance(workflow_payload, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"workflow payload at index {idx} must be an object",
        )
    graph_payload = (
        workflow_payload["graph"] if "graph" in workflow_payload else workflow_payload
    )
    graph_specs[graph_name] = {
        "graph": graph_payload,
        "inputs": batch_inputs,
    }


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    submitted_at: str
    inputs: dict[str, dict[str, list[str]]]
    output_location: dict[str, IOLocation]
    org_id: str = "default"
    user_id: str = "local"
    started_at: str | None = None
    finished_at: str | None = None
    optimization_seconds: float | None = None
    selection_seconds: float | None = None
    clustering_seconds: float | None = None
    error: str | None = None
    progress: JobProgress = field(default_factory=JobProgress)
    result: LumilakeResponse | None = None
    folder_inputs: dict[str, str] = field(default_factory=dict)
    trace_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.output_location:
            raise ValueError("output_location is required")
        self.progress = JobProgress.model_validate(self.progress)
        if self.result is not None:
            self.result = LumilakeResponse.model_validate(self.result)
        normalized: dict[str, IOLocation] = {}
        for key, loc in self.output_location.items():
            normalized[key] = _IO_LOCATION_ADAPTER.validate_python(loc)
        self.output_location = normalized


class JobSubmitPayload(BaseModel):
    job_id: str = Field(description="Job identifier.")
    status: JobStatus = Field(description=JOB_STATUS_DESCRIPTION)


class JobSubmitResponse(BaseModel):
    ok: bool = Field(description="Whether the request succeeded.")
    data: JobSubmitPayload = Field(description="Job submission payload.")


class JobStatusPayload(BaseModel):
    job_id: str = Field(description="Job identifier.")
    status: JobStatus = Field(description=JOB_STATUS_DESCRIPTION)
    submitted_at: dt.datetime = Field(description="Submission timestamp.")
    started_at: dt.datetime | None = Field(default=None, description="Start timestamp.")
    finished_at: dt.datetime | None = Field(
        default=None, description="Finish timestamp."
    )
    optimization_seconds: float | None = Field(
        default=None,
        description="Accumulated optimizer scheduling time in seconds.",
    )
    selection_seconds: float | None = Field(
        default=None,
        description=(
            "Accumulated job-manager batch-selection time in seconds, excluding"
            " the clustering substep reported in clustering_seconds."
        ),
    )
    clustering_seconds: float | None = Field(
        default=None,
        description=(
            "Accumulated affinity-clustering time in seconds, attributed to this"
            " request as its share of the batches it participated in."
        ),
    )
    error: str | None = Field(
        default=None,
        description="Error message, if any.",
    )


class JobStatusResponse(BaseModel):
    ok: bool = Field(description="Whether the request succeeded.")
    data: JobStatusPayload = Field(description="Job status payload.")


class JobListItem(BaseModel):
    job_id: str
    status: JobStatus = Field(description=JOB_STATUS_DESCRIPTION)
    submitted_at: dt.datetime
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    optimization_seconds: float | None = None
    selection_seconds: float | None = None
    clustering_seconds: float | None = None
    error: str | None = None


class JobListPayload(BaseModel):
    items: list[JobListItem]
    page: int
    page_size: int
    total: int
    total_pages: int


class JobListResponse(BaseModel):
    ok: bool
    data: JobListPayload


class JobProgressPayload(BaseModel):
    job_id: str = Field(description="Job identifier.")
    progress: JobProgress = Field(description="Progress payload.")


class JobProgressResponse(BaseModel):
    ok: bool = Field(description="Whether the request succeeded.")
    data: JobProgressPayload = Field(description="Progress payload.")


class JobResultPayload(BaseModel):
    job_id: str = Field(description="Job identifier.")
    result: LumilakeResponse = Field(description="Result payload.")


class JobResultResponse(BaseModel):
    ok: bool = Field(description="Whether the request succeeded.")
    data: JobResultPayload = Field(description="Result payload.")


class JobCancelPayload(BaseModel):
    job_id: str = Field(description="Job identifier.")
    status: JobStatus = Field(description=JOB_STATUS_DESCRIPTION)


class JobCancelResponse(BaseModel):
    ok: bool = Field(description="Whether the request succeeded.")
    data: JobCancelPayload = Field(description="Job cancellation payload.")


class JobInputsPayload(BaseModel):
    job_id: str = Field(description="Job identifier.")
    inputs: dict[str, dict[str, list[str]]] = Field(description="Inputs payload.")


class JobInputsResponse(BaseModel):
    ok: bool = Field(description="Whether the request succeeded.")
    data: JobInputsPayload = Field(description="Inputs payload.")


class EmptyInputsErrorDetail(BaseModel):
    message: str
    parsed_input_names: list[str]


class JobAlreadyFinishedDetail(BaseModel):
    message: str
    status: str
    job_id: str


def _format_validation_errors(exc: ValidationError) -> str:
    """Collapse Pydantic validation errors into a readable message."""
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"])
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


_IO_LOCATION_TYPES = {"db", "s3"}


def _validate_inputs_shape(data: Any) -> None:
    """Validate raw inputs before Pydantic union parsing to give clear errors."""
    if not isinstance(data, dict):
        return
    inputs = data.get("inputs")
    if inputs is None or not isinstance(inputs, dict):
        return
    for key, value in inputs.items():
        if isinstance(value, str):
            raise ValueError(
                f"inputs['{key}']: expected a list of strings, got a plain string. "
                f'Wrap it as ["{value}"]'
            )
        if isinstance(value, (int, float, bool)):
            raise ValueError(
                f"inputs['{key}']: expected a list of strings, got"
                f" {type(value).__name__}"
            )
        if isinstance(value, dict):
            if value.get("type") not in _IO_LOCATION_TYPES:
                raise ValueError(
                    f"inputs['{key}']: expected a list of strings or an IO location "
                    f'({{"type": "db"|"s3", ...}}), got a dict with keys '
                    f"{sorted(value.keys())}"
                )
        elif isinstance(value, list):
            if not value:
                raise ValueError(
                    f"inputs['{key}']: expected a non-empty list of strings, got []"
                )
            for idx, item in enumerate(value):
                if not isinstance(item, str):
                    raise ValueError(
                        f"inputs['{key}'][{idx}]: expected a string, "
                        f"got {type(item).__name__} ({item!r})"
                    )


class JobSubmitItem(BaseModel):
    workflow: str = Field(
        description=(
            "Workflow JSON string. For native submissions, this should be the "
            "serialized graph only (json.dumps of graph)."
        )
    )
    inputs: dict[str, list[str] | IOLocation] = Field(
        description="Input mapping from input name to list of strings or location."
    )
    output_location: IOLocation = Field(description="Output location.")
    input_batch_size: int | None = Field(
        default=None,
        description=(
            "Optional batch size for slicing list inputs. "
            "When omitted, list inputs are sliced with batch size 1."
        ),
    )
    name: str | None = Field(default=None, description="Optional workflow name.")

    @model_validator(mode="before")
    @classmethod
    def _check_inputs_shape(cls, data: Any) -> Any:
        _validate_inputs_shape(data)
        return data

    @model_validator(mode="after")
    def _validate_inputs(self) -> "JobSubmitItem":
        if not self.inputs:
            raise ValueError("inputs is required")
        return self


class JobSubmitRequest(BaseModel):
    data: list[JobSubmitItem] = Field(description="Workflow submission entries.")
    priority: Priority = Field(
        default=Priority.MEDIUM,
        description="Job scheduling priority: `low`, `medium`, or `high`.",
    )


class JobPreviewItem(BaseModel):
    """Mirrors ``JobSubmitItem`` so the same payload can be sent to both
    ``/jobs`` and ``/jobs/preview``.  ``output_location`` is accepted but
    ignored during preview."""

    workflow: str = Field(
        description=(
            "Workflow JSON string. For native submissions, this should be the "
            "serialized graph only (json.dumps of graph)."
        )
    )
    inputs: dict[str, list[str] | IOLocation] = Field(
        description="Input mapping from input name to list of strings or location."
    )
    output_location: IOLocation | None = Field(
        default=None,
        description=(
            "Output location (accepted for payload compatibility, ignored during"
            " preview)."
        ),
    )
    input_batch_size: int | None = Field(
        default=None,
        description=(
            "Optional batch size for slicing list inputs. "
            "When omitted, list inputs are sliced with batch size 1."
        ),
    )
    name: str | None = Field(default=None, description="Optional workflow name.")

    @model_validator(mode="before")
    @classmethod
    def _check_inputs_shape(cls, data: Any) -> Any:
        _validate_inputs_shape(data)
        return data

    @model_validator(mode="after")
    def _validate_inputs(self) -> "JobPreviewItem":
        if not self.inputs:
            raise ValueError("inputs is required")
        return self


class JobPreviewRequest(BaseModel):
    """Mirrors ``JobSubmitRequest`` so the same payload can be sent to both
    ``/jobs`` and ``/jobs/preview``.  ``priority`` is accepted but ignored
    during preview."""

    data: list[JobPreviewItem] = Field(description="Workflow preview entries.")
    priority: Priority = Field(
        default=Priority.MEDIUM,
        description="Accepted for payload compatibility; ignored during preview.",
    )


class JobPreviewPayload(BaseModel):
    request_id: str = Field(description="Preview request identifier.")
    selected_workers: list[str] = Field(
        description="Workers selected for schedule generation."
    )
    worker_assignment: dict[str, list[str]] = Field(
        description="Generated schedule mapping of worker to node execution order."
    )
    runtime_graph_node_counts: dict[str, int] = Field(
        description="Per-graph runtime node counts before merge optimization."
    )
    merged_runtime_node_count: int = Field(
        description="Merged runtime node count used by optimizer."
    )


class JobPreviewResponse(BaseModel):
    ok: bool = Field(description="Whether the request succeeded.")
    data: JobPreviewPayload = Field(description="Preview schedule payload.")


jobs: dict[str, JobRecord] = {}
jobs_lock = asyncio.Lock()

router = APIRouter(tags=["Jobs"])
_job_storage: JobStorage = get_job_storage()

logger = init_child_logger("JobRoutes")
_IO_LOCATION_ADAPTER: TypeAdapter[IOLocation] = TypeAdapter(IOLocation)

_DATA_URL_RE = re.compile(r"^data:(image/[^;]+);base64,(.+)$", re.DOTALL)
_ARTIFACT_PATH_TOKEN = "/artifacts/"
_IMAGE_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
}


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _release_output_locations(_record: JobRecord) -> None:
    return


async def mark_running_jobs_failed(reason: str = "server shutdown") -> None:
    async with jobs_lock:
        active = [
            record
            for record in jobs.values()
            if record.status in {"pending", "running"}
        ]
        if not active:
            return
        for record in active:
            record.status = "failed"
            if not record.error:
                record.error = reason
            record.finished_at = _now()
            _job_storage.save(record)
            _release_output_locations(record)
        logger.warning("Marked %d jobs failed due to shutdown", len(active))


async def _load_job_record(job_id: str) -> JobRecord | None:
    record: JobRecord | None
    async with jobs_lock:
        record = jobs.get(job_id)
    if record is None:
        try:
            loaded = _job_storage.load(job_id)
        except KeyError:
            loaded = None
        if loaded:
            try:
                record = JobRecord(**loaded)
            except ValueError:
                record = None
            async with jobs_lock:
                if record is not None:
                    jobs[job_id] = record
    return record


async def _load_authorized_job_record(
    job_id: str,
    principal: PrincipalContext,
    action: ResourceAction,
    hook_logger: Logger,
) -> JobRecord:
    record = await _load_job_record(job_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="job not found"
        )
    await require_permission(principal, ResourceKind.JOB, job_id, action, hook_logger)
    return record


def _usage_row(record: JobRecord, principal: PrincipalContext) -> UsageRow:
    return {
        "org_id": record.org_id,
        "principal_id": principal.principal_id,
        "job_id": record.job_id,
        "status": record.status,
        "submitted_at": record.submitted_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "optimization_seconds": record.optimization_seconds,
        "trace_ids": list(record.trace_ids),
        "emitted_at": dt.datetime.now(dt.UTC),
    }


def _store_artifacts(job_id: str, payload: Any) -> Any:
    if isinstance(payload, dict):
        updated: dict[str, Any] = {}
        for key, value in payload.items():
            updated[key] = _store_artifacts(job_id, value)
        if "image_base64" in payload and isinstance(payload["image_base64"], str):
            encoded = payload["image_base64"]
            mime = payload.get("mime_type", "image/png")
            try:
                data = base64.b64decode(encoded)
            except (binascii.Error, ValueError):
                return updated
            ext = _IMAGE_EXT.get(mime, "bin")
            filename = f"{unique_id()}.{ext}"
            uri = _job_storage.save_artifact(job_id, filename, data, mime)
            updated["image_uri"] = uri
            updated.pop("image_base64", None)
        return updated
    if isinstance(payload, list):
        return [_store_artifacts(job_id, item) for item in payload]
    if isinstance(payload, str):
        match = _DATA_URL_RE.match(payload)
        if match:
            mime, encoded = match.groups()
            try:
                data = base64.b64decode(encoded)
            except (binascii.Error, ValueError):
                return payload
            ext = _IMAGE_EXT.get(mime, "bin")
            filename = f"{unique_id()}.{ext}"
            return _job_storage.save_artifact(job_id, filename, data, mime)
    return payload


def _artifact_name_from_uri(value: str) -> str | None:
    if _ARTIFACT_PATH_TOKEN not in value:
        return None
    parsed = urlparse(value)
    path = parsed.path or ""
    idx = path.rfind(_ARTIFACT_PATH_TOKEN)
    if idx < 0:
        return None
    name = path[idx + len(_ARTIFACT_PATH_TOKEN) :]
    if not name:
        return None
    name = name.split("/")[-1]
    return name or None


def _collect_artifact_uris(payload: Any) -> set[str]:
    uris: set[str] = set()
    if isinstance(payload, dict):
        for value in payload.values():
            uris.update(_collect_artifact_uris(value))
        return uris
    if isinstance(payload, list):
        for item in payload:
            uris.update(_collect_artifact_uris(item))
        return uris
    if isinstance(payload, str) and _artifact_name_from_uri(payload):
        uris.add(payload)
    return uris


def _normalize_artifact_path(path: str) -> str:
    cleaned = path.strip()
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="artifact path is required",
        )
    return cleaned


def _summarize_error_info(error_info: list[dict[str, Any]] | None) -> str | None:
    if not error_info:
        return None
    for entry in error_info:
        if not isinstance(entry, dict):
            continue
        batch_error = entry.get("batch_error")
        if isinstance(batch_error, str) and batch_error.strip():
            return batch_error.strip()
    first = error_info[0]
    if isinstance(first, dict):
        for key in ("error", "message", "detail"):
            value = first.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return json.dumps(first, sort_keys=True, default=str)
    if isinstance(first, str):
        return first
    return str(first)


def _location_to_literal(location: IOLocation) -> str:
    if isinstance(location, DBLocation):
        table = location.table.strip()
        if "." not in table:
            table = f"public.{table}"
        return f"db://{table}.{location.column.strip()}"
    return location.prefix.strip()


def _resolve_input_location(location: IOLocation) -> IOLocation:
    if isinstance(location, S3Location):
        return location.model_copy(
            update={"prefix": normalize_s3_literal(location.prefix)}
        )
    return location


def _resolve_output_location(location: IOLocation) -> IOLocation:
    if isinstance(location, S3Location):
        return location.model_copy(
            update={"prefix": normalize_s3_literal(location.prefix)}
        )
    return location


async def _require_location_permission(
    location: IOLocation,
    action: ResourceAction,
    principal: PrincipalContext,
    hook_logger: Logger,
) -> None:
    if isinstance(location, DBLocation):
        await require_permission(
            principal,
            ResourceKind.TABLE,
            location.table,
            action,
            hook_logger,
        )
        return
    await require_permission(
        principal,
        ResourceKind.OBJECT_PREFIX,
        location.prefix,
        action,
        hook_logger,
    )


def _coerce_output_values(graph_outputs: Any) -> list[str]:
    """Flatten ``{output_name: [values]}`` to a single string list.

    Picks the first output key's values when multiple are present; returns
    ``[]`` for malformed payloads.
    """
    if not isinstance(graph_outputs, dict):
        return []
    first: list[Any] | None = None
    for value in graph_outputs.values():
        if isinstance(value, list):
            first = value
            break
    if first is None:
        return []
    return [str(item) for item in first]


def _write_output_value_set(
    client: Minio,
    bucket: str,
    key_prefix: str,
    is_folder: bool,
    values: list[str],
) -> None:
    """Write ``values`` to ``(bucket, key_prefix)``.

    ``is_folder=True`` writes one ``item-000001.txt`` per value under the
    prefix; ``False`` concatenates values into a single object at the key.
    """
    if is_folder:
        for idx, value in enumerate(values, start=1):
            object_name = f"{key_prefix.rstrip('/')}/item-{idx:06d}.txt"
            data = str(value).encode("utf-8")
            client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=BytesIO(data),
                length=len(data),
                content_type="text/plain",
            )
        return
    payload = "\n".join(values).encode("utf-8")
    client.put_object(
        bucket_name=bucket,
        object_name=key_prefix,
        data=BytesIO(payload),
        length=len(payload),
        content_type="text/plain",
    )


async def _dump_output_locations(
    *,
    output_locations: dict[str, IOLocation],
    response_outputs: dict[str, Any],
    compute_pool: AsyncConnectionPool | None,
) -> None:
    """Write each graph's output to its declared location.

    Runs from the job-finalize task; a write failure is logged and the job
    still records as completed (best-effort dump semantics).
    """
    s3_client: Minio | None = None
    for graph_name, location in output_locations.items():
        graph_outputs = response_outputs.get(graph_name)
        values = _coerce_output_values(graph_outputs)
        if not values:
            continue
        if isinstance(location, DBLocation):
            schema, table = _parse_table_ref(location.table)
            column = location.column.strip()
            insert_stmt = sql.SQL("INSERT INTO {}.{} ({}) VALUES (%s)").format(
                sql.Identifier(schema),
                sql.Identifier(table),
                sql.Identifier(column),
            )
            if compute_pool is None:
                raise RuntimeError(
                    "DATABASE_URL is not configured; cannot dump DB outputs"
                )
            async with compute_pool.connection() as conn:
                async with conn.cursor() as cur:
                    for value in values:
                        await cur.execute(insert_stmt, (value,))
            continue
        assert isinstance(location, S3Location)
        normalized = normalize_s3_literal(location.prefix)
        bucket, key_prefix = _resolve_output_s3_to_physical(normalized)
        is_folder = normalized.endswith("/")
        # The Minio client is sync; offload so the event loop doesn't
        # block on large output writes during job-finalize.
        if s3_client is None:
            s3_client = _compute_minio_client()
        await asyncio.to_thread(
            _write_output_value_set,
            s3_client,
            bucket,
            key_prefix,
            is_folder,
            values,
        )


def _parse_table_ref(table_ref: str) -> tuple[str, str]:
    """``schema.table`` or bare ``table`` -> ``(schema, table)`` tuple."""
    parts = table_ref.strip().split(".", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "public", parts[0].strip()


async def _validate_db_location_live(
    location: DBLocation,
    *,
    compute_pool: AsyncConnectionPool | None,
) -> DBLocation:
    """Validate that the referenced table/column actually exists on compute PG."""
    schema, table = _parse_table_ref(location.table)
    column = location.column.strip()
    if not column:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="column is required",
        )
    column_exists_sql = (
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s AND column_name = %s"
    )
    if compute_pool is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DATABASE_URL is not configured",
        )
    async with compute_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(column_exists_sql, (schema, table, column))
            found = await cur.fetchone() is not None
    if not found:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"column {column} not found on {schema}.{table} (compute DB)",
        )
    return DBLocation(type="db", table=f"{schema}.{table}", column=column)


def _compute_minio_client():
    endpoint = envs.S3_ENDPOINT
    access_key = envs.S3_ACCESS_KEY
    connection_value = envs.S3_CONNECTION_VALUE
    if not (endpoint and access_key and connection_value):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compute S3 is not configured (S3_ENDPOINT/KEYS missing)",
        )
    return create_minio_client(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=connection_value,
        cert_file=envs.S3_CERT_FILE,
    )


def _resolve_logical_s3_to_physical(logical: str) -> tuple[str, str]:
    """Treat ``logical`` as a key path under the configured ``S3_URL`` bucket."""
    if not envs.S3_URL:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="S3_URL is not configured",
        )
    bucket, base_prefix = split_bucket_prefix(envs.S3_URL)
    rel = logical.lstrip("/")
    if base_prefix:
        return bucket, f"{base_prefix}/{rel}" if rel else base_prefix
    return bucket, rel


def _resolve_output_s3_to_physical(logical: str) -> tuple[str, str]:
    """Resolve output locations to compute S3."""
    if not envs.S3_URL:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="S3_URL is not configured",
        )
    return _resolve_logical_s3_to_physical(logical)


def _validate_s3_location_live(location: S3Location, *, must_exist: bool) -> S3Location:
    normalized = normalize_s3_literal(location.prefix)
    if not must_exist:
        return location.model_copy(update={"prefix": normalized})
    bucket, key_prefix = _resolve_logical_s3_to_physical(normalized)
    client = _compute_minio_client()
    try:
        objs = client.list_objects(bucket, prefix=key_prefix, recursive=False)
        first = next(iter(objs), None)
    except S3Error as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"s3 prefix {normalized} not accessible: {exc}",
        ) from exc
    if first is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"s3 prefix {normalized} missing on compute S3",
        )
    return location.model_copy(update={"prefix": normalized})


async def _validate_location(
    *,
    location: IOLocation,
    compute_pool: AsyncConnectionPool | None,
    must_exist_for_s3: bool,
) -> IOLocation:
    """Validate a DBLocation or S3Location against the compute cluster."""
    if isinstance(location, DBLocation):
        return await _validate_db_location_live(
            location,
            compute_pool=compute_pool,
        )
    assert isinstance(location, S3Location)
    return _validate_s3_location_live(
        location,
        must_exist=must_exist_for_s3,
    )


async def _resolve_s3_input_values(
    *,
    input_name: str,
    location: S3Location,
) -> list[str]:
    """Expand an S3 prefix to ``s3://bucket/key`` URLs via ``Minio.list_objects``
    dispatched to a worker thread."""
    literal = location.prefix.strip()
    if not literal:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"s3 input {input_name!r} prefix is required",
        )
    normalized = normalize_s3_literal(literal)
    bucket, key_prefix = _resolve_logical_s3_to_physical(normalized)

    client = _compute_minio_client()

    def _list() -> list[str]:
        urls: list[str] = []
        for obj in client.list_objects(bucket, prefix=key_prefix, recursive=True):
            if obj.is_dir:
                continue
            name = obj.object_name or ""
            if not name:
                continue
            urls.append(f"s3://{bucket}/{name}")
        return urls

    try:
        urls = await asyncio.to_thread(_list)
    except S3Error as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"s3 resolve failed for input {input_name!r}: {exc}",
        ) from exc
    if not urls:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"s3 resolve returned no files for input {input_name!r}",
        )
    return urls


async def _resolve_input_values(
    *,
    input_name: str,
    raw: list[str] | IOLocation,
    compute_pool: AsyncConnectionPool | None,
    principal: PrincipalContext,
    hook_logger: Logger,
) -> list[str]:
    values = await _resolve_input_values_raw(
        input_name=input_name,
        raw=raw,
        compute_pool=compute_pool,
        principal=principal,
        hook_logger=hook_logger,
    )
    if not values:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=EmptyInputsErrorDetail(
                message=f"input {input_name!r} resolved to an empty value list",
                parsed_input_names=[input_name],
            ).model_dump(),
        )
    return values


async def _resolve_input_values_raw(
    *,
    input_name: str,
    raw: list[str] | IOLocation,
    compute_pool: AsyncConnectionPool | None,
    principal: PrincipalContext,
    hook_logger: Logger,
) -> list[str]:
    if isinstance(raw, list):
        return raw
    location = _IO_LOCATION_ADAPTER.validate_python(raw)
    location = _resolve_input_location(location)
    await _require_location_permission(
        location,
        ResourceAction.READ,
        principal,
        hook_logger,
    )
    if isinstance(location, DBLocation):
        validated_location = await _validate_location(
            location=location,
            compute_pool=compute_pool,
            must_exist_for_s3=True,
        )
        return [_location_to_literal(validated_location)]
    if isinstance(location, S3Location):
        return await _resolve_s3_input_values(
            input_name=input_name,
            location=location,
        )
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"unsupported input location type for {input_name!r}",
    )


async def _run_job(
    job_id: str,
    graph_specs: dict[str, dict[str, Any]],
    workflow_slices: dict[str, WorkflowSliceMeta],
    record: JobRecord,
    priority: Priority,
    compute_pool: AsyncConnectionPool | None,
    principal: PrincipalContext,
    trace_id: str,
) -> None:
    set_trace_id(trace_id)
    server = LumilakeServer.get_started_instance()

    async with jobs_lock:
        if record.status == "cancelled":
            _job_storage.save(record)
            cancelled_before_start = True
        else:
            cancelled_before_start = False
            record.status = "running"
            record.started_at = _now()
            record.progress.queuing.completed = True
            _job_storage.save(record)
    if cancelled_before_start:
        if record.finished_at:
            await emit_usage([_usage_row(record, principal)], logger)
        return

    try:
        graphs = server.parse_query(graph_specs)
        record.progress.query_parsing.completed = True
        if envs.LUMILAKE_DISABLE_DATA_PROFILE:
            logger.info(
                "Skipping inline data profile build/run for job %s "
                "(LUMILAKE_DISABLE_DATA_PROFILE)",
                job_id,
            )
        else:
            data_profile_tasks = build_request_data_profile_tasks(
                request_id=job_id,
                graphs=graphs,
                workflow_slices=workflow_slices,
            )
            for task in data_profile_tasks:
                try:
                    result = await asyncio.to_thread(
                        run_data_profile_task, task.payload
                    )
                except Exception:
                    logger.exception(
                        "Data profile task %s failed for job %s; continuing",
                        task.task_key,
                        job_id,
                    )
                    continue
                data_profile_registry[task.task_key] = result.model_dump(mode="json")
            if data_profile_tasks:
                logger.info(
                    "Ran %d data profile task(s) inline for job %s",
                    len(data_profile_tasks),
                    job_id,
                )
        response = await server.execute(
            graphs,
            job_id,
            LumilakeRequestConfig(
                priority=priority,
                user_id=principal.external_id,
                org_id=principal.org_id,
            ),
            workflow_slices=workflow_slices,
        )
        trace_ids: list[str] = []
        try:
            trace_ids = server.trace_ids_for_request(job_id)
        except Exception:
            logger.exception("Failed to resolve trace ids for job %s", job_id)
        trace_ids = [trace_id.strip() for trace_id in trace_ids if trace_id.strip()]
        do_dump = False
        artifact_uris: set[str] = set()
        async with jobs_lock:
            stored_payload = _store_artifacts(job_id, response.model_dump())
            artifact_uris = _collect_artifact_uris(stored_payload)
            record.result = LumilakeResponse.model_validate(stored_payload)
            record.trace_ids = list(trace_ids)
            try:
                record.optimization_seconds = server.optimization_seconds_for_request(
                    job_id
                )
            except Exception:
                logger.exception(
                    "Failed to resolve optimizer timing for job %s",
                    job_id,
                )
            try:
                record.selection_seconds = server.selection_seconds_for_request(job_id)
                record.clustering_seconds = server.clustering_seconds_for_request(
                    job_id
                )
            except Exception:
                logger.exception(
                    "Failed to resolve job-manager timing for job %s",
                    job_id,
                )
            record.finished_at = _now()
            if record.result.error_info:
                record.status = "failed"
                summary = _summarize_error_info(record.result.error_info)
                if summary:
                    record.error = summary
                _job_storage.save(record)
            else:
                record.status = "completed"
                _job_storage.save(record)
                do_dump = True
        for trace_id in trace_ids:
            try:
                await register_resource(
                    principal,
                    ResourceKind.TRACE,
                    trace_id,
                    {"job_id": job_id},
                    logger,
                )
            except Exception:
                logger.exception("Failed to register trace %s", trace_id)
        for artifact_uri in sorted(artifact_uris):
            filename = _artifact_name_from_uri(artifact_uri)
            if not filename:
                continue
            artifact_id = f"{job_id}/{filename}"
            try:
                await register_resource(
                    principal,
                    ResourceKind.ARTIFACT,
                    artifact_id,
                    {"job_id": job_id, "uri": artifact_uri},
                    logger,
                )
            except Exception:
                logger.exception("Failed to register artifact %s", artifact_id)
        try:
            server.release_request_workflows(job_id)
        except Exception:
            logger.exception("Failed to release runtime trace state for job %s", job_id)
        if do_dump:
            try:
                result_outputs = record.result.outputs if record.result else {}
                await _dump_output_locations(
                    output_locations=record.output_location,
                    response_outputs=result_outputs,
                    compute_pool=compute_pool,
                )
            except Exception:
                logger.exception(
                    "Failed to write outputs for job %s",
                    job_id,
                )
    except RequestCancelledError:
        logger.info("Job %s cancelled", job_id)
        return
    except Exception as exc:  # pragma: no cover
        logger.exception("Job %s failed with exception", job_id)
        async with jobs_lock:
            if record.status != "cancelled":
                record.status = "failed"
                if not record.error:
                    record.error = str(exc)
                record.finished_at = _now()
                _job_storage.save(record)
    finally:
        # ensure latest progress flushed
        _job_storage.save(record)
        if record.status in {"completed", "failed", "cancelled"} and record.finished_at:
            await emit_usage([_usage_row(record, principal)], logger)


@router.post(
    "/jobs/preview",
    summary="Preview job schedule",
    description=(
        "Compile workflow(s) and generate optimizer schedule without runtime "
        "execution. Accepts the same payload shape as POST /jobs so clients "
        "can reuse the request body. Fields `priority` and `output_location` "
        "are accepted for compatibility but ignored."
    ),
    response_description="Preview schedule result.",
    status_code=status.HTTP_200_OK,
    response_model=JobPreviewResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "workflow": {
                                            "type": "string",
                                            "description": (
                                                "Workflow body. Use the "
                                                "Workflow-Format header to indicate "
                                                "whether this is a native Lumilake "
                                                "graph, an n8n workflow payload, or a "
                                                "Lumilake YAML document."
                                            ),
                                        },
                                        "inputs": {
                                            "type": "object",
                                            "additionalProperties": {
                                                "oneOf": [
                                                    {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                    {
                                                        "type": "object",
                                                        "properties": {
                                                            "type": {
                                                                "type": "string",
                                                                "enum": ["db", "s3"],
                                                            },
                                                            "table": {"type": "string"},
                                                            "column": {
                                                                "type": "string"
                                                            },
                                                            "prefix": {
                                                                "type": "string"
                                                            },
                                                        },
                                                        "required": ["type"],
                                                        "additionalProperties": False,
                                                    },
                                                ],
                                            },
                                        },
                                        "output_location": {
                                            "type": "object",
                                            "description": (
                                                "Accepted but ignored during preview."
                                            ),
                                            "properties": {
                                                "type": {
                                                    "type": "string",
                                                    "enum": ["db", "s3"],
                                                },
                                                "table": {"type": "string"},
                                                "column": {"type": "string"},
                                                "prefix": {"type": "string"},
                                            },
                                            "required": ["type"],
                                            "additionalProperties": False,
                                        },
                                        "input_batch_size": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "description": (
                                                "Optional server-side slice size for"
                                                " list inputs. Defaults to 1 when"
                                                " omitted. Only the first batch is used"
                                                " for preview."
                                            ),
                                        },
                                        "name": {"type": "string"},
                                    },
                                    "required": ["workflow", "inputs"],
                                    "additionalProperties": False,
                                },
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                                "default": "medium",
                                "description": "Accepted but ignored during preview.",
                            },
                        },
                        "required": ["data"],
                    }
                }
            },
        },
    },
)
async def preview_job(
    request: Request,
    principal: PrincipalContext = Depends(authenticate_request),
    workflow_format: str = Header(
        default="native",
        alias="Workflow-Format",
        description=(
            "Workflow format: `native` (compiled Lumilake graph JSON), "
            "`n8n` (n8n workflow payload), or `yaml` (Lumilake YAML workflow)."
        ),
    ),
) -> dict[str, Any]:
    hook_logger = request.app.state.logger
    await require_permission(
        principal, ResourceKind.JOB, None, ResourceAction.WRITE, hook_logger
    )
    await run_submission_guards(principal, hook_logger)
    compute_pool: AsyncConnectionPool | None = request.app.state.compute_db_pool
    try:
        json_body = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON body: {exc}",
        ) from exc
    workflow_format = workflow_format.lower()
    if workflow_format not in {"native", "n8n", "yaml"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Workflow-Format must be 'native', 'n8n', or 'yaml'",
        )

    try:
        preview_request = JobPreviewRequest.model_validate(json_body)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_format_validation_errors(exc),
        ) from exc
    entries = preview_request.data
    if not entries:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="data must contain at least one entry",
        )

    graph_specs: dict[str, dict[str, Any]] = {}
    seen_public_names: set[str] = set()
    preview_request_id = f"preview-{unique_id()}"

    for idx, entry in enumerate(entries):
        workflow_payload = _decode_workflow_body(entry.workflow, workflow_format, idx)

        name = entry.name or f"graph_{idx}"
        if name in seen_public_names:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"duplicate workflow name: {name}",
            )
        seen_public_names.add(name)

        inputs: dict[str, list[str]] = {}
        for input_name, raw in entry.inputs.items():
            inputs[input_name] = await _resolve_input_values(
                input_name=input_name,
                raw=raw,
                compute_pool=compute_pool,
                principal=principal,
                hook_logger=hook_logger,
            )

        try:
            _input_shape(inputs)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        input_batch_size = entry.input_batch_size
        if input_batch_size is not None and input_batch_size <= 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="input_batch_size must be a positive integer",
            )
        # Preview uses only the first batch per entry — enough to generate
        # a representative schedule without processing the full input set.
        effective_batch_size = input_batch_size or 1
        try:
            input_batches = _chunk_inputs(inputs, effective_batch_size)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        first_batch = input_batches[0]
        graph_name = name
        if graph_name in graph_specs:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"duplicate internal graph name: {graph_name}",
            )
        _dispatch_workflow_to_graph_specs(
            workflow_format=workflow_format,
            workflow_payload=workflow_payload,
            batch_inputs=first_batch,
            graph_name=graph_name,
            graph_specs=graph_specs,
            idx=idx,
        )

    server = LumilakeServer.get_started_instance()
    try:
        graphs = server.parse_query(graph_specs)
    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Graph compilation failed: {exc}",
        ) from exc
    try:
        preview = await server.preview_schedule(
            graphs=graphs,
            request_id=preview_request_id,
            data_profile_results={},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"schedule preview failed: {exc}",
        ) from exc

    return {
        "ok": True,
        "data": {
            "request_id": preview.request_id,
            "selected_workers": preview.selected_workers,
            "worker_assignment": preview.schedule.worker_assignment,
            "runtime_graph_node_counts": preview.runtime_graph_node_counts,
            "merged_runtime_node_count": preview.merged_runtime_node_count,
            "selection_seconds": preview.selection_seconds,
            "clustering_seconds": preview.clustering_seconds,
            "optimization_seconds": preview.optimization_seconds,
        },
    }


@router.post(
    "/jobs",
    summary="Submit a job",
    description=(
        "Submit one or more compiled graphs for execution. "
        "Use Workflow-Format header for n8n or yaml payloads."
    ),
    response_description="Job submission result.",
    status_code=status.HTTP_200_OK,
    response_model=JobSubmitResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "workflow": {"type": "string"},
                                        "inputs": {
                                            "type": "object",
                                            "additionalProperties": {
                                                "oneOf": [
                                                    {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                    {
                                                        "type": "object",
                                                        "properties": {
                                                            "type": {
                                                                "type": "string",
                                                                "enum": ["db", "s3"],
                                                            },
                                                            "table": {"type": "string"},
                                                            "column": {
                                                                "type": "string"
                                                            },
                                                            "prefix": {
                                                                "type": "string"
                                                            },
                                                        },
                                                        "required": ["type"],
                                                        "additionalProperties": False,
                                                    },
                                                ],
                                            },
                                        },
                                        "output_location": {
                                            "type": "object",
                                            "properties": {
                                                "type": {
                                                    "type": "string",
                                                    "enum": ["db", "s3"],
                                                },
                                                "table": {"type": "string"},
                                                "column": {"type": "string"},
                                                "prefix": {"type": "string"},
                                            },
                                            "required": ["type"],
                                            "additionalProperties": False,
                                        },
                                        "input_batch_size": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "description": (
                                                "Optional server-side slice size for"
                                                " list inputs. Defaults to 1 when"
                                                " omitted."
                                            ),
                                        },
                                        "name": {"type": "string"},
                                    },
                                    "required": [
                                        "workflow",
                                        "inputs",
                                        "output_location",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                                "default": "medium",
                            },
                        },
                        "required": ["data"],
                    }
                }
            },
        },
    },
)
async def submit_job(
    request: Request,
    principal: PrincipalContext = Depends(authenticate_request),
    workflow_format: str = Header(
        default="native",
        alias="Workflow-Format",
        description=(
            "Workflow format: `native` (compiled Lumilake graph JSON), "
            "`n8n` (n8n workflow payload), or `yaml` (Lumilake YAML workflow)."
        ),
    ),
) -> dict[str, Any]:
    hook_logger = request.app.state.logger
    await require_permission(
        principal, ResourceKind.JOB, None, ResourceAction.WRITE, hook_logger
    )
    await run_submission_guards(principal, hook_logger)
    compute_pool: AsyncConnectionPool | None = request.app.state.compute_db_pool
    try:
        json_body = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON body: {exc}",
        ) from exc
    workflow_format = workflow_format.lower()
    if workflow_format not in {"native", "n8n", "yaml"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Workflow-Format must be 'native', 'n8n', or 'yaml'",
        )

    try:
        submit_request = JobSubmitRequest.model_validate(json_body)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_format_validation_errors(exc),
        ) from exc
    entries = submit_request.data
    priority = submit_request.priority
    if not entries:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="data must contain at least one entry",
        )
    job_id = f"req-{unique_id()}"
    graph_specs: dict[str, dict[str, Any]] = {}
    workflow_slices: dict[str, WorkflowSliceMeta] = {}
    resolved_inputs: dict[str, dict[str, list[str]]] = {}
    output_locations: dict[str, IOLocation] = {}
    seen_public_names: set[str] = set()
    for idx, entry in enumerate(entries):
        workflow_payload = _decode_workflow_body(entry.workflow, workflow_format, idx)

        name = entry.name or f"graph_{idx}"
        if name in seen_public_names:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"duplicate workflow name: {name}",
            )
        seen_public_names.add(name)
        inputs: dict[str, list[str]] = {}
        output_location = _resolve_output_location(entry.output_location)
        await _require_location_permission(
            output_location,
            ResourceAction.WRITE,
            principal,
            hook_logger,
        )
        output_location = await _validate_location(
            location=output_location,
            compute_pool=compute_pool,
            must_exist_for_s3=False,
        )
        for input_name, raw in entry.inputs.items():
            inputs[input_name] = await _resolve_input_values(
                input_name=input_name,
                raw=raw,
                compute_pool=compute_pool,
                principal=principal,
                hook_logger=hook_logger,
            )
        try:
            total_length, varying_input_keys = _input_shape(inputs)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        input_batch_size = entry.input_batch_size
        if input_batch_size is not None and input_batch_size <= 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="input_batch_size must be a positive integer",
            )
        effective_batch_size = input_batch_size or 1
        try:
            input_batches = _chunk_inputs(inputs, effective_batch_size)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        resolved_inputs[name] = inputs
        output_locations[name] = output_location

        template_hash = _workflow_template_hash(workflow_payload, workflow_format)
        slice_start = 0
        for batch_idx, batch_inputs in enumerate(input_batches):
            graph_name = (
                name if len(input_batches) == 1 else f"{name}__slice_{batch_idx + 1}"
            )
            if graph_name in graph_specs:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"duplicate internal graph name: {graph_name}",
                )
            slice_length, _ = _input_shape(batch_inputs)
            workflow_slices[graph_name] = WorkflowSliceMeta(
                public_graph_name=name,
                slice_index=batch_idx,
                slice_start=slice_start,
                slice_length=slice_length,
                total_length=total_length,
                template_hash=template_hash,
                varying_input_keys=varying_input_keys,
            )
            logger.debug(
                "Resolved inputs for %s: %s",
                graph_name,
                {key: list(vals) for key, vals in batch_inputs.items()},
            )
            _dispatch_workflow_to_graph_specs(
                workflow_format=workflow_format,
                workflow_payload=workflow_payload,
                batch_inputs=batch_inputs,
                graph_name=graph_name,
                graph_specs=graph_specs,
                idx=idx,
            )
            slice_start += slice_length

        logger.info(
            "Prepared workflow '%s' into %d slice(s) with batch size %d",
            name,
            len(input_batches),
            effective_batch_size,
        )

    record = JobRecord(
        job_id=job_id,
        status="pending",
        submitted_at=_now(),
        inputs=resolved_inputs,
        output_location=output_locations,
        org_id=principal.org_id,
        user_id=principal.external_id,
        progress=JobProgress(),
    )

    async with jobs_lock:
        jobs[job_id] = record
        _job_storage.save(record)
    await register_resource(
        principal,
        ResourceKind.JOB,
        job_id,
        {"workflow_count": len(entries), "status": record.status},
        hook_logger,
    )

    asyncio.create_task(
        _run_job(
            job_id,
            graph_specs,
            workflow_slices,
            record,
            priority,
            compute_pool,
            principal,
            str(getattr(request.state, "trace_id", job_id)),
        )
    )
    return {"ok": True, "data": {"job_id": job_id, "status": record.status}}


@router.get(
    "/jobs",
    summary="List jobs",
    description="List jobs with pagination and optional status filters.",
    response_description="Paginated job list.",
    status_code=status.HTTP_200_OK,
    response_model=JobListResponse,
)
async def list_jobs(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    principal: PrincipalContext = Depends(authenticate_request),
    status_filter: list[str] | None = Query(
        default=None,
        alias="status",
        description=(
            "Optional status filters (repeat query key). "
            "Allowed values: pending, running, completed, failed, cancelled."
        ),
    ),
) -> dict[str, Any]:
    hook_logger = request.app.state.logger
    allowed = set(JOB_STATUS_VALUES)
    statuses = set(status_filter or [])
    invalid = sorted(status for status in statuses if status not in allowed)
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"invalid status filters: {', '.join(invalid)}",
        )

    await require_permission(
        principal,
        ResourceKind.JOB,
        None,
        ResourceAction.READ,
        hook_logger,
    )
    accessible_job_ids = await resolve_accessible_ids(
        principal,
        ResourceKind.JOB,
        ResourceAction.READ,
        hook_logger,
    )
    items, total = _job_storage.list_summaries(
        org_id=principal.org_id,
        user_id=None,
        job_ids=accessible_job_ids,
        statuses=statuses or None,
        page=page,
        page_size=page_size,
    )
    total_pages = (total + page_size - 1) // page_size if total else 0
    return {
        "ok": True,
        "data": {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
        },
    }


@router.get(
    "/jobs/{job_id}",
    summary="Get job status",
    description="Fetch job status metadata.",
    response_description="Job status metadata.",
    status_code=status.HTTP_200_OK,
    response_model=JobStatusResponse,
    openapi_extra={
        "responses": {
            "404": {"description": "Job not found"},
        },
    },
)
async def get_job(
    job_id: str,
    request: Request,
    principal: PrincipalContext = Depends(authenticate_request),
) -> dict[str, Any]:
    hook_logger = request.app.state.logger
    record = await _load_authorized_job_record(
        job_id,
        principal,
        ResourceAction.READ,
        hook_logger,
    )

    data = asdict(record)
    data.pop("progress", None)
    data.pop("result", None)
    return {"ok": True, "data": data}


@router.post(
    "/jobs/{job_id}/cancel",
    summary="Cancel a job",
    description="Request cancellation for a pending or running job.",
    response_description="Job cancellation result.",
    status_code=status.HTTP_200_OK,
    response_model=JobCancelResponse,
    openapi_extra={
        "responses": {
            "404": {"description": "Job not found"},
            "409": {"description": "Job already finished"},
        },
    },
)
async def cancel_job(
    job_id: str,
    request: Request,
    principal: PrincipalContext = Depends(authenticate_request),
) -> dict[str, Any]:
    hook_logger = request.app.state.logger
    record = await _load_authorized_job_record(
        job_id,
        principal,
        ResourceAction.CANCEL,
        hook_logger,
    )

    async with jobs_lock:
        if record.status in {"completed", "failed", "cancelled"}:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=JobAlreadyFinishedDetail(
                    message="job already finished",
                    status=record.status,
                    job_id=job_id,
                ).model_dump(),
            )
        record.status = "cancelled"
        if not record.error:
            record.error = "cancelled by user"
        record.finished_at = _now()
        _job_storage.save(record)

    _release_output_locations(record)

    server = LumilakeServer.get_instance()
    if server.is_started:
        try:
            await server.cancel_request(job_id)
        except Exception:
            logger.warning(
                "Failed to cancel job %s in runtime backend", job_id, exc_info=True
            )

    return {"ok": True, "data": {"job_id": job_id, "status": record.status}}


@router.get(
    "/jobs/{job_id}/progress",
    summary="Get job progress",
    description=(
        "Fetch job progress details (available while job is pending/running and after"
        " completion)."
    ),
    response_description="Job progress payload.",
    status_code=status.HTTP_200_OK,
    response_model=JobProgressResponse,
    openapi_extra={
        "responses": {
            "404": {"description": "Job not found"},
        },
    },
)
async def get_job_progress(
    job_id: str,
    request: Request,
    principal: PrincipalContext = Depends(authenticate_request),
) -> dict[str, Any]:
    hook_logger = request.app.state.logger
    record = await _load_authorized_job_record(
        job_id,
        principal,
        ResourceAction.READ,
        hook_logger,
    )

    if record.status == "cancelled":
        cancelled_progress = JobProgress()
        return {
            "ok": True,
            "data": {
                "job_id": job_id,
                "progress": cancelled_progress.model_dump(by_alias=True),
            },
        }

    server = LumilakeServer.get_started_instance()
    progress_payload = await server.get_request_status(job_id)
    if "error" not in progress_payload:
        progress_model = record.progress.model_copy(deep=True)
        progress_model.apply_status(progress_payload)
        async with jobs_lock:
            if record.progress != progress_model:
                record.progress = progress_model
                _job_storage.save(record)

    return {
        "ok": True,
        "data": {
            "job_id": job_id,
            "progress": record.progress.model_dump(by_alias=True),
        },
    }


@router.get(
    "/jobs/{job_id}/result",
    summary="Get job result",
    description="Fetch job result for a completed job (returns 409 if not completed).",
    response_description="Job result payload.",
    status_code=status.HTTP_200_OK,
    response_model=JobResultResponse,
    openapi_extra={
        "responses": {
            "404": {"description": "Job not found"},
            "409": {"description": "Job not finished yet"},
        },
    },
)
async def get_job_result(
    job_id: str,
    request: Request,
    principal: PrincipalContext = Depends(authenticate_request),
) -> dict[str, Any]:
    hook_logger = request.app.state.logger
    record = await _load_authorized_job_record(
        job_id,
        principal,
        ResourceAction.READ,
        hook_logger,
    )

    if record.status == "cancelled":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="job cancelled",
        )
    if record.status not in {"completed", "failed"} or record.result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="job not finished yet",
        )
    try:
        result_model = LumilakeResponse.model_validate(record.result)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Stored result is malformed: {exc.error_count()} validation error(s)"
            ),
        ) from exc
    return {
        "ok": True,
        "data": {
            "job_id": job_id,
            "result": result_model,
        },
    }


@router.get(
    "/jobs/{job_id}/inputs",
    summary="Get job inputs",
    description="Fetch job inputs payload.",
    response_description="Job inputs payload.",
    status_code=status.HTTP_200_OK,
    response_model=JobInputsResponse,
    openapi_extra={
        "responses": {
            "404": {"description": "Job not found"},
        },
    },
)
async def get_job_inputs(
    job_id: str,
    request: Request,
    principal: PrincipalContext = Depends(authenticate_request),
) -> dict[str, Any]:
    hook_logger = request.app.state.logger
    record = await _load_authorized_job_record(
        job_id,
        principal,
        ResourceAction.READ,
        hook_logger,
    )

    return {"ok": True, "data": {"job_id": job_id, "inputs": record.inputs}}


@router.get(
    "/jobs/{job_id}/artifact",
    summary="Download job artifact",
    description="Download a stored artifact referenced by the job result.",
    response_description="Artifact file stream.",
    status_code=status.HTTP_200_OK,
    openapi_extra={
        "responses": {
            "404": {"description": "Job or artifact not found"},
            "409": {"description": "Job not finished yet"},
        },
    },
)
async def get_job_artifact(
    job_id: str,
    request: Request,
    artifact_path: str = Query(
        ...,
        alias="path",
        description="Artifact path to download (as shown in results).",
    ),
    principal: PrincipalContext = Depends(authenticate_request),
) -> Response:
    hook_logger = request.app.state.logger
    record = await _load_authorized_job_record(
        job_id,
        principal,
        ResourceAction.READ,
        hook_logger,
    )

    if record.status == "cancelled":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="job cancelled",
        )
    if record.status not in {"completed", "failed"} or record.result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="job not finished yet",
        )

    requested_path = _normalize_artifact_path(artifact_path)
    result_payload = (
        record.result.model_dump()
        if isinstance(record.result, LumilakeResponse)
        else record.result
    )
    available = _collect_artifact_uris(result_payload)
    if requested_path not in available:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="artifact not found"
        )
    filename = _artifact_name_from_uri(requested_path)
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="artifact path is invalid",
        )
    await require_permission(
        principal,
        ResourceKind.ARTIFACT,
        f"{job_id}/{filename}",
        ResourceAction.READ,
        hook_logger,
    )

    try:
        data, content_type = _job_storage.get_artifact(job_id, filename)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="artifact not found"
        ) from exc

    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=data, media_type=content_type, headers=headers)
