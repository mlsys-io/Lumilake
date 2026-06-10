import datetime as dt
import json
import logging
from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import asdict
from typing import Any, Literal

from lumilake import envs
from pydantic import BaseModel, ConfigDict, ValidationError

from lumilake_server.utils import lumid_data_client


class ArchiveNotFound(Exception):
    """Raised by archive backends when a key is missing."""


def _normalize_payload(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(by_alias=True)
    if isinstance(value, dict):
        return {key: _normalize_payload(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize_payload(val) for val in value]
    return value


class JobStorage:
    def __init__(self) -> None:
        self.logger = logging.getLogger("JobStorage")

    @abstractmethod
    def save(self, record: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, job_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def reserve_output_location(self, location_key: str, job_id: str) -> None:
        raise NotImplementedError

    def release_output_location(self, location_key: str, job_id: str) -> None:
        raise NotImplementedError

    def save_artifact(
        self, job_id: str, filename: str, data: bytes, content_type: str
    ) -> str:
        raise NotImplementedError

    def get_artifact(self, job_id: str, filename: str) -> tuple[bytes, str]:
        raise NotImplementedError

    def list_summaries(
        self,
        *,
        org_id: str,
        user_id: str | None,
        job_ids: frozenset[str] | None,
        statuses: set[str] | None,
        page: int,
        page_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        raise NotImplementedError

    def iter_summaries(
        self, statuses: set[str] | None = None
    ) -> Iterable["JobSummary"]:
        raise NotImplementedError


class JobSummary(BaseModel):
    job_id: str
    org_id: str
    user_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    submitted_at: dt.datetime
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    optimization_seconds: float | None = None
    selection_seconds: float | None = None
    clustering_seconds: float | None = None
    error: str | None = None

    model_config = ConfigDict(extra="forbid")


def _summary_from_payload(data: dict[str, Any]) -> JobSummary:
    payload = {
        "job_id": data["job_id"],
        "org_id": data["org_id"],
        "user_id": data["user_id"],
        "status": data["status"],
        "submitted_at": data["submitted_at"],
        "started_at": data["started_at"] if "started_at" in data else None,
        "finished_at": data["finished_at"] if "finished_at" in data else None,
        "optimization_seconds": data.get("optimization_seconds"),
        "selection_seconds": data.get("selection_seconds"),
        "clustering_seconds": data.get("clustering_seconds"),
        "error": data["error"] if "error" in data else None,
    }
    return JobSummary.model_validate(payload)


def _sort_summaries(summaries: list[JobSummary]) -> list[JobSummary]:
    return sorted(
        summaries,
        key=lambda item: (
            item.submitted_at,
            item.job_id,
        ),
        reverse=True,
    )


def _filter_summaries(
    summaries: list[JobSummary],
    *,
    org_id: str,
    user_id: str | None,
    job_ids: frozenset[str] | None,
    statuses: set[str] | None,
) -> list[JobSummary]:
    filtered: list[JobSummary] = []
    for summary in summaries:
        if summary.org_id != org_id:
            continue
        if user_id is not None and summary.user_id != user_id:
            continue
        if job_ids is not None and summary.job_id not in job_ids:
            continue
        if statuses and summary.status not in statuses:
            continue
        filtered.append(summary)
    return filtered


class InMemoryJobStorage(JobStorage):
    def __init__(self) -> None:
        super().__init__()
        self._storage: dict[str, dict[str, Any]] = {}
        self._inputs: dict[str, dict[str, Any]] = {}
        self._progress: dict[str, dict[str, Any]] = {}
        self._result: dict[str, dict[str, Any]] = {}
        self._output_index: dict[str, str] = {}
        self._artifacts: dict[str, dict[str, tuple[bytes, str]]] = {}
        self._summaries: dict[str, JobSummary] = {}

    def save(self, record: Any) -> None:
        data = _normalize_payload(asdict(record))
        record_data = dict(data)
        record_data.pop("inputs", None)
        record_data.pop("progress", None)
        record_data.pop("result", None)
        self._storage[record.job_id] = record_data
        self._inputs[record.job_id] = data.get("inputs", {})
        self._progress[record.job_id] = data.get("progress", {})
        if data.get("result") is not None:
            self._result[record.job_id] = data.get("result", {})
        self._summaries[record.job_id] = _summary_from_payload(data)
        self.logger.info("Saved job %s to in-memory storage", record.job_id)

    def load(self, job_id: str) -> dict[str, Any] | None:
        self.logger.info("Loading job %s from in-memory storage", job_id)
        record = self._storage.get(job_id)
        if record is None:
            return None
        data = dict(record)
        data["inputs"] = self._inputs.get(job_id, {})
        data["progress"] = self._progress.get(job_id, {})
        if job_id in self._result:
            data["result"] = self._result.get(job_id)
        return data

    def reserve_output_location(self, location_key: str, job_id: str) -> None:
        existing = self._output_index.get(location_key)
        if existing and existing != job_id:
            record = self._storage.get(existing)
            if record and record.get("status") == "completed":
                self._output_index[location_key] = job_id
                return
            raise ValueError(f"output location {location_key} already reserved")
        self._output_index[location_key] = job_id

    def release_output_location(self, location_key: str, job_id: str) -> None:
        existing = self._output_index.get(location_key)
        if existing == job_id:
            del self._output_index[location_key]

    def save_artifact(
        self, job_id: str, filename: str, data: bytes, content_type: str
    ) -> str:
        self._artifacts.setdefault(job_id, {})[filename] = (data, content_type)
        return f"memory://{job_id}/artifacts/{filename}"

    def get_artifact(self, job_id: str, filename: str) -> tuple[bytes, str]:
        job_artifacts = self._artifacts.get(job_id)
        if not job_artifacts or filename not in job_artifacts:
            raise KeyError(filename)
        return job_artifacts[filename]

    def list_summaries(
        self,
        *,
        org_id: str,
        user_id: str | None,
        job_ids: frozenset[str] | None,
        statuses: set[str] | None,
        page: int,
        page_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        items = list(self._summaries.values())
        filtered = _filter_summaries(
            items, org_id=org_id, user_id=user_id, job_ids=job_ids, statuses=statuses
        )
        sorted_items = _sort_summaries(filtered)
        total = len(sorted_items)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = [item.model_dump(mode="json") for item in sorted_items[start:end]]
        return page_items, total

    def iter_summaries(self, statuses: set[str] | None = None) -> Iterable[JobSummary]:
        for summary in self._summaries.values():
            if statuses is None or summary.status in statuses:
                yield summary


class PersistentJobStorage(JobStorage):
    """Archive backed by the lumid-data-app blob HTTP API."""

    def __init__(self) -> None:
        super().__init__()
        prefix = envs.S3_ARCHIVE_PREFIX
        assert prefix, "S3_ARCHIVE_PREFIX is not set"
        self.key_prefix = prefix.strip("/")

    def _put_blob(self, key: str, body: bytes, content_type: str) -> None:
        lumid_data_client.put_blob(key, body, content_type)

    def _get_blob(self, key: str) -> tuple[bytes, str]:
        try:
            return lumid_data_client.get_blob(key)
        except lumid_data_client.BlobNotFound as exc:
            raise ArchiveNotFound(key) from exc

    def _object_name(self, job_id: str) -> str:
        if self.key_prefix:
            return f"{self.key_prefix}/{job_id}/record.json"
        return f"{job_id}/record.json"

    def _job_object_name(self, job_id: str, filename: str) -> str:
        if self.key_prefix:
            return f"{self.key_prefix}/{job_id}/{filename}"
        return f"{job_id}/{filename}"

    def _output_index_name(self) -> str:
        if self.key_prefix:
            return f"{self.key_prefix}/output_index.json"
        return "output_index.json"

    def _jobs_index_name(self) -> str:
        if self.key_prefix:
            return f"{self.key_prefix}/jobs_index.json"
        return "jobs_index.json"

    def save(self, record: Any) -> None:
        data = _normalize_payload(asdict(record))
        record_data = dict(data)
        record_data.pop("inputs", None)
        record_data.pop("progress", None)
        record_data.pop("result", None)
        record_body = json.dumps(record_data, ensure_ascii=False).encode("utf-8")
        obj_name = self._object_name(record.job_id)
        self._put_json(obj_name, record_body)
        inputs_body = json.dumps(data.get("inputs", {}), ensure_ascii=False).encode(
            "utf-8"
        )
        progress_body = json.dumps(data.get("progress", {}), ensure_ascii=False).encode(
            "utf-8"
        )
        self._put_json(self._job_object_name(record.job_id, "inputs.json"), inputs_body)
        self._put_json(
            self._job_object_name(record.job_id, "progress.json"), progress_body
        )
        if data.get("result") is not None:
            result_body = json.dumps(data.get("result", {}), ensure_ascii=False).encode(
                "utf-8"
            )
            self._put_json(
                self._job_object_name(record.job_id, "result.json"), result_body
            )
        summary = _summary_from_payload(data)
        index_name = self._jobs_index_name()
        index = self._get_json_optional(index_name) or {}
        index[record.job_id] = summary.model_dump(mode="json")
        self._put_json(
            index_name, json.dumps(index, ensure_ascii=False).encode("utf-8")
        )
        self.logger.info("Saved job %s to archive blob %s", record.job_id, obj_name)

    def load(self, job_id: str) -> dict[str, Any] | None:
        obj_name = self._object_name(job_id)
        try:
            data = self._get_json(obj_name)
        except ArchiveNotFound:
            return None
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Failed to load job %s: %s", job_id, exc)
            return None
        data["inputs"] = (
            self._get_json_optional(self._job_object_name(job_id, "inputs.json")) or {}
        )
        data["progress"] = (
            self._get_json_optional(self._job_object_name(job_id, "progress.json"))
            or {}
        )
        result = self._get_json_optional(self._job_object_name(job_id, "result.json"))
        if result is not None:
            data["result"] = result
        return data

    def reserve_output_location(self, location_key: str, job_id: str) -> None:
        index_name = self._output_index_name()
        index = self._get_json_optional(index_name) or {}
        existing = index.get(location_key)
        if existing and existing != job_id:
            record = self.load(existing)
            if record and record.get("status") == "completed":
                index[location_key] = job_id
            else:
                raise ValueError(f"output location {location_key} already reserved")
        else:
            index[location_key] = job_id
        body = json.dumps(index, ensure_ascii=False).encode("utf-8")
        self._put_json(index_name, body)

    def release_output_location(self, location_key: str, job_id: str) -> None:
        index_name = self._output_index_name()
        index = self._get_json_optional(index_name)
        if index is None:
            return
        if index.get(location_key) != job_id:
            return
        del index[location_key]
        body = json.dumps(index, ensure_ascii=False).encode("utf-8")
        self._put_json(index_name, body)

    def save_artifact(
        self, job_id: str, filename: str, data: bytes, content_type: str
    ) -> str:
        object_name = self._job_object_name(job_id, f"artifacts/{filename}")
        self._put_blob(object_name, data, content_type)
        return object_name

    def get_artifact(self, job_id: str, filename: str) -> tuple[bytes, str]:
        object_name = self._job_object_name(job_id, f"artifacts/{filename}")
        try:
            return self._get_blob(object_name)
        except ArchiveNotFound as exc:
            raise KeyError(filename) from exc

    def _put_json(self, object_name: str, body: bytes) -> None:
        self._put_blob(object_name, body, "application/json")

    def _get_json(self, object_name: str) -> dict[str, Any]:
        body, _ = self._get_blob(object_name)
        return json.loads(body.decode("utf-8"))

    def _get_json_optional(self, object_name: str) -> dict[str, Any] | None:
        try:
            return self._get_json(object_name)
        except ArchiveNotFound:
            return None

    def list_summaries(
        self,
        *,
        org_id: str,
        user_id: str | None,
        job_ids: frozenset[str] | None,
        statuses: set[str] | None,
        page: int,
        page_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        index = self._get_json_optional(self._jobs_index_name()) or {}
        items: list[JobSummary] = []
        for value in index.values():
            if isinstance(value, dict):
                try:
                    items.append(JobSummary.model_validate(value))
                except ValidationError as exc:
                    self.logger.warning("Ignoring invalid job summary entry: %s", exc)
        filtered = _filter_summaries(
            items, org_id=org_id, user_id=user_id, job_ids=job_ids, statuses=statuses
        )
        sorted_items = _sort_summaries(filtered)
        total = len(sorted_items)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = [item.model_dump(mode="json") for item in sorted_items[start:end]]
        return page_items, total

    def iter_summaries(self, statuses: set[str] | None = None) -> Iterable[JobSummary]:
        index = self._get_json_optional(self._jobs_index_name()) or {}
        for value in index.values():
            if not isinstance(value, dict):
                continue
            try:
                summary = JobSummary.model_validate(value)
            except ValidationError as exc:
                self.logger.warning("Ignoring invalid job summary entry: %s", exc)
                continue
            if statuses is None or summary.status in statuses:
                yield summary


_job_storage: JobStorage | None = None


def get_job_storage() -> JobStorage:
    global _job_storage
    if _job_storage is None:
        _job_storage = PersistentJobStorage()
    return _job_storage
