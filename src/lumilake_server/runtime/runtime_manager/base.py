"""Base classes for runtime execution backends."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.request import RequestInfo
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.utils.job_storage import get_job_storage


class BaseRuntimeManager(ABC):
    """Interface for runtime execution backends."""

    @abstractmethod
    def result_dir(self, request_info: RequestInfo) -> Path:
        """Return per-batch artifact directory for the request."""

    def save_runtime_artifact(
        self,
        request_info: RequestInfo,
        filename: str,
        data: bytes,
        content_type: str,
    ) -> str:
        return get_job_storage().save_artifact(
            request_info.request_id,
            f"runtime/{request_info.batch_id}/{filename}",
            data,
            content_type,
        )

    @abstractmethod
    async def get_workers(self) -> list[str]:
        """Return available worker IDs."""

    @abstractmethod
    async def get_worker_profile(self, worker_id: str) -> dict[str, Any]:
        """Return hardware profile for a worker."""

    @abstractmethod
    def count_runtime_nodes(self, graphs: dict[str, RuntimeGraph]) -> int:
        """Return the number of runtime nodes that will be submitted."""

    @abstractmethod
    def mark_batch_pending(
        self,
        request_id: str,
        batch_id: str,
        total_nodes: int,
        output_nodes: int,
    ) -> None:
        """Mark a batch pending."""

    @abstractmethod
    def mark_batch_running(self, request_id: str, batch_id: str) -> None:
        """Mark a batch running."""

    @abstractmethod
    def mark_batch_completed(self, request_id: str, batch_id: str) -> None:
        """Mark a batch completed."""

    @abstractmethod
    def mark_batch_failed(self, request_id: str, batch_id: str) -> None:
        """Mark a batch failed."""

    @abstractmethod
    async def process_request(
        self,
        request_info: RequestInfo,
        schedule: Schedule,
        worker_ids: list[str],
        data_profile_results: dict[str, list[dict[str, Any]]] | None,
    ) -> dict[str, Any]:
        """Submit a request for execution and return outputs.

        `schedule` contains worker assignment only.
        """

    @abstractmethod
    async def get_request_status(self, request_id: str) -> dict[str, Any]:
        """Return request status from backend."""

    @abstractmethod
    async def cancel_request(self, request_id: str) -> None:
        """Cancel a request."""

    def get_task_node_map(self, request_id: str, batch_id: str) -> dict[str, str]:
        """Return task-id -> node-id mapping for a batch, if available."""
        return {}

    async def is_request_cancelled(self, request_id: str) -> bool:
        """Return whether a request has been cancelled."""
        return False
