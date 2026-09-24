"""Task resource operations."""

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from ..models.common import (
    TERMINAL_TASK_STATUSES,
    LogEntry,
    LogEvent,
    LogQueryResponse,
    OkResponse,
)
from ..models.tasks import TaskInfo
from ..params import append_param, extend_params
from ..ssh import task_ssh_info, wait_for_ssh_info, wait_for_ssh_info_async
from ._base import AsyncResource, SyncResource


class Tasks(SyncResource):
    """Synchronous task operations."""

    def retrieve(self, task_id: str) -> TaskInfo:
        """Retrieve task details by ID."""
        data = self._client._request("GET", f"/tasks/{task_id}")
        return TaskInfo.model_validate(data)

    def list(
        self,
        task_id: str | None = None,
        workflow_id: str | None = None,
        status: str | list[str] | None = None,
        category: str | None = None,
        task_type: str | None = None,
        assigned_worker: str | None = None,
        graph_node_name: str | None = None,
        completed: bool | None = None,
        failed: bool | None = None,
        query_params: list[tuple[str, str]] | None = None,
    ) -> list[TaskInfo]:
        """List tasks with optional filters."""
        params: list[tuple[str, str]] = []
        append_param(params, "task_id", task_id)
        append_param(params, "workflow_id", workflow_id)
        extend_params(params, "status", status)
        append_param(params, "category", category)
        append_param(params, "task_type", task_type)
        append_param(params, "assigned_worker", assigned_worker)
        append_param(params, "graph_node_name", graph_node_name)
        append_param(params, "completed", completed)
        append_param(params, "failed", failed)
        if query_params:
            params.extend(query_params)
        data = self._client._request("GET", "/tasks", params=params or None)
        return [TaskInfo.model_validate(t) for t in data]

    def stop(self, task_id: str) -> OkResponse:
        """Request a running task to stop."""
        data = self._client._request("POST", f"/tasks/{task_id}/stop")
        return OkResponse.model_validate(data)

    def get_logs(
        self,
        task_id: str,
        limit: int = 200,
        before: str | None = None,
        after: str | None = None,
    ) -> LogQueryResponse:
        """Query task logs with cursor-based pagination."""
        params: dict[str, str] = {"limit": str(limit)}
        if before:
            params["before"] = before
        if after:
            params["after"] = after
        data = self._client._request("GET", f"/tasks/{task_id}/logs", params=params)
        return LogQueryResponse.model_validate(data)

    def stream_logs(
        self,
        task_id: str,
        cursor: str | None = None,
    ) -> Iterator[LogEntry]:
        """Stream task logs via SSE. Yields ``LogEntry`` objects."""
        for event_cursor, event_data in self._client._stream_sse(
            f"/tasks/{task_id}/logs/stream", cursor=cursor
        ):
            yield LogEntry(
                cursor=event_cursor or "",
                event=LogEvent.model_validate(event_data),
            )

    def download_logs(
        self,
        task_id: str,
        output_path: Path,
    ) -> None:
        """Download archived logs.jsonl for a task."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._client._download(f"/results/{task_id}/logs", output_path)

    def wait(
        self,
        task_id: str,
        interval: float = 2.0,
    ) -> TaskInfo:
        """Poll a task until it reaches a terminal state."""
        while True:
            task = self.retrieve(task_id)
            if task.status in TERMINAL_TASK_STATUSES:
                return task
            time.sleep(interval)

    def ssh_info(self, task_id: str) -> dict[str, Any] | None:
        """Return SSH publish info for a task if present."""
        return task_ssh_info(self.retrieve(task_id))

    def wait_for_ssh(
        self,
        task_id: str,
        interval: float = 2.0,
    ) -> dict[str, Any]:
        """Poll a task until SSH session information is available."""
        return wait_for_ssh_info(
            self.retrieve, task_id, interval=interval, sleep=time.sleep
        )


class AsyncTasks(AsyncResource):
    """Asynchronous task operations."""

    async def retrieve(self, task_id: str) -> TaskInfo:
        """Retrieve task details by ID."""
        data = await self._client._request("GET", f"/tasks/{task_id}")
        return TaskInfo.model_validate(data)

    async def list(
        self,
        task_id: str | None = None,
        workflow_id: str | None = None,
        status: str | list[str] | None = None,
        category: str | None = None,
        task_type: str | None = None,
        assigned_worker: str | None = None,
        graph_node_name: str | None = None,
        completed: bool | None = None,
        failed: bool | None = None,
        query_params: list[tuple[str, str]] | None = None,
    ) -> list[TaskInfo]:
        """List tasks with optional filters."""
        params: list[tuple[str, str]] = []
        append_param(params, "task_id", task_id)
        append_param(params, "workflow_id", workflow_id)
        extend_params(params, "status", status)
        append_param(params, "category", category)
        append_param(params, "task_type", task_type)
        append_param(params, "assigned_worker", assigned_worker)
        append_param(params, "graph_node_name", graph_node_name)
        append_param(params, "completed", completed)
        append_param(params, "failed", failed)
        if query_params:
            params.extend(query_params)
        data = await self._client._request("GET", "/tasks", params=params or None)
        return [TaskInfo.model_validate(t) for t in data]

    async def stop(self, task_id: str) -> OkResponse:
        """Request a running task to stop."""
        data = await self._client._request("POST", f"/tasks/{task_id}/stop")
        return OkResponse.model_validate(data)

    async def get_logs(
        self,
        task_id: str,
        limit: int = 200,
        before: str | None = None,
        after: str | None = None,
    ) -> LogQueryResponse:
        """Query task logs with cursor-based pagination."""
        params: dict[str, str] = {"limit": str(limit)}
        if before:
            params["before"] = before
        if after:
            params["after"] = after
        data = await self._client._request(
            "GET", f"/tasks/{task_id}/logs", params=params
        )
        return LogQueryResponse.model_validate(data)

    async def stream_logs(
        self,
        task_id: str,
        cursor: str | None = None,
    ) -> AsyncIterator[LogEntry]:
        """Stream task logs via SSE. Yields ``LogEntry`` objects."""
        async for event_cursor, event_data in self._client._stream_sse(
            f"/tasks/{task_id}/logs/stream", cursor=cursor
        ):
            yield LogEntry(
                cursor=event_cursor or "",
                event=LogEvent.model_validate(event_data),
            )

    async def download_logs(
        self,
        task_id: str,
        output_path: Path,
    ) -> None:
        """Download archived logs.jsonl for a task."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await self._client._download(f"/results/{task_id}/logs", output_path)

    async def wait(
        self,
        task_id: str,
        interval: float = 2.0,
    ) -> TaskInfo:
        """Poll a task until it reaches a terminal state."""
        while True:
            task = await self.retrieve(task_id)
            if task.status in TERMINAL_TASK_STATUSES:
                return task
            await asyncio.sleep(interval)

    async def ssh_info(self, task_id: str) -> dict[str, Any] | None:
        """Return SSH publish info for a task if present."""
        task = await self.retrieve(task_id)
        return task_ssh_info(task)

    async def wait_for_ssh(
        self,
        task_id: str,
        interval: float = 2.0,
    ) -> dict[str, Any]:
        """Poll a task until SSH session information is available."""
        return await wait_for_ssh_info_async(
            self.retrieve,
            task_id,
            interval=interval,
            sleep=asyncio.sleep,
        )
