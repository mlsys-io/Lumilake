import asyncio
import copy
import hashlib
import json
import math
import multiprocessing as mp
import queue
import time
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import yaml
from lumilake import envs
from lumilake.log import (
    Logger,
    LogLevel,
    init_child_logger,
    log_on_exception_async,
)

from lumilake_server.graphs import CompiledGraph, Graph
from lumilake_server.ops import DataRetrievalOp, LLMChatOp
from lumilake_server.runtime.data_profile_utils import (
    DataProfileSource,
    collect_data_profile,
)
from lumilake_server.runtime.job_manager import (
    BaseJobManager,
    BatchSelection,
    Job,
    create_job_manager,
)
from lumilake_server.runtime.optimizer import create_optimizer
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.protocol import (
    LumilakeRequest,
    LumilakeRequestConfig,
    LumilakeResponse,
    Priority,
    RequestCancelledError,
)
from lumilake_server.runtime.request import (
    RequestHandler,
    RequestInfo,
    WorkflowSliceMeta,
)
from lumilake_server.runtime.runtime_graph import RuntimeGraph, RuntimeGraphBuilder
from lumilake_server.runtime.runtime_manager import (
    FlowmeshRuntimeManager,
    create_runtime_manager,
)
from lumilake_server.runtime.utils.loop import AsyncEventLoop
from lumilake_server.runtime.utils.queue import TSQueue
from lumilake_server.schemas.progress import JobProgress
from lumilake_server.utils.job_storage import get_job_storage
from lumilake_server.utils.utils import async_runner, stop_async_runner, unique_id


@dataclass(slots=True)
class RequestState:
    handler: RequestHandler
    config: LumilakeRequestConfig
    pending_workflows: set[str]
    outputs: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    chat_histories: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    output_buffers: dict[str, dict[str, list[str | None]]] = field(default_factory=dict)
    chat_history_buffers: dict[str, dict[str, list[Any | None]]] = field(
        default_factory=dict
    )
    workflow_lengths: dict[str, int] = field(default_factory=dict)
    error_info: list[dict[str, Any]] | None = None
    task_node_map: dict[str, dict[str, str]] = field(default_factory=dict)
    plan_cache: dict[str, "PlanCacheEntry"] = field(default_factory=dict)
    pending_runtime_nodes_raw: int = 0
    processing_runtime_nodes_raw: int = 0
    processing_runtime_nodes_optimized: int = 0
    processed_runtime_nodes_raw: int = 0
    processed_runtime_nodes_optimized: int = 0
    optimization_seconds: float = 0.0
    batch_node_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    total_input_items: int = 0
    completed_input_items_success: int = 0
    successful_workflow_ids: set[str] = field(default_factory=set)
    ready: bool = False


@dataclass(slots=True)
class PlanCacheEntry:
    data_profile_results: dict[str, list[dict[str, Any]]]
    schedule: Schedule


@dataclass(slots=True)
class ExecutionBatchContext:
    execution_request_id: str
    batch_id: str
    request_ids: tuple[str, ...]
    workflow_ids: tuple[str, ...]


@dataclass(slots=True)
class SchedulePreview:
    request_id: str
    selected_workers: list[str]
    worker_profiles: dict[str, dict[str, Any]]
    runtime_graph_node_counts: dict[str, int]
    merged_runtime_node_count: int
    schedule: Schedule


def _optimizer_subprocess_entry(
    optimizer_type: str,
    runtime_graph: Any,
    selected_workers: list[str],
    worker_profiles: dict[str, dict[str, Any]],
    data_profile_results: dict[str, list[dict[str, Any]]],
    result_queue: Any,
) -> None:
    try:
        optimizer = create_optimizer(optimizer_type=optimizer_type)
        schedule = optimizer.generate_schedule(
            runtime_graph,
            selected_workers,
            worker_profiles,
            data_profile_results,
        )
        result_queue.put(
            {
                "ok": True,
                "schedule": schedule,
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


class LumilakeServerError(Exception):
    pass


class LumilakeServerConfig:
    """Configuration for Lumilake logical optimizer server."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        is_local: bool = False,
        runtime_url: str | None = None,
        runtime_token: str | None = None,
        batch_size: int = envs.LUMILAKE_OPTIMIZER_BATCH_SIZE,
        batch_accumulation_seconds: float = envs.LUMILAKE_BATCH_ACCUMULATION_SECONDS,
        cpu_worker_group_size: int = envs.LUMILAKE_CPU_WORKER_GROUP_SIZE,
        gpu_worker_group_size: int = envs.LUMILAKE_GPU_WORKER_GROUP_SIZE,
        queue_quantums: dict[Priority, int] | None = None,
        starvation_limit: int = envs.LUMILAKE_STARVATION_LIMIT,
    ) -> None:
        self.is_local = is_local
        """Whether to use a local Lumilake server."""
        self.runtime_url = runtime_url
        """Runtime orchestrator URL for plan submission."""
        self.runtime_token = runtime_token
        """Authentication token for runtime orchestrator."""
        self.batch_size = batch_size
        """Number of graphs per batch for workload processing."""
        self.batch_accumulation_seconds = batch_accumulation_seconds
        """Optional queue accumulation window before selecting/optimizing a batch."""
        self.cpu_worker_group_size = max(0, cpu_worker_group_size)
        """Minimum number of CPU-only workers required before dispatch."""
        self.gpu_worker_group_size = max(0, gpu_worker_group_size)
        """Minimum number of GPU workers required before dispatch."""
        if self.cpu_worker_group_size == 0 and self.gpu_worker_group_size == 0:
            raise ValueError(
                "At least one of cpu_worker_group_size or gpu_worker_group_size must"
                " be > 0"
            )
        self.queue_quantums = queue_quantums
        """Queue quantums per priority (high/medium/low)."""
        self.starvation_limit = starvation_limit
        """Number of candidate misses before forcing selection."""

        if is_local:
            self._host = self._port = None
        else:
            self._host = host or envs.LUMILAKE_SERVER_HOST
            self._port = port or envs.LUMILAKE_SERVER_PORT

    @property
    def host(self) -> str:
        """Hostname of the Lumilake server."""
        if self._host is None:
            raise ValueError("Host is not set")
        return self._host

    @property
    def port(self) -> int:
        """Port of the Lumilake server."""
        if self._port is None:
            raise ValueError("Port is not set")
        return self._port

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class LumilakeServer:
    _instance: "LumilakeServer | None" = None

    def __init__(
        self,
        config: LumilakeServerConfig | None = None,
        logger: Logger | None = None,
        log_level: LogLevel | None = None,
    ) -> None:
        self.logger: Logger = init_child_logger("Server", logger, log_level)
        if self.__class__._instance is not None:
            self.logger.warning(
                "Server instance already exists. "
                "You may want to use LumilakeServer.get_instance() instead."
            )
        else:
            self.__class__._instance = self

        self.config = LumilakeServerConfig() if config is None else config

        # Initialize logical optimizer (no physical execution dependencies)
        # Use the optimizer factory with default settings
        self.optimizer = create_optimizer(logger=self.logger)

        # Initialize job manager and runtime manager
        self.job_manager: BaseJobManager = create_job_manager(
            optimizer=self.optimizer,
            quantums=self.config.queue_quantums,
            starvation_limit=self.config.starvation_limit,
            logger=self.logger,
        )
        # FlowmeshRuntimeManager reads orchestrator URL + token from envs directly
        # via lumilake_server.runtime.flowmesh_client; no need to pass them here.
        self.runtime_manager: FlowmeshRuntimeManager = create_runtime_manager(
            logger=self.logger,
        )
        self._runtime_builder = RuntimeGraphBuilder(logger=self.logger)

        self._event_loop: AsyncEventLoop[RequestHandler, None, None] | None = None
        self._scheduler_task: asyncio.Task[Any] | None = None
        self._inflight_tasks: set[asyncio.Task[Any]] = set()
        self._worker_lock = asyncio.Lock()
        self._busy_workers: set[str] = set()
        self._optimizer_lock = asyncio.Lock()

        self._requests: dict[str, RequestState] = {}
        self._execution_contexts: dict[str, ExecutionBatchContext] = {}
        self._request_execution_ids: dict[str, set[str]] = {}
        self._request_execution_history_ids: dict[str, set[str]] = {}

        self._progress: dict[str, JobProgress] = {}
        self._optimizer_type = envs.LUMILAKE_OPTIMIZER_TYPE

    @property
    def is_started(self) -> bool:
        return self._event_loop is not None and self._event_loop.is_started()

    def start(self) -> None:
        async def inner(event_loop: AsyncEventLoop) -> None:
            self.logger.info("Starting logical optimizer server...")
            await event_loop.start()
            if self._scheduler_task is None or self._scheduler_task.done():
                self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            self.logger.info("Server started.")

        if self._event_loop is not None and self._event_loop.is_started():
            self.logger.warning("Server has already been started.")
            return
        self._event_loop = AsyncEventLoop(
            handler_func=self.handle_request, in_channel=TSQueue()
        )
        async_runner().run(inner(self._event_loop))

    def close(self) -> None:
        async def inner(event_loop: AsyncEventLoop) -> None:
            # Cancel all active tasks before shutting down
            self.logger.info("Cancelling all active tasks...")

            if self._scheduler_task is not None:
                self._scheduler_task.cancel()

            for task in list(self._inflight_tasks):
                task.cancel()

            for execution_request_id in list(self._execution_contexts.keys()):
                try:
                    await self.runtime_manager.cancel_request(execution_request_id)
                except Exception:
                    self.logger.warning(
                        "Failed to cancel execution request %s",
                        execution_request_id,
                        exc_info=True,
                    )

            # Cancel tasks for all active requests
            for request_id in list(self._requests.keys()):
                try:
                    await self.runtime_manager.cancel_request(request_id)
                except Exception:
                    self.logger.warning(
                        "Failed to cancel request %s", request_id, exc_info=True
                    )
            try:
                from lumilake_server.routes import jobs as job_routes

                await job_routes.mark_running_jobs_failed("server shutdown")
            except Exception:
                self.logger.warning(
                    "Failed to mark running jobs as failed during shutdown",
                    exc_info=True,
                )

            # Continue with existing shutdown logic
            event_loop.loop_task.cancel()
            self.logger.info("Terminating server...")
            await event_loop.stop()
            self.logger.info("Server terminated.")
            self.__class__._instance = None

        if self._event_loop is None or not self._event_loop.is_started():
            self.logger.warning("Server has not been started.")
            return
        if self._event_loop.is_stopped():
            self.logger.warning("Server is already stopped.")
            return

        async_runner().run(inner(self._event_loop))
        stop_async_runner()
        self._event_loop = None

    def __del__(self) -> None:
        if self._event_loop is not None:
            self.close()

    async def get_request_status(
        self,
        request_id: str,
    ) -> dict[str, Any]:
        """Get the current status of a request by its ID."""
        if request_id not in self._progress:
            return {"error": "Request ID not found"}
        progress = self._progress[request_id].model_copy(deep=True)
        execution_ids = sorted(
            self._request_execution_ids.get(request_id, set())
            | self._request_execution_history_ids.get(request_id, set())
        )
        runtime_statuses: list[dict[str, Any]] = []
        if execution_ids:
            for execution_id in execution_ids:
                status = await self.runtime_manager.get_request_status(execution_id)
                if "error" not in status:
                    runtime_statuses.append(status)
        else:
            status = await self.runtime_manager.get_request_status(request_id)
            if "error" not in status:
                runtime_statuses.append(status)
        state = self._requests.get(request_id)
        if runtime_statuses:
            merged_status = self._merge_runtime_statuses(runtime_statuses)
            progress.apply_status(merged_status)
        if state is not None:
            overall = progress.batch_progress.overall_progress
            overall.total_nodes_runtime = (
                state.processing_runtime_nodes_optimized
                + state.processed_runtime_nodes_optimized
            )
            total_inputs = max(0, int(state.total_input_items))
            completed_inputs = max(0, int(state.completed_input_items_success))
            overall.total_inputs = total_inputs
            overall.completed_inputs = (
                min(completed_inputs, total_inputs)
                if total_inputs > 0
                else completed_inputs
            )
            overall.pending_runtime_nodes_raw = state.pending_runtime_nodes_raw
            overall.processing_runtime_nodes_raw = state.processing_runtime_nodes_raw
            overall.processing_runtime_nodes_optimized = (
                state.processing_runtime_nodes_optimized
            )
            overall.processed_runtime_nodes_raw = state.processed_runtime_nodes_raw
            overall.processed_runtime_nodes_optimized = (
                state.processed_runtime_nodes_optimized
            )
            percentage = (
                overall.completed_nodes / overall.total_nodes_runtime * 100
                if overall.total_nodes_runtime > 0
                else 0.0
            )
            overall.percentage = min(100.0, max(0.0, percentage))
        return progress.model_dump(by_alias=True)

    @staticmethod
    def _merge_step_details(
        statuses: list[dict[str, Any]], key: str
    ) -> dict[str, Any] | None:
        merged_details = {
            "succeeded": 0,
            "failed": 0,
            "pending": 0,
            "dispatched": 0,
        }
        found = False
        completed = True
        has_details = False
        for status in statuses:
            step = status.get(key)
            if not isinstance(step, dict):
                continue
            found = True
            completed = completed and bool(step.get("completed", False))
            details = step.get("details")
            if not isinstance(details, dict):
                continue
            has_details = True
            for detail_key in merged_details:
                raw_value = details.get(detail_key, 0)
                if isinstance(raw_value, (int, float)):
                    merged_details[detail_key] += int(raw_value)
        if not found:
            return None
        payload: dict[str, Any] = {"completed": completed}
        if has_details:
            payload["details"] = merged_details
        return payload

    @classmethod
    def _merge_runtime_statuses(cls, statuses: list[dict[str, Any]]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for key in ("data probing", "execution", "outputs"):
            payload = cls._merge_step_details(statuses, key)
            if payload is not None:
                merged[key] = payload

        batch_totals = {
            "total": 0,
            "completed": 0,
            "running": 0,
            "pending": 0,
            "failed": 0,
        }
        batch_entries: list[dict[str, Any]] = []
        overall_totals = {
            "total_nodes": 0,
            "completed_nodes": 0,
            "raw_nodes": 0,
            "flowmesh_nodes": 0,
        }
        eta_values: list[float] = []
        for status in statuses:
            batch_progress = status.get("batch_progress")
            if not isinstance(batch_progress, dict):
                continue
            for key in batch_totals:
                value = batch_progress.get(key, 0)
                if isinstance(value, (int, float)):
                    batch_totals[key] += int(value)
            batches = batch_progress.get("batches")
            if isinstance(batches, list):
                for item in batches:
                    if isinstance(item, dict):
                        batch_entries.append(item)
            overall = batch_progress.get("overall_progress")
            if isinstance(overall, dict):
                for key in overall_totals:
                    value = overall.get(key, 0)
                    if isinstance(value, (int, float)):
                        overall_totals[key] += int(value)
            eta = batch_progress.get("eta_seconds")
            if isinstance(eta, (int, float)):
                eta_values.append(float(eta))

        if any(batch_totals.values()) or batch_entries:
            percentage = 0.0
            if overall_totals["total_nodes"] > 0:
                percentage = round(
                    overall_totals["completed_nodes"]
                    / overall_totals["total_nodes"]
                    * 100,
                    1,
                )
            merged["batch_progress"] = {
                **batch_totals,
                "batches": sorted(
                    batch_entries, key=lambda item: str(item.get("batch_id", ""))
                ),
                "overall_progress": {
                    "total_nodes": overall_totals["total_nodes"],
                    "completed_nodes": overall_totals["completed_nodes"],
                    "percentage": percentage,
                    "raw_nodes": overall_totals["raw_nodes"],
                    "flowmesh_nodes": overall_totals["flowmesh_nodes"],
                },
                "eta_seconds": max(eta_values) if eta_values else None,
            }
        return merged

    def trace_ids_for_request(self, request_id: str) -> list[str]:
        """FlowMesh workflow ids submitted under this lumilake request.

        A lumilake request fans out into one or more execution batches —
        each ``execution_request_id`` keys an entry in the runtime manager's
        ``_batch_workflow_id`` map. Resolve through the request → execution
        index first, then collect the FM workflow ids from each execution.

        Called from the FastAPI event loop; ``runtime_manager._batch_workflow_id``
        is mutated on the runtime's ``_AsyncRunner`` thread, so snapshot via
        ``.copy()`` before iterating.
        """
        execution_ids = self._request_execution_ids.get(
            request_id, set()
        ) | self._request_execution_history_ids.get(request_id, set())
        batch_snapshot = self.runtime_manager._batch_workflow_id.copy()
        return list(
            {wf for (rid, _), wf in batch_snapshot.items() if rid in execution_ids}
        )

    def release_request_workflows(self, request_id: str) -> None:
        """Drop the runtime manager's per-batch FM workflow id entries for
        this request once ``trace_ids`` have been persisted to JobStorage."""
        execution_ids = self._request_execution_ids.get(
            request_id, set()
        ) | self._request_execution_history_ids.get(request_id, set())
        if execution_ids:
            self.runtime_manager.release_executions(execution_ids)

    def optimization_seconds_for_request(self, request_id: str) -> float | None:
        state = self._requests.get(request_id)
        if state is None:
            return None
        return float(state.optimization_seconds)

    @log_on_exception_async()
    async def handle_request(self, request: RequestHandler, _) -> None:
        """
        Core request handler for workload processing with background batching.
        """
        # Initialize progress tracking for this request
        progress = JobProgress()
        progress.query_parsing.completed = True
        self._progress[request.request_id] = progress

        self.logger.info(
            "Queueing request %s with %d graphs",
            request.request_id,
            len(request.query),
        )
        self.logger.info(
            "Splitting request %s into %d workflows for queueing",
            request.request_id,
            len(request.query),
        )

        if not request.query:
            await request.put_result(
                LumilakeResponse(outputs={}, error_info=None, chat_histories={})
            )
            return

        if request.dsl_graphs is None:
            raise RuntimeError("DSL graphs are required before queueing")

        config = request.config
        pending_runtime_nodes_raw = sum(
            graph.node_count for graph in request.query.values()
        )
        workflow_lengths: dict[str, int] = {}
        for graph_name, slice_meta in request.workflow_slices.items():
            current = workflow_lengths.get(slice_meta.public_graph_name, 0)
            workflow_lengths[slice_meta.public_graph_name] = max(
                current,
                slice_meta.total_length,
            )
            if graph_name not in request.query:
                raise RuntimeError(
                    f"Missing runtime graph for workflow slice {graph_name}"
                )
        state = RequestState(
            handler=request,
            config=config,
            pending_workflows=set(),
            workflow_lengths=workflow_lengths,
            pending_runtime_nodes_raw=pending_runtime_nodes_raw,
            total_input_items=sum(
                max(0, int(length)) for length in workflow_lengths.values()
            ),
        )
        self._requests[request.request_id] = state

        job = Job(
            request_id=request.request_id,
            runtime_graphs=request.query,
            data_profile_graphs=request.data_profile_graphs,
            dsl_graphs=request.dsl_graphs,
            workflow_slices=request.workflow_slices,
            config=config,
        )
        enqueued = await self.job_manager.enqueue(job)
        for item in enqueued:
            state.pending_workflows.add(item.workflow_id)

        self._progress[request.request_id].queuing.completed = True
        state.ready = True

    async def _scheduler_loop(self) -> None:
        while True:
            try:
                await self.job_manager.wait_for_work()
                await self._wait_for_batch_accumulation()
                workers = await self._wait_for_available_worker_group(
                    cpu_group_size=self.config.cpu_worker_group_size,
                    gpu_group_size=self.config.gpu_worker_group_size,
                )
                if workers is None:
                    continue
                batch = await self.job_manager.select_batch(self.config.batch_size)
                if batch is None:
                    async with self._worker_lock:
                        self._busy_workers.difference_update(workers)
                    continue
                self.logger.info(
                    "Dispatching batch (size=%d) to workers %s "
                    "(cpu_group_size=%d gpu_group_size=%d)",
                    len(batch.workflows),
                    workers,
                    self.config.cpu_worker_group_size,
                    self.config.gpu_worker_group_size,
                )
                task = asyncio.create_task(self._run_batch(workers, batch))
                self._inflight_tasks.add(task)
                task.add_done_callback(self._inflight_tasks.discard)
            except asyncio.CancelledError:
                return
            except Exception:
                self.logger.exception("Scheduler loop encountered an error")
                await asyncio.sleep(envs.LUMILAKE_POLL_INTERVAL_SECONDS)

    async def _wait_for_batch_accumulation(self) -> None:
        wait_seconds = self.config.batch_accumulation_seconds
        if wait_seconds <= 0:
            return
        pending_count, oldest_enqueued_at = await self.job_manager.get_pending_stats()
        if pending_count <= 0 or oldest_enqueued_at is None:
            return
        elapsed = time.time() - oldest_enqueued_at
        remaining = wait_seconds - elapsed
        if remaining <= 0:
            return
        self.logger.info(
            "Accumulating queued workflows before optimization "
            "(pending=%d, wait=%.2fs, elapsed=%.2fs, remaining=%.2fs)",
            pending_count,
            wait_seconds,
            elapsed,
            remaining,
        )
        await asyncio.sleep(remaining)

    async def _wait_for_available_worker_group(
        self, cpu_group_size: int, gpu_group_size: int
    ) -> list[str] | None:
        start_time = time.perf_counter()
        wait_warning_interval_s = max(30.0, envs.LUMILAKE_POLL_INTERVAL_SECONDS)
        next_wait_warning_at = start_time + wait_warning_interval_s
        required_cpu = max(0, cpu_group_size)
        required_gpu = max(0, gpu_group_size)
        if required_cpu == 0 and required_gpu == 0:
            raise ValueError("CPU and GPU worker group sizes cannot both be zero")
        while True:
            workers = await self.runtime_manager.get_workers()
            async with self._worker_lock:
                busy_worker_count = len(self._busy_workers)
                available_workers = [w for w in workers if w not in self._busy_workers]
            available_gpu_workers: list[str] = []
            available_cpu_workers: list[str] = []
            for candidate in available_workers:
                try:
                    profile = await self.runtime_manager.get_worker_profile(candidate)
                except Exception:
                    self.logger.warning(
                        "Failed to fetch worker profile for %s; skipping",
                        candidate,
                        exc_info=True,
                    )
                    continue
                if self._has_gpu(profile):
                    available_gpu_workers.append(candidate)
                else:
                    available_cpu_workers.append(candidate)

            if (
                len(available_cpu_workers) >= required_cpu
                and len(available_gpu_workers) >= required_gpu
            ):
                selected_cpu_workers = available_cpu_workers[:required_cpu]
                selected_gpu_workers = available_gpu_workers[:required_gpu]
                selected_workers = selected_gpu_workers + selected_cpu_workers
                async with self._worker_lock:
                    if any(worker in self._busy_workers for worker in selected_workers):
                        continue
                    self._busy_workers.update(selected_workers)
                self.logger.debug(
                    "Workers available for dispatch: %s (available_cpu=%d"
                    " required_cpu=%d available_gpu=%d required_gpu=%d)",
                    selected_workers,
                    len(available_cpu_workers),
                    required_cpu,
                    len(available_gpu_workers),
                    required_gpu,
                )
                return selected_workers
            now = time.perf_counter()
            elapsed = now - start_time
            if now >= next_wait_warning_at:
                self.logger.warning(
                    "Waiting for worker group (required_cpu=%d required_gpu=%d"
                    " available_cpu=%d available_gpu=%d available_workers=%d"
                    " busy_workers=%d elapsed=%.1fs)",
                    required_cpu,
                    required_gpu,
                    len(available_cpu_workers),
                    len(available_gpu_workers),
                    len(available_workers),
                    busy_worker_count,
                    elapsed,
                )
                next_wait_warning_at = now + wait_warning_interval_s
            if (
                envs.LUMILAKE_POLL_TIMEOUT_SECONDS != float("inf")
                and elapsed > envs.LUMILAKE_POLL_TIMEOUT_SECONDS
            ):
                return None
            await asyncio.sleep(envs.LUMILAKE_POLL_INTERVAL_SECONDS)

    async def _select_preview_workers_and_profiles(
        self, runtime_graph: RuntimeGraph
    ) -> tuple[list[str], dict[str, dict[str, Any]]]:
        workers = await self.runtime_manager.get_workers()
        if not workers:
            raise RuntimeError("No workers available for schedule preview")

        requires_gpu = any(
            self._is_gpu_runtime_backend(op.backend)
            for op in runtime_graph.nodes.values()
        )
        requires_cpu = any(
            op.task_type == "data_retrieval" for op in runtime_graph.nodes.values()
        )

        gpu_workers: list[str] = []
        cpu_workers: list[str] = []
        normalized_profiles: dict[str, dict[str, Any]] = {}
        for worker in workers:
            profile = self._normalize_worker_profile(
                await self.runtime_manager.get_worker_profile(worker)
            )
            normalized_profiles[worker] = profile
            if profile["has_gpu"]:
                gpu_workers.append(worker)
            else:
                cpu_workers.append(worker)

        selected_workers: list[str] = []
        if requires_gpu:
            if not gpu_workers:
                raise RuntimeError(
                    "No GPU worker available for schedule preview, but graph contains"
                    " GPU nodes"
                )
            selected_workers.append(gpu_workers[0])
        if requires_cpu:
            if not cpu_workers:
                raise RuntimeError(
                    "No CPU worker available for schedule preview, but graph contains"
                    " data-retrieval nodes"
                )
            selected_workers.append(cpu_workers[0])
        if not selected_workers:
            selected_workers.append(workers[0])

        selected_profiles = {
            worker_id: normalized_profiles[worker_id] for worker_id in selected_workers
        }
        return selected_workers, selected_profiles

    @staticmethod
    def _schedule_node_names(schedule: dict[str, list[str]]) -> list[str]:
        names: list[str] = []
        for worker, worker_schedule in schedule.items():
            if not isinstance(worker, str) or not worker.strip():
                raise ValueError("Invalid worker id in schedule mapping")
            if not isinstance(worker_schedule, list):
                raise ValueError(
                    f"Invalid schedule for worker {worker!r}; expected list of node"
                    " IDs."
                )
            for node_name in worker_schedule:
                if not isinstance(node_name, str) or not node_name.strip():
                    raise ValueError(
                        f"Invalid node id in schedule for worker {worker!r}."
                    )
                names.append(node_name)
        return names

    @staticmethod
    def _has_gpu(worker_profile: dict[str, Any]) -> bool:
        try:
            devices = worker_profile["gpu"]["devices"]
        except (KeyError, TypeError):
            return False
        return isinstance(devices, list) and len(devices) > 0

    @staticmethod
    def _is_gpu_runtime_backend(backend: str) -> bool:
        return backend in {"vllm", "transformers", "diffusers"}

    @classmethod
    def _normalize_worker_profile(
        cls, worker_profile: dict[str, Any]
    ) -> dict[str, Any]:
        has_gpu = worker_profile.get("has_gpu")
        if isinstance(has_gpu, bool):
            return {"has_gpu": has_gpu}
        return {
            "has_gpu": cls._has_gpu(worker_profile),
        }

    @staticmethod
    def _request_workflow_parent_id(workflow: Any) -> str:
        # Each input slice gets its own parent id so multi-input
        # submissions (``Stock=["NVDA","AAPL","TSLA"]`` with batch-size 1)
        # dispatch as N independent workflow executions instead of being
        # coalesced back into a single graph carrying the full list. The
        # worker's SQL template substitution only handles scalar values;
        # a coalesced list-of-3 produces ``WHERE symbol = '["NVDA",...]'``.
        return (
            f"request::{workflow.request_id}::"
            f"{workflow.public_graph_name}::{workflow.template_hash}"
            f"::slice_{workflow.slice_index}"
        )

    @classmethod
    def _request_data_profile_task_key(cls, workflow: Any) -> str:
        return cls._request_workflow_parent_id(workflow)

    @classmethod
    def _group_workflows_by_parent_workflow(
        cls,
        workflows: list[Any],
    ) -> dict[str, list[Any]]:
        if not workflows:
            return {}
        # Keep slices from the same request workflow together; do not split one
        # request's slices into cross-request composite groups.
        grouped: dict[str, list[Any]] = {}
        for workflow in workflows:
            parent_workflow_id = cls._request_workflow_parent_id(workflow)
            grouped.setdefault(parent_workflow_id, []).append(workflow)
        for parent_workflow_id, items in grouped.items():
            grouped[parent_workflow_id] = sorted(
                items,
                key=lambda item: (
                    item.request_id,
                    item.public_graph_name,
                    item.slice_index,
                    item.workflow_id,
                ),
            )
        return grouped

    async def _collect_cancelled_requests(
        self,
        request_ids: set[str],
    ) -> set[str]:
        cancelled: set[str] = set()
        for request_id in request_ids:
            if await self.runtime_manager.is_request_cancelled(request_id):
                cancelled.add(request_id)
        return cancelled

    async def cancel_request(self, request_id: str) -> None:
        await self.runtime_manager.cancel_request(request_id)
        execution_ids = tuple(self._request_execution_ids.get(request_id, set()))
        for execution_id in execution_ids:
            context = self._execution_contexts.get(execution_id)
            if context is None:
                continue
            await self._should_cancel_execution(execution_id, set(context.request_ids))

    async def _should_cancel_execution(
        self,
        execution_request_id: str,
        member_request_ids: set[str],
    ) -> bool:
        if await self.runtime_manager.is_request_cancelled(execution_request_id):
            return True
        cancelled_members = await self._collect_cancelled_requests(member_request_ids)
        if len(cancelled_members) < len(member_request_ids):
            return False
        await self.runtime_manager.cancel_request(execution_request_id)
        return True

    @staticmethod
    def _artifact_filename_from_uri(value: str) -> str | None:
        if "/artifacts/" not in value:
            return None
        trimmed = value.split("?", 1)[0].split("#", 1)[0].rstrip("/")
        filename = trimmed.rsplit("/artifacts/", 1)[-1].split("/")[-1]
        return filename or None

    def _relocate_artifact_uri(
        self,
        uri: str,
        *,
        source_request_id: str,
        target_request_id: str,
        cache: dict[tuple[str, str], str],
    ) -> str:
        if source_request_id == target_request_id:
            return uri
        if source_request_id not in uri:
            return uri
        filename = self._artifact_filename_from_uri(uri)
        if filename is None:
            return uri
        cache_key = (target_request_id, filename)
        if cache_key in cache:
            return cache[cache_key]
        storage = get_job_storage()
        try:
            data, content_type = storage.get_artifact(source_request_id, filename)
        except KeyError:
            return uri.replace(
                f"/{source_request_id}/artifacts/",
                f"/{target_request_id}/artifacts/",
            )
        relocated = storage.save_artifact(
            target_request_id, filename, data, content_type
        )
        cache[cache_key] = relocated
        return relocated

    def _relocate_artifacts_for_request(
        self,
        value: Any,
        *,
        source_request_id: str,
        target_request_id: str,
        cache: dict[tuple[str, str], str],
    ) -> Any:
        if isinstance(value, str):
            return self._relocate_artifact_uri(
                value,
                source_request_id=source_request_id,
                target_request_id=target_request_id,
                cache=cache,
            )
        if isinstance(value, list):
            return [
                self._relocate_artifacts_for_request(
                    item,
                    source_request_id=source_request_id,
                    target_request_id=target_request_id,
                    cache=cache,
                )
                for item in value
            ]
        if isinstance(value, dict):
            return {
                key: self._relocate_artifacts_for_request(
                    item,
                    source_request_id=source_request_id,
                    target_request_id=target_request_id,
                    cache=cache,
                )
                for key, item in value.items()
            }
        return value

    def _save_runtime_artifact(
        self,
        request_info: RequestInfo,
        filename: str,
        data: bytes,
        content_type: str,
    ) -> str:
        return self.runtime_manager.save_runtime_artifact(
            request_info,
            filename,
            data,
            content_type,
        )

    @classmethod
    def _validate_schedule(
        cls,
        schedule: Schedule,
        selected_workers: list[str],
        runtime_nodes: set[str],
    ) -> None:
        selected_workers_set = set(selected_workers)
        schedule_workers = set(schedule.worker_assignment.keys())
        if schedule_workers != selected_workers_set:
            missing = sorted(selected_workers_set - schedule_workers)
            extra = sorted(schedule_workers - selected_workers_set)
            raise ValueError(
                "Schedule workers must match selected worker group. "
                f"Missing: {missing}, extra: {extra}"
            )
        schedule_names = cls._schedule_node_names(schedule.worker_assignment)
        schedule_nodes = set(schedule_names)
        if schedule_nodes != runtime_nodes:
            missing = sorted(runtime_nodes - schedule_nodes)
            extra = sorted(schedule_nodes - runtime_nodes)
            raise ValueError(
                "Schedule must match runtime graph nodes. "
                f"Missing: {missing}, extra: {extra}"
            )
        for worker, worker_nodes in schedule.worker_assignment.items():
            if len(worker_nodes) != len(set(worker_nodes)):
                raise ValueError(
                    "Schedule worker_assignment has duplicate node IDs for worker"
                    f" '{worker}'"
                )

    @staticmethod
    def _build_plan_cache_key(
        selected_workers: list[str],
        grouped_workflows: dict[str, list[Any]],
    ) -> str:
        groups_payload: list[dict[str, Any]] = []
        for parent_workflow_id in sorted(grouped_workflows):
            items = sorted(
                grouped_workflows[parent_workflow_id],
                key=lambda item: (
                    item.request_id,
                    item.public_graph_name,
                    item.slice_index,
                    item.workflow_id,
                ),
            )
            if not items:
                continue
            template_hash = items[0].template_hash
            groups_payload.append(
                {
                    "template_hash": template_hash,
                    "requests": sorted({item.request_id for item in items}),
                    "public_graphs": sorted({item.public_graph_name for item in items}),
                    "slice_lengths": [item.slice_length for item in items],
                }
            )
        payload = {
            "workers": selected_workers,
            "groups": groups_payload,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _merge_group_context(workflows: list[Any]) -> str:
        first = workflows[0]
        return (
            f"request='{first.request_id}' "
            f"public_graph='{first.public_graph_name}' "
            f"template_hash='{first.template_hash}'"
        )

    @staticmethod
    def _iter_input_bound_list_specs(
        graph: Graph,
    ) -> Generator[tuple[str, dict[str, Any]], None, None]:
        for op in graph.as_dict().values():
            if isinstance(op, DataRetrievalOp):
                params = op.data_spec.get("params")
                if isinstance(params, list):
                    for param in params:
                        if not isinstance(param, dict):
                            continue
                        label = param.get("label")
                        data = param.get("data")
                        if not isinstance(label, str) or not isinstance(data, dict):
                            continue
                        data_type = data.get("type")
                        items = data.get("items")
                        if data_type != "list" or not isinstance(items, list):
                            continue
                        yield label, data
            if isinstance(op, LLMChatOp):
                columns = op.rowwise_columns
                if not isinstance(columns, list):
                    continue
                for column in columns:
                    if not isinstance(column, dict):
                        continue
                    label = column.get("label")
                    data = column.get("data")
                    if not isinstance(label, str) or not isinstance(data, dict):
                        continue
                    data_type = data.get("type")
                    items = data.get("items")
                    if data_type != "list" or not isinstance(items, list):
                        continue
                    yield label, data

    @classmethod
    def _validate_merged_varying_inputs(
        cls,
        *,
        workflows: list[Any],
        merged_inputs: dict[str, list[str]],
        varying_input_keys: set[str],
        group_context: str,
    ) -> None:
        if not varying_input_keys:
            return

        input_keys = set(workflows[0].dsl_graph.inputs.keys())
        missing_keys = sorted(varying_input_keys - input_keys)
        if missing_keys:
            raise ValueError(
                "Varying input keys missing from compiled graph inputs for "
                f"{group_context}: missing={missing_keys}"
            )

        first_total_length = workflows[0].total_length
        for item in workflows:
            if item.total_length != first_total_length:
                raise ValueError(
                    "Inconsistent total_length across slices in merge group for "
                    f"{group_context}: expected={first_total_length} "
                    f"workflow_id='{item.workflow_id}' got={item.total_length}"
                )

        for key in sorted(varying_input_keys):
            expected_merged_len = 0
            for item in workflows:
                values = item.dsl_graph.inputs.get(key)
                if values is None:
                    raise ValueError(
                        f"Missing varying input key '{key}' for workflow "
                        f"'{item.workflow_id}' in {group_context}"
                    )
                value_len = len(values)
                if value_len != item.slice_length:
                    raise ValueError(
                        "Workflow input length mismatch for varying key "
                        f"'{key}' in {group_context}: workflow_id='{item.workflow_id}' "
                        f"slice_length={item.slice_length} input_len={value_len}"
                    )
                expected_merged_len += value_len
            if expected_merged_len > first_total_length:
                raise ValueError(
                    "Merged varying input length exceeds total_length for "
                    f"{group_context}: key='{key}' merged_len={expected_merged_len} "
                    f"total_length={first_total_length}"
                )
            if key not in merged_inputs:
                raise ValueError(
                    f"Merged inputs missing varying key '{key}' for {group_context}"
                )
            merged_len = len(merged_inputs[key])
            if merged_len != expected_merged_len:
                raise ValueError(
                    "Merged varying input length mismatch for "
                    f"{group_context}: key='{key}' merged_len={merged_len} "
                    f"expected={expected_merged_len}"
                )

    @classmethod
    def _rewrite_varying_input_literals(
        cls,
        *,
        graph: Graph,
        merged_inputs: dict[str, list[str]],
        first_slice_inputs: dict[str, list[str]],
        varying_input_keys: set[str],
        group_context: str,
    ) -> dict[str, int]:
        rewrite_hits: dict[str, int] = {key: 0 for key in varying_input_keys}
        inferred_aliases: dict[str, str] = {}

        for label, data in cls._iter_input_bound_list_specs(graph):
            if label in varying_input_keys:
                continue
            items = data["items"]
            if not isinstance(items, list):
                continue
            matched_keys = [
                key for key, values in first_slice_inputs.items() if items == values
            ]
            if len(matched_keys) > 1:
                raise ValueError(
                    "Ambiguous varying input alias while coalescing "
                    f"{group_context}: label='{label}' "
                    f"matches keys={sorted(matched_keys)}"
                )
            if len(matched_keys) == 1:
                inferred_aliases[label] = matched_keys[0]

        for label, data in cls._iter_input_bound_list_specs(graph):
            rewrite_target_key: str | None = None
            if label in varying_input_keys:
                rewrite_target_key = label
            elif label in inferred_aliases:
                rewrite_target_key = inferred_aliases[label]
            if rewrite_target_key is None:
                continue
            replacement = merged_inputs[rewrite_target_key]
            data["items"] = list(replacement)
            rewrite_hits[rewrite_target_key] += 1

        for key, hit_count in rewrite_hits.items():
            if hit_count <= 0:
                # Some workflows (for example ETL image inputs) consume varying keys
                # via InputOp directly and do not expose any rewriteable list literals.
                # In that case, merged_inputs already carries the coalesced values.
                if key in graph.input_ops and key in merged_inputs:
                    continue
                raise ValueError(
                    "No rewrite target found for varying key "
                    f"'{key}' in {group_context}. "
                    "Expected at least one input-bound list field keyed by this label."
                )

        for label, data in cls._iter_input_bound_list_specs(graph):
            validation_target_key: str | None = None
            if label in varying_input_keys:
                validation_target_key = label
            elif label in inferred_aliases:
                validation_target_key = inferred_aliases[label]
            if validation_target_key is None:
                continue
            items = data["items"]
            if not isinstance(items, list):
                raise ValueError(
                    "Rewritten field is no longer a list for "
                    f"key='{label}' in {group_context}"
                )
            expected_len = len(merged_inputs[validation_target_key])
            actual_len = len(items)
            if actual_len != expected_len:
                raise ValueError(
                    f"Rewritten field length mismatch for key='{validation_target_key}'"
                    f" in {group_context}: expected={expected_len} got={actual_len}"
                )
        return rewrite_hits

    @classmethod
    def _merge_group_compiled_graph(cls, workflows: list[Any]) -> CompiledGraph:
        if not workflows:
            raise ValueError("Cannot merge an empty workflow group")
        ordered = sorted(
            workflows,
            key=lambda item: (
                item.request_id,
                item.public_graph_name,
                item.slice_index,
                item.workflow_id,
            ),
        )
        first = ordered[0]
        graph = copy.deepcopy(first.dsl_graph.graph)
        input_keys = tuple(first.dsl_graph.inputs.keys())
        varying_input_keys = set(first.varying_input_keys)
        group_context = cls._merge_group_context(ordered)

        if len(ordered) == 1:
            single_slice_inputs = {
                input_key: list(first.dsl_graph.inputs[input_key])
                for input_key in input_keys
            }
            compiled_graph = CompiledGraph(graph, single_slice_inputs)
            compiled_graph._coalesce_rewrite_hits = {}
            compiled_graph._coalesce_rewrite_skipped = True
            return compiled_graph

        for item in ordered:
            if set(item.varying_input_keys) != varying_input_keys:
                raise ValueError(
                    "Inconsistent varying_input_keys in merge group for "
                    f"{group_context}: workflow_id='{item.workflow_id}' "
                    f"expected={sorted(varying_input_keys)} "
                    f"got={sorted(item.varying_input_keys)}"
                )

        merged_inputs: dict[str, list[str]] = {}
        for input_key in input_keys:
            if input_key in varying_input_keys:
                merged_values: list[str] = []
                for item in ordered:
                    values = item.dsl_graph.inputs.get(input_key)
                    if values is None:
                        raise ValueError(
                            f"Missing input key '{input_key}' in workflow slice"
                        )
                    merged_values.extend(values)
                merged_inputs[input_key] = merged_values
            else:
                values = first.dsl_graph.inputs.get(input_key)
                if values is None:
                    raise ValueError(
                        f"Missing non-varying input key '{input_key}' in workflow slice"
                    )
                merged_inputs[input_key] = list(values)
        cls._validate_merged_varying_inputs(
            workflows=ordered,
            merged_inputs=merged_inputs,
            varying_input_keys=varying_input_keys,
            group_context=group_context,
        )
        first_slice_inputs = {
            key: list(first.dsl_graph.inputs[key]) for key in sorted(varying_input_keys)
        }
        rewrite_hits = cls._rewrite_varying_input_literals(
            graph=graph,
            merged_inputs=merged_inputs,
            first_slice_inputs=first_slice_inputs,
            varying_input_keys=varying_input_keys,
            group_context=group_context,
        )
        compiled_graph = CompiledGraph(graph, merged_inputs)
        compiled_graph._coalesce_rewrite_hits = rewrite_hits
        compiled_graph._coalesce_rewrite_skipped = False
        return compiled_graph

    async def _run_batch(self, workers: list[str], batch: BatchSelection) -> None:
        batch_id = f"batch-{unique_id()}"
        request_ids = tuple(
            sorted({workflow.request_id for workflow in batch.workflows})
        )
        execution_request_id = f"exec-{unique_id()}"
        self._execution_contexts[execution_request_id] = ExecutionBatchContext(
            execution_request_id=execution_request_id,
            batch_id=batch_id,
            request_ids=request_ids,
            workflow_ids=tuple(workflow.workflow_id for workflow in batch.workflows),
        )
        for request_id in request_ids:
            self._request_execution_ids.setdefault(request_id, set()).add(
                execution_request_id
            )
            self._request_execution_history_ids.setdefault(request_id, set()).add(
                execution_request_id
            )
        cancelled_requests = await self._collect_cancelled_requests(set(request_ids))
        active_workflows = [
            workflow
            for workflow in batch.workflows
            if workflow.request_id not in cancelled_requests
        ]
        active_request_ids = {workflow.request_id for workflow in active_workflows}
        request_raw_nodes: dict[str, int] = {}
        for workflow in active_workflows:
            request_raw_nodes[workflow.request_id] = (
                request_raw_nodes.get(workflow.request_id, 0)
                + workflow.runtime_graph.node_count
            )
        try:
            self.logger.info(
                "Starting batch %s on workers %s (workflows=%d requests=%d"
                " execution=%s)",
                batch_id,
                workers,
                len(batch.workflows),
                len(request_ids),
                execution_request_id,
            )
            for request_id, raw_nodes in request_raw_nodes.items():
                state = self._requests.get(request_id)
                if state is None:
                    continue
                if state.pending_runtime_nodes_raw < raw_nodes:
                    raise RuntimeError(
                        "Pending runtime nodes underflow for request "
                        f"{request_id}: pending={state.pending_runtime_nodes_raw} "
                        f"batch={raw_nodes}"
                    )
                state.pending_runtime_nodes_raw -= raw_nodes
                state.processing_runtime_nodes_raw += raw_nodes
                state.batch_node_counts[batch_id] = {
                    "raw": raw_nodes,
                    "optimized": 0,
                }

            worker_profiles: dict[str, dict[str, Any]] = {}
            if active_workflows:
                worker_profiles = {
                    worker: self._normalize_worker_profile(
                        await self.runtime_manager.get_worker_profile(worker)
                    )
                    for worker in workers
                }

            batch_outputs: dict[str, Any] = {}
            batch_histories: dict[str, Any] = {}
            if active_workflows:
                active_batch = BatchSelection(
                    workflows=active_workflows,
                    runtime_graphs={
                        item.workflow_id: item.runtime_graph
                        for item in active_workflows
                    },
                    data_profile_graphs={
                        item.workflow_id: item.data_profile_graph
                        for item in active_workflows
                    },
                    config=batch.config,
                )
                batch_outputs, batch_histories = await self._process_batch(
                    active_batch,
                    batch_id,
                    workers,
                    worker_profiles,
                    execution_request_id=execution_request_id,
                    member_request_ids=active_request_ids,
                )
                self.runtime_manager.mark_batch_completed(
                    execution_request_id, batch_id
                )
                cancelled_requests = await self._collect_cancelled_requests(
                    set(request_ids)
                )
            else:
                await self.runtime_manager.cancel_request(execution_request_id)
            await self._handle_batch_results(
                batch,
                batch_outputs,
                batch_histories,
                None,
                batch_id=batch_id,
                cancelled_requests=cancelled_requests,
                execution_request_id=execution_request_id,
            )
            self.logger.info("Completed batch %s", batch_id)
        except RequestCancelledError as exc:
            self.logger.info(
                "Batch %s cancelled (execution request %s)",
                batch_id,
                execution_request_id,
            )
            if active_workflows:
                self.runtime_manager.mark_batch_failed(execution_request_id, batch_id)
            cancelled_requests = await self._collect_cancelled_requests(
                set(request_ids)
            )
            await self._handle_batch_results(
                batch,
                {},
                {},
                exc,
                batch_id=batch_id,
                cancelled_requests=cancelled_requests,
                execution_request_id=execution_request_id,
            )
        except Exception as exc:
            self.logger.error("Batch %s failed", batch_id, exc_info=True)
            if active_workflows:
                self.runtime_manager.mark_batch_failed(execution_request_id, batch_id)
            cancelled_requests = await self._collect_cancelled_requests(
                set(request_ids)
            )
            await self._handle_batch_results(
                batch,
                {},
                {},
                exc,
                batch_id=batch_id,
                cancelled_requests=cancelled_requests,
                execution_request_id=execution_request_id,
            )
        finally:
            try:
                for request_id in request_raw_nodes:
                    state = self._requests.get(request_id)
                    if state is None:
                        continue
                    counts = state.batch_node_counts.pop(batch_id, None)
                    if counts is None:
                        continue
                    raw_nodes = counts.get("raw", 0)
                    optimized_nodes = counts.get("optimized", 0)
                    if raw_nodes:
                        if state.processing_runtime_nodes_raw < raw_nodes:
                            raise RuntimeError(
                                "Processing runtime raw underflow for request "
                                f"{request_id}: processing="
                                f"{state.processing_runtime_nodes_raw} "
                                f"batch={raw_nodes}"
                            )
                        state.processing_runtime_nodes_raw -= raw_nodes
                        state.processed_runtime_nodes_raw += raw_nodes
                    if optimized_nodes:
                        if state.processing_runtime_nodes_optimized < optimized_nodes:
                            raise RuntimeError(
                                "Processing runtime optimized underflow for request "
                                f"{request_id}: processing="
                                f"{state.processing_runtime_nodes_optimized} "
                                f"batch={optimized_nodes}"
                            )
                        state.processing_runtime_nodes_optimized -= optimized_nodes
                        state.processed_runtime_nodes_optimized += optimized_nodes
                async with self._worker_lock:
                    self._busy_workers.difference_update(workers)
                self.job_manager.finalize_workflows(
                    [workflow.workflow_id for workflow in batch.workflows]
                )
            finally:
                self._cleanup_execution_tracking(execution_request_id, request_ids)

    def _cleanup_execution_tracking(
        self,
        execution_request_id: str,
        request_ids: tuple[str, ...],
    ) -> None:
        self._execution_contexts.pop(execution_request_id, None)
        for request_id in request_ids:
            execution_ids = self._request_execution_ids.get(request_id)
            if execution_ids is None:
                continue
            execution_ids.discard(execution_request_id)
            if execution_ids:
                continue
            self._request_execution_ids.pop(request_id, None)

    def _record_optimizer_time(
        self,
        *,
        member_request_ids: set[str],
        optimized_nodes_by_request: dict[str, int],
        schedule_elapsed_seconds: float,
    ) -> None:
        if schedule_elapsed_seconds <= 0:
            return
        if not member_request_ids:
            return

        relevant_request_ids = sorted(member_request_ids)
        optimized_total = sum(
            max(0, optimized_nodes_by_request.get(request_id, 0))
            for request_id in relevant_request_ids
        )

        if optimized_total <= 0:
            equal_share = schedule_elapsed_seconds / len(relevant_request_ids)
            for request_id in relevant_request_ids:
                state = self._requests.get(request_id)
                if state is not None:
                    state.optimization_seconds += equal_share
            return

        for request_id in relevant_request_ids:
            state = self._requests.get(request_id)
            if state is None:
                continue
            optimized_nodes = max(0, optimized_nodes_by_request.get(request_id, 0))
            if optimized_nodes <= 0:
                continue
            state.optimization_seconds += (
                schedule_elapsed_seconds * optimized_nodes / optimized_total
            )

    async def _handle_batch_results(
        self,
        batch: BatchSelection,
        batch_outputs: dict[str, Any],
        batch_histories: dict[str, Any],
        error: Exception | None,
        *,
        batch_id: str | None = None,
        task_node_map: dict[str, str] | None = None,
        cancelled_requests: set[str] | None = None,
        execution_request_id: str | None = None,
    ) -> None:
        cancelled_requests = cancelled_requests or set()

        def mark_successful_input_completion(
            state: RequestState, workflow: Any
        ) -> None:
            workflow_id = workflow.workflow_id
            if workflow_id in state.successful_workflow_ids:
                return
            state.successful_workflow_ids.add(workflow_id)
            state.completed_input_items_success += max(0, int(workflow.slice_length))

        def append_error(
            state: RequestState,
            payload: dict[str, Any],
        ) -> None:
            if state.error_info is None:
                state.error_info = []
            state.error_info.append(payload)

        def merge_slice_payload(
            state: RequestState,
            workflow: Any,
            payload: dict[str, list[Any]],
            *,
            is_history: bool,
        ) -> bool:
            public_name = workflow.public_graph_name
            total_length = workflow.total_length
            if total_length <= 0:
                append_error(
                    state,
                    {
                        "graph": public_name,
                        "slice_index": workflow.slice_index,
                        "workflow_id": workflow.workflow_id,
                        "error": "invalid total slice length",
                    },
                )
                return False
            start = workflow.slice_start
            stop = start + workflow.slice_length
            if start < 0 or stop > total_length:
                append_error(
                    state,
                    {
                        "graph": public_name,
                        "slice_index": workflow.slice_index,
                        "workflow_id": workflow.workflow_id,
                        "error": "slice range out of bounds",
                    },
                )
                return False

            buffers = (
                state.chat_history_buffers if is_history else state.output_buffers
            ).setdefault(public_name, {})
            merged_without_errors = True
            for output_name, values in payload.items():
                if len(values) != workflow.slice_length:
                    append_error(
                        state,
                        {
                            "graph": public_name,
                            "slice_index": workflow.slice_index,
                            "workflow_id": workflow.workflow_id,
                            "output": output_name,
                            "error": (
                                "slice payload length mismatch: "
                                f"expected={workflow.slice_length} got={len(values)}"
                            ),
                        },
                    )
                    merged_without_errors = False
                    continue
                target = buffers.setdefault(output_name, [None] * total_length)
                if len(target) != total_length:
                    raise RuntimeError(
                        f"Inconsistent buffer size for {public_name}:{output_name}"
                    )
                overlap = [idx for idx in range(start, stop) if target[idx] is not None]
                if overlap:
                    append_error(
                        state,
                        {
                            "graph": public_name,
                            "slice_index": workflow.slice_index,
                            "workflow_id": workflow.workflow_id,
                            "output": output_name,
                            "error": (
                                f"overlapping slice assignment at indices {overlap}"
                            ),
                        },
                    )
                    merged_without_errors = False
                for idx, value in enumerate(values, start=start):
                    target[idx] = value
            return merged_without_errors

        def finalize_state(state: RequestState) -> None:
            finalized_outputs: dict[str, dict[str, list[str]]] = {}
            finalized_histories: dict[str, dict[str, list[Any]]] = {}
            for public_name, output_map in state.output_buffers.items():
                for output_name, values in output_map.items():
                    missing = [idx for idx, value in enumerate(values) if value is None]
                    if missing:
                        append_error(
                            state,
                            {
                                "graph": public_name,
                                "output": output_name,
                                "error": f"missing output indices {missing}",
                            },
                        )
                    finalized_outputs.setdefault(public_name, {})[output_name] = [
                        str(value) if value is not None else "" for value in values
                    ]
            for public_name, output_map in state.chat_history_buffers.items():
                for output_name, values in output_map.items():
                    missing = [idx for idx, value in enumerate(values) if value is None]
                    if missing:
                        append_error(
                            state,
                            {
                                "graph": public_name,
                                "output": output_name,
                                "error": f"missing chat history indices {missing}",
                            },
                        )
                    finalized_histories.setdefault(public_name, {})[output_name] = [
                        value if value is not None else [] for value in values
                    ]
            state.outputs = finalized_outputs
            state.chat_histories = finalized_histories

        affected_requests: set[str] = set()
        for workflow in batch.workflows:
            request_id = workflow.request_id
            state = self._requests.get(request_id)
            if state is None:
                continue
            affected_requests.add(request_id)

            if request_id in cancelled_requests:
                append_error(
                    state,
                    {
                        "request_cancelled": request_id,
                        "execution_request_id": execution_request_id,
                        "graph": workflow.public_graph_name,
                        "slice_index": workflow.slice_index,
                        "workflow_id": workflow.workflow_id,
                    },
                )
                state.pending_workflows.discard(workflow.workflow_id)
                continue

            if error:
                append_error(
                    state,
                    {
                        "batch_error": str(error),
                        "execution_request_id": execution_request_id,
                        "graph": workflow.public_graph_name,
                        "slice_index": workflow.slice_index,
                        "workflow_id": workflow.workflow_id,
                    },
                )

            if workflow.workflow_id in batch_outputs:
                merged_successfully = merge_slice_payload(
                    state,
                    workflow,
                    batch_outputs[workflow.workflow_id],
                    is_history=False,
                )
                if merged_successfully:
                    mark_successful_input_completion(state, workflow)
            elif error is None:
                append_error(
                    state,
                    {
                        "missing_output": workflow.workflow_id,
                        "execution_request_id": execution_request_id,
                        "graph": workflow.public_graph_name,
                        "slice_index": workflow.slice_index,
                    },
                )
                self.logger.warning(
                    "Missing output for workflow %s (graph=%s, slice=%d, request=%s)",
                    workflow.workflow_id,
                    workflow.public_graph_name,
                    workflow.slice_index,
                    request_id,
                )
            if workflow.workflow_id in batch_histories:
                merge_slice_payload(
                    state,
                    workflow,
                    batch_histories[workflow.workflow_id],
                    is_history=True,
                )
            state.pending_workflows.discard(workflow.workflow_id)

        if error and task_node_map:
            for request_id in affected_requests:
                state = self._requests.get(request_id)
                if state is None:
                    continue
                append_error(
                    state,
                    {
                        "batch_id": batch_id,
                        "execution_request_id": execution_request_id,
                        "task_node_map": task_node_map,
                    },
                )

        for request_id in affected_requests:
            state = self._requests.get(request_id)
            if state is None or state.pending_workflows or not state.ready:
                continue
            finalize_state(state)
            response = LumilakeResponse(
                outputs=state.outputs,
                error_info=state.error_info,
                chat_histories=state.chat_histories,
            )
            await state.handler.put_result(response)
            # Keep request state available for progress polling after completion.

    async def _process_batch(
        self,
        batch: BatchSelection,
        batch_id: str,
        selected_workers: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        *,
        execution_request_id: str,
        member_request_ids: set[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Process a single batch of graphs on a selected worker.

        Parameters
        ----------
        batch : BatchSelection
            Selected batch metadata including workflows and graphs
        batch_id : str
            Identifier for this batch
        selected_workers : list[str]
            Workers selected to execute this batch.
        worker_profiles : dict[str, dict[str, Any]]
            Worker hardware profiles for optimizer scheduling.

        Returns
        -------
        tuple[dict[str, Any], dict[str, Any]]
            (outputs, chat_histories) for this batch, keyed by workflow id
        """
        if not batch.workflows:
            raise ValueError("No workflows in batch")

        if await self._should_cancel_execution(
            execution_request_id,
            member_request_ids,
        ):
            self.logger.info(
                "Skipping batch %s for execution request %s (all members cancelled)",
                batch_id,
                execution_request_id,
            )
            raise RequestCancelledError(execution_request_id)

        grouped_workflows = self._group_workflows_by_parent_workflow(batch.workflows)
        for group_key, workflows in grouped_workflows.items():
            if len(workflows) > 1:
                self.logger.info(
                    "Coalescing %d slices under parent workflow %s in batch %s "
                    "(members=%s)",
                    len(workflows),
                    group_key,
                    batch_id,
                    [item.workflow_id for item in workflows],
                )

        data_profile_sources: dict[str, list[DataProfileSource]] = {}
        for group_key, workflows in grouped_workflows.items():
            ordered_workflows = sorted(
                workflows,
                key=lambda item: (
                    item.request_id,
                    item.public_graph_name,
                    item.slice_index,
                    item.workflow_id,
                ),
            )
            candidate_sources: list[DataProfileSource] = []
            seen: set[tuple[str, str]] = set()
            seen_orgs: set[str] = set()
            ordered_org_ids: list[str] = []
            for item in ordered_workflows:
                org_id = item.config.org_id
                if org_id in seen_orgs:
                    continue
                seen_orgs.add(org_id)
                ordered_org_ids.append(org_id)
            for org_id in ordered_org_ids:
                source = DataProfileSource(task_key=group_key, org_id=org_id)
                dedupe_key = (source.task_key, source.org_id)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                candidate_sources.append(source)
            for item in ordered_workflows:
                source = DataProfileSource(
                    task_key=self._request_data_profile_task_key(item),
                    org_id=item.config.org_id,
                )
                dedupe_key = (source.task_key, source.org_id)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                candidate_sources.append(source)
            data_profile_sources[group_key] = candidate_sources

        runtime_graphs_by_name: dict[str, Any] = {}
        data_profile_graphs_by_name: dict[str, Any] = {}
        for group_key, workflows in grouped_workflows.items():
            varying_input_keys = sorted(
                {key for item in workflows for key in item.varying_input_keys}
            )
            if len(workflows) == 1:
                self.logger.info(
                    "Single-slice workflow group %s; skipping varying-input literal"
                    " rewrite",
                    group_key,
                )
            try:
                merged_compiled = self._merge_group_compiled_graph(workflows)
            except Exception:
                self.logger.exception(
                    "Failed to coalesce workflow group %s (slices=%d,"
                    " varying_input_keys=%s)",
                    group_key,
                    len(workflows),
                    varying_input_keys,
                )
                raise

            rewrite_hits: dict[str, int] = {}
            candidate_hits = merged_compiled._coalesce_rewrite_hits
            if isinstance(candidate_hits, dict):
                rewrite_hits = {
                    str(key): int(value) for key, value in candidate_hits.items()
                }

            merged_input_lengths: dict[str, int] = {}
            if isinstance(merged_compiled, CompiledGraph):
                merged_input_lengths = {
                    input_key: len(values)
                    for input_key, values in merged_compiled.inputs.items()
                }
            self.logger.info(
                "Coalesced workflow group %s (slices=%d, varying_input_keys=%s,"
                " merged_input_lengths=%s, rewrite_hits=%s)",
                group_key,
                len(workflows),
                varying_input_keys,
                merged_input_lengths,
                rewrite_hits,
            )
            runtime_graphs_by_name[group_key] = self._runtime_builder.build(
                merged_compiled,
                node_prefix=group_key,
            )
            data_profile_graphs_by_name[group_key] = self._runtime_builder.build(
                merged_compiled,
                task_type_override="data_profile",
                node_prefix=group_key,
            )

        merged_graph, output_mapping = self.optimizer.optimize_graphs(
            runtime_graphs_by_name
        )
        merged_data_profile_graph, _ = self.optimizer.optimize_graphs(
            data_profile_graphs_by_name
        )
        total_nodes = merged_graph.node_count
        output_nodes = len(merged_graph.output_node_map)
        self.runtime_manager.mark_batch_pending(
            execution_request_id,
            batch_id,
            total_nodes,
            output_nodes,
        )
        self.runtime_manager.mark_batch_running(execution_request_id, batch_id)
        raw_nodes_by_request: dict[str, int] = {}
        for workflow in batch.workflows:
            raw_nodes_by_request[workflow.request_id] = (
                raw_nodes_by_request.get(workflow.request_id, 0)
                + workflow.runtime_graph.node_count
            )
        total_raw_nodes = sum(raw_nodes_by_request.values())
        optimized_nodes_by_request: dict[str, int] = {}
        assigned = 0
        request_order = sorted(raw_nodes_by_request.keys())
        for request_id in request_order:
            raw_nodes = raw_nodes_by_request[request_id]
            optimized = (
                0
                if total_raw_nodes <= 0
                else (total_nodes * raw_nodes) // total_raw_nodes
            )
            optimized_nodes_by_request[request_id] = optimized
            assigned += optimized
        remainder = total_nodes - assigned
        for request_id in request_order:
            if remainder <= 0:
                break
            optimized_nodes_by_request[request_id] += 1
            remainder -= 1
        for request_id, optimized_nodes in optimized_nodes_by_request.items():
            state = self._requests.get(request_id)
            if state is None:
                continue
            counts = state.batch_node_counts.get(batch_id)
            if counts is None:
                raise RuntimeError(
                    f"Missing batch node counts for request {request_id} ({batch_id})"
                )
            if counts.get("optimized", 0):
                raise RuntimeError(
                    f"Batch optimized nodes already recorded for request {request_id}"
                    f" ({batch_id})"
                )
            state.processing_runtime_nodes_optimized += optimized_nodes
            counts["optimized"] = optimized_nodes

        cache_owner_request_id = batch.workflows[0].request_id
        for workflow in batch.workflows:
            if workflow.request_id in member_request_ids:
                cache_owner_request_id = workflow.request_id
                break
        cache_state = self._requests.get(cache_owner_request_id)

        batch_request_info = RequestInfo(
            request_id=execution_request_id,
            runtime_graphs=runtime_graphs_by_name,
            data_profile_graphs=data_profile_graphs_by_name,
            data_profile_sources=data_profile_sources,
        )
        batch_request_info.batch_id = batch_id
        batch_request_info.runtime_graph = merged_graph
        batch_request_info.data_profile_graph = merged_data_profile_graph
        batch_request_info.output_node_map = output_mapping

        schedule: Schedule | None = None
        cache_key = self._build_plan_cache_key(selected_workers, grouped_workflows)
        normalized_data_profile_results: dict[str, list[dict[str, Any]]] | None = None
        cache_hit = False

        if cache_state is not None:
            cached = cache_state.plan_cache.get(cache_key)
        else:
            cached = None
        if cached is not None:
            try:
                schedule = copy.deepcopy(cached.schedule)
                normalized_data_profile_results = copy.deepcopy(
                    cached.data_profile_results
                )
                self._validate_schedule(
                    schedule,
                    selected_workers,
                    set(batch_request_info.runtime_graph.nodes),
                )
                cache_hit = True
                self.logger.info(
                    "Plan cache hit for execution request %s (batch %s, key=%s"
                    " owner=%s)",
                    execution_request_id,
                    batch_id,
                    cache_key[:12],
                    cache_owner_request_id,
                )
            except Exception:
                self.logger.info(
                    "Plan cache invalid for execution request %s (batch %s, key=%s);"
                    " recomputing",
                    execution_request_id,
                    batch_id,
                    cache_key[:12],
                )
                cached = None

        if cached is None:
            if await self._should_cancel_execution(
                execution_request_id,
                member_request_ids,
            ):
                self.logger.info(
                    "Skipping data profile for batch %s (execution request %s"
                    " cancelled)",
                    batch_id,
                    execution_request_id,
                )
                raise RequestCancelledError(execution_request_id)
            self.logger.debug("Starting data profile for batch %s", batch_id)

            async def _is_data_profile_cancelled() -> bool:
                return await self._should_cancel_execution(
                    execution_request_id,
                    member_request_ids,
                )

            normalized_data_profile_results = await collect_data_profile(
                request_id=execution_request_id,
                data_profile_graphs=batch_request_info.data_profile_graphs,
                data_profile_sources=batch_request_info.data_profile_sources,
                cancellation_callback=_is_data_profile_cancelled,
                logger=self.logger,
            )
            profile_uri = self._save_runtime_artifact(
                batch_request_info,
                "flowmesh_data_profile_result.yaml",
                yaml.dump(
                    normalized_data_profile_results,
                    default_flow_style=False,
                    sort_keys=False,
                ).encode("utf-8"),
                "application/x-yaml",
            )
            self.logger.debug(
                "Archived data profile for batch %s to %s",
                batch_id,
                profile_uri,
            )
            self.logger.debug(
                "Data profile completed for batch %s: %d tables",
                batch_id,
                len(normalized_data_profile_results),
            )

            schedule_start_time = time.perf_counter()
            async with self._optimizer_lock:
                self.logger.debug(
                    "Generating schedule for batch %s on workers %s",
                    batch_id,
                    selected_workers,
                )
                schedule = await self._generate_schedule_in_subprocess(
                    request_id=execution_request_id,
                    batch_id=batch_id,
                    runtime_graph=batch_request_info.runtime_graph,
                    selected_workers=selected_workers,
                    worker_profiles=worker_profiles,
                    data_profile_results=normalized_data_profile_results,
                    member_request_ids=member_request_ids,
                )
                self.logger.debug("Schedule for batch %s: %s", batch_id, schedule)
            schedule_elapsed = time.perf_counter() - schedule_start_time
            self._record_optimizer_time(
                member_request_ids=member_request_ids,
                optimized_nodes_by_request=optimized_nodes_by_request,
                schedule_elapsed_seconds=schedule_elapsed,
            )
            self._validate_schedule(
                schedule,
                selected_workers,
                set(batch_request_info.runtime_graph.nodes),
            )
            if cache_state is not None:
                cache_state.plan_cache[cache_key] = PlanCacheEntry(
                    data_profile_results=copy.deepcopy(normalized_data_profile_results),
                    schedule=copy.deepcopy(schedule),
                )
            self.logger.info(
                "Plan cache miss for execution request %s (batch %s, key=%s owner=%s)",
                execution_request_id,
                batch_id,
                cache_key[:12],
                cache_owner_request_id,
            )
        if schedule is None or normalized_data_profile_results is None:
            raise RuntimeError(
                "Schedule generation failed to produce schedule/data profile results"
            )

        # Step 3: Submit to runtime backend with schedule hint and worker assignment
        if await self._should_cancel_execution(
            execution_request_id,
            member_request_ids,
        ):
            self.logger.info(
                "Skipping runtime submission for batch %s (execution request %s"
                " cancelled)",
                batch_id,
                execution_request_id,
            )
            raise RequestCancelledError(execution_request_id)
        self.logger.debug(
            "Submitting batch %s to runtime (workers: %s)",
            batch_id,
            selected_workers,
        )
        connector_result = await self.runtime_manager.process_request(
            batch_request_info,
            schedule,
            selected_workers,
            normalized_data_profile_results,
        )
        flat_outputs = connector_result["flat_outputs"]
        chat_histories = connector_result["chat_histories"]
        task_node_map = connector_result.get("task_node_map")
        if isinstance(task_node_map, dict):
            for request_id in member_request_ids:
                state = self._requests.get(request_id)
                if state is not None:
                    state.task_node_map[batch_id] = dict(task_node_map)
        self.logger.debug(
            "Batch %s completed: %d outputs (cache_hit=%s)",
            batch_id,
            len(flat_outputs),
            cache_hit,
        )

        # Step 5: Remap outputs to public workflow names, then demux to slices.
        grouped_outputs: dict[str, dict[str, list[Any]]] = {}
        grouped_histories: dict[str, dict[str, list[Any]]] = {}
        direct_outputs: dict[str, dict[str, list[Any]]] = {}
        direct_histories: dict[str, dict[str, list[Any]]] = {}
        group_workflow_prefixes: dict[str, list[tuple[str, Any]]] = {}
        for group_key, workflows in grouped_workflows.items():
            ordered = sorted(
                workflows,
                key=lambda item: len(item.workflow_id),
                reverse=True,
            )
            group_workflow_prefixes[group_key] = [
                (f"{item.workflow_id}__", item) for item in ordered
            ]

        def match_workflow_for_group_node(group_key: str, node_id: str) -> Any | None:
            for prefix, workflow in group_workflow_prefixes.get(group_key, ()):
                if node_id.startswith(prefix):
                    return workflow
            return None

        for node_id, outputs in flat_outputs.items():
            mapping = output_mapping.get(node_id)
            if mapping is None:
                raise ValueError(f"Missing output mapping for runtime node {node_id}")
            group_key, output_name = mapping
            if not isinstance(outputs, list):
                raise ValueError(
                    f"Expected list output for node {node_id}, got"
                    f" {type(outputs).__name__}"
                )
            matched_workflow = match_workflow_for_group_node(group_key, node_id)
            if matched_workflow is not None:
                if len(outputs) != matched_workflow.slice_length:
                    raise ValueError(
                        "Output length mismatch for workflow slice "
                        f"{matched_workflow.workflow_id}:{output_name}. "
                        f"expected={matched_workflow.slice_length} got={len(outputs)}"
                    )
                direct_outputs.setdefault(matched_workflow.workflow_id, {})[
                    output_name
                ] = outputs
                continue
            grouped_outputs.setdefault(group_key, {}).setdefault(
                output_name, []
            ).extend(outputs)
        for node_id, history in chat_histories.items():
            mapping = output_mapping.get(node_id)
            if mapping is None:
                raise ValueError(f"Missing output mapping for runtime node {node_id}")
            group_key, output_name = mapping
            if not isinstance(history, list):
                raise ValueError(
                    f"Expected list history for node {node_id}, got"
                    f" {type(history).__name__}"
                )
            matched_workflow = match_workflow_for_group_node(group_key, node_id)
            if matched_workflow is not None:
                if len(history) != matched_workflow.slice_length:
                    raise ValueError(
                        "History length mismatch for workflow slice "
                        f"{matched_workflow.workflow_id}:{output_name}. "
                        f"expected={matched_workflow.slice_length} got={len(history)}"
                    )
                direct_histories.setdefault(matched_workflow.workflow_id, {})[
                    output_name
                ] = history
                continue
            grouped_histories.setdefault(group_key, {}).setdefault(
                output_name, []
            ).extend(history)

        remapped_outputs: dict[str, dict[str, Any]] = {
            workflow_id: dict(outputs)
            for workflow_id, outputs in direct_outputs.items()
        }
        remapped_histories: dict[str, dict[str, Any]] = {
            workflow_id: dict(histories)
            for workflow_id, histories in direct_histories.items()
        }
        for group_key, workflows in grouped_workflows.items():
            ordered = sorted(workflows, key=lambda item: item.slice_index)
            total_slice_len = sum(item.slice_length for item in ordered)
            group_outputs = grouped_outputs.get(group_key, {})
            for output_name, values in group_outputs.items():
                if len(values) != total_slice_len:
                    raise ValueError(
                        "Output length mismatch for merged workflow"
                        f" {group_key}:{output_name}. expected={total_slice_len}"
                        f" got={len(values)}"
                    )
                offset = 0
                for workflow in ordered:
                    next_offset = offset + workflow.slice_length
                    workflow_outputs = remapped_outputs.setdefault(
                        workflow.workflow_id, {}
                    )
                    if output_name in workflow_outputs:
                        raise ValueError(
                            "Duplicate output assignment for workflow "
                            f"{workflow.workflow_id}:{output_name}"
                        )
                    workflow_outputs[output_name] = values[offset:next_offset]
                    offset = next_offset
            group_histories = grouped_histories.get(group_key, {})
            for output_name, values in group_histories.items():
                if len(values) != total_slice_len:
                    raise ValueError(
                        "History length mismatch for merged workflow"
                        f" {group_key}:{output_name}. expected={total_slice_len}"
                        f" got={len(values)}"
                    )
                offset = 0
                for workflow in ordered:
                    next_offset = offset + workflow.slice_length
                    workflow_histories = remapped_histories.setdefault(
                        workflow.workflow_id, {}
                    )
                    if output_name in workflow_histories:
                        raise ValueError(
                            "Duplicate history assignment for workflow "
                            f"{workflow.workflow_id}:{output_name}"
                        )
                    workflow_histories[output_name] = values[offset:next_offset]
                    offset = next_offset

        missing_outputs = set(batch.runtime_graphs.keys()) - set(
            remapped_outputs.keys()
        )
        if missing_outputs:
            self.logger.warning(
                "Batch %s missing outputs for %d workflow(s): %s",
                batch_id,
                len(missing_outputs),
                sorted(missing_outputs),
            )

        workflow_request_map = {
            workflow.workflow_id: workflow.request_id for workflow in batch.workflows
        }
        artifact_cache: dict[tuple[str, str], str] = {}
        for workflow_id, output_payload in list(remapped_outputs.items()):
            request_id = workflow_request_map[workflow_id]
            remapped_outputs[workflow_id] = self._relocate_artifacts_for_request(
                output_payload,
                source_request_id=execution_request_id,
                target_request_id=request_id,
                cache=artifact_cache,
            )
        for workflow_id, history_payload in list(remapped_histories.items()):
            request_id = workflow_request_map[workflow_id]
            remapped_histories[workflow_id] = self._relocate_artifacts_for_request(
                history_payload,
                source_request_id=execution_request_id,
                target_request_id=request_id,
                cache=artifact_cache,
            )

        return remapped_outputs, remapped_histories

    @staticmethod
    def _optimizer_progress_interval_s() -> float:
        raw = envs.LUMILAKE_POLL_INTERVAL_SECONDS
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 5.0
        if not math.isfinite(value) or value <= 0:
            return 5.0
        return value

    @staticmethod
    def _optimizer_subprocess_timeout_s() -> float:
        raw = envs.LUMILAKE_OPTIMIZER_SUBPROCESS_TIMEOUT_SECONDS
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 60.0
        if not math.isfinite(value) or value <= 0:
            return 60.0
        return value

    async def _generate_schedule_in_subprocess(
        self,
        *,
        request_id: str,
        batch_id: str,
        runtime_graph: Any,
        selected_workers: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        data_profile_results: dict[str, list[dict[str, Any]]],
        member_request_ids: set[str] | None = None,
    ) -> Schedule:
        ctx = mp.get_context("spawn")
        result_queue: Any = ctx.Queue(maxsize=1)
        process = ctx.Process(
            target=_optimizer_subprocess_entry,
            args=(
                self._optimizer_type,
                runtime_graph,
                selected_workers,
                worker_profiles,
                data_profile_results,
                result_queue,
            ),
        )
        interval_s = self._optimizer_progress_interval_s()
        timeout_s = self._optimizer_subprocess_timeout_s()
        next_log_at = time.perf_counter() + interval_s
        start_time = time.perf_counter()
        started = False
        try:
            process.start()
            started = True
            self.logger.info(
                "Optimizer subprocess started for request %s batch %s (pid=%s,"
                " type=%s)",
                request_id,
                batch_id,
                process.pid,
                self._optimizer_type,
            )
            while process.is_alive():
                should_cancel = await self.runtime_manager.is_request_cancelled(
                    request_id
                )
                if (
                    not should_cancel
                    and member_request_ids
                    and await self._should_cancel_execution(
                        request_id, member_request_ids
                    )
                ):
                    should_cancel = True
                if should_cancel:
                    self.logger.info(
                        "Terminating optimizer subprocess due to cancellation:"
                        " request=%s batch=%s pid=%s",
                        request_id,
                        batch_id,
                        process.pid,
                    )
                    process.terminate()
                    process.join(timeout=2)
                    raise RequestCancelledError(request_id)
                now = time.perf_counter()
                elapsed = now - start_time
                if elapsed >= timeout_s:
                    self.logger.error(
                        "Optimizer subprocess timed out after %.1fs: request=%s"
                        " batch=%s pid=%s",
                        elapsed,
                        request_id,
                        batch_id,
                        process.pid,
                    )
                    process.terminate()
                    process.join(timeout=2)
                    if process.is_alive():
                        self.logger.error(
                            "Force-killing optimizer subprocess after timeout:"
                            " request=%s batch=%s pid=%s",
                            request_id,
                            batch_id,
                            process.pid,
                        )
                        process.kill()
                        process.join(timeout=2)
                    raise RuntimeError(
                        "Optimizer subprocess timed out after "
                        f"{timeout_s:.1f}s for request={request_id} batch={batch_id}"
                    )
                if now >= next_log_at:
                    self.logger.info(
                        "Optimizer subprocess still running: request=%s batch=%s pid=%s"
                        " elapsed=%.1fs",
                        request_id,
                        batch_id,
                        process.pid,
                        elapsed,
                    )
                    next_log_at = now + interval_s
                await asyncio.sleep(0.2)

            process.join(timeout=1)
            try:
                payload = result_queue.get_nowait()
            except queue.Empty as exc:
                raise RuntimeError(
                    "Optimizer subprocess exited without result payload: "
                    f"request={request_id} batch={batch_id} exitcode={process.exitcode}"
                ) from exc
            if not isinstance(payload, dict):
                raise RuntimeError(
                    "Optimizer subprocess returned invalid payload type: "
                    f"{type(payload).__name__}"
                )
            if not bool(payload.get("ok")):
                error = str(payload.get("error", "unknown error"))
                tb = str(payload.get("traceback", "")).strip()
                details = f"{error}\n{tb}" if tb else error
                raise RuntimeError(
                    "Optimizer subprocess failed for "
                    f"request={request_id} batch={batch_id}: {details}"
                )
            schedule = payload.get("schedule")
            if not isinstance(schedule, Schedule):
                raise RuntimeError(
                    "Optimizer subprocess returned invalid schedule payload: "
                    f"{type(schedule).__name__}"
                )
            self.logger.info(
                "Optimizer subprocess finished for request %s batch %s in %.2fs",
                request_id,
                batch_id,
                time.perf_counter() - start_time,
            )
            return schedule
        except asyncio.CancelledError:
            if started and process.is_alive():
                self.logger.info(
                    "Terminating optimizer subprocess due to task cancellation:"
                    " request=%s batch=%s pid=%s",
                    request_id,
                    batch_id,
                    process.pid,
                )
                process.terminate()
                process.join(timeout=2)
            raise
        finally:
            if started and process.is_alive():
                process.terminate()
                process.join(timeout=2)
            try:
                result_queue.close()
                result_queue.join_thread()
            except (OSError, ValueError) as exc:
                # Queue cleanup races: handle (already-closed / broken pipe)
                # are documented mp.Queue teardown failures we can't avoid.
                # Anything else should propagate.
                self.logger.debug(
                    "Optimizer result queue cleanup raised %s: %s",
                    type(exc).__name__,
                    exc,
                )

    async def _process_request(self, handler: RequestHandler) -> LumilakeResponse:
        if self._event_loop is None:
            raise RuntimeError("Server has not been started.")

        if await self.runtime_manager.is_request_cancelled(handler.request_id):
            self.logger.info("Request %s cancelled before enqueue.", handler.request_id)
            raise RequestCancelledError(handler.request_id)

        self.logger.info("Request %s received.", handler.request_id)

        start_time = time.perf_counter()
        await self._event_loop.add_event(handler)
        result = await handler.get_result()
        elapsed_time = time.perf_counter() - start_time

        if await self.runtime_manager.is_request_cancelled(handler.request_id):
            self.logger.info("Request %s cancelled.", handler.request_id)
            raise RequestCancelledError(handler.request_id)

        if result.error_info:
            self.logger.warning(
                "Request %s finished with errors (%.3f seconds, errors=%d).",
                handler.request_id,
                elapsed_time,
                len(result.error_info),
            )
        else:
            self.logger.info(
                "Request %s completed (%.3f seconds).",
                handler.request_id,
                elapsed_time,
            )
        return result

    async def preview_schedule(
        self,
        graphs: dict[str, CompiledGraph],
        *,
        request_id: str | None = None,
        selected_workers: list[str] | None = None,
        worker_profiles: dict[str, dict[str, Any]] | None = None,
        data_profile_results: dict[str, list[dict[str, Any]]] | None = None,
        use_subprocess: bool = True,
    ) -> SchedulePreview:
        if not graphs:
            raise ValueError("No graphs provided for preview")

        resolved_request_id = request_id or f"preview-{unique_id()}"
        runtime_graphs_by_name = {
            name: self._runtime_builder.build(graph, node_prefix=name)
            for name, graph in graphs.items()
        }
        merged_graph, _ = self.optimizer.optimize_graphs(runtime_graphs_by_name)
        merged_nodes = set(merged_graph.nodes)
        if not merged_nodes:
            raise ValueError("Preview graph has no runtime nodes")

        if selected_workers is None:
            (
                resolved_workers,
                resolved_worker_profiles,
            ) = await self._select_preview_workers_and_profiles(merged_graph)
        else:
            resolved_workers = [worker for worker in selected_workers if worker]
            if not resolved_workers:
                raise ValueError("selected_workers cannot be empty")
            if worker_profiles is None:
                resolved_worker_profiles = {
                    worker: self._normalize_worker_profile(
                        await self.runtime_manager.get_worker_profile(worker)
                    )
                    for worker in resolved_workers
                }
            else:
                resolved_worker_profiles = {}
                for worker in resolved_workers:
                    if worker not in worker_profiles:
                        raise ValueError(
                            f"Missing worker profile for preview worker '{worker}'"
                        )
                    resolved_worker_profiles[worker] = self._normalize_worker_profile(
                        dict(worker_profiles[worker])
                    )

        resolved_data_profile_results = data_profile_results or {}
        if use_subprocess:
            schedule = await self._generate_schedule_in_subprocess(
                request_id=resolved_request_id,
                batch_id="preview",
                runtime_graph=merged_graph,
                selected_workers=resolved_workers,
                worker_profiles=resolved_worker_profiles,
                data_profile_results=resolved_data_profile_results,
                member_request_ids=None,
            )
        else:
            schedule = self.optimizer.generate_schedule(
                merged_graph,
                resolved_workers,
                resolved_worker_profiles,
                resolved_data_profile_results,
            )
        self._validate_schedule(schedule, resolved_workers, merged_nodes)

        return SchedulePreview(
            request_id=resolved_request_id,
            selected_workers=resolved_workers,
            worker_profiles=resolved_worker_profiles,
            runtime_graph_node_counts={
                name: graph.node_count for name, graph in runtime_graphs_by_name.items()
            },
            merged_runtime_node_count=merged_graph.node_count,
            schedule=schedule,
        )

    async def request(self, request: LumilakeRequest) -> LumilakeResponse:
        try:
            graphs = self.parse_query(request.query)
        except Exception as e:
            err_message = f"Failed to parse query: {repr(e)}"
            self.logger.exception(err_message)
            return LumilakeResponse(error_info=[{"parsing": err_message}])

        return await self.execute(graphs, request.request_id, request.config)

    async def execute(
        self,
        graphs: dict[str, CompiledGraph],
        request_id: str | None = None,
        config: LumilakeRequestConfig | None = None,
        workflow_slices: dict[str, WorkflowSliceMeta] | None = None,
    ) -> LumilakeResponse:
        if request_id is None:
            request_id = unique_id()
        resolved_workflow_slices: dict[str, WorkflowSliceMeta] = {}
        for graph_name, compiled_graph in graphs.items():
            slice_meta = (
                None if workflow_slices is None else workflow_slices.get(graph_name)
            )
            if slice_meta is None:
                lengths = [len(values) for values in compiled_graph.inputs.values()]
                total_length = max(lengths) if lengths else 1
                varying_input_keys = tuple(
                    sorted(
                        key
                        for key, values in compiled_graph.inputs.items()
                        if len(values) > 1
                    )
                )
                template_payload = {
                    "graph": compiled_graph.graph.serialize(),
                }
                template_hash = hashlib.sha256(
                    json.dumps(
                        template_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                slice_meta = WorkflowSliceMeta(
                    public_graph_name=graph_name,
                    slice_index=0,
                    slice_start=0,
                    slice_length=total_length,
                    total_length=total_length,
                    template_hash=template_hash,
                    varying_input_keys=varying_input_keys,
                )
            resolved_workflow_slices[graph_name] = slice_meta

        runtime_graphs = {
            name: self._runtime_builder.build(graph, node_prefix=name)
            for name, graph in graphs.items()
        }
        data_profile_graphs = {
            name: self._runtime_builder.build(
                graph, task_type_override="data_profile", node_prefix=name
            )
            for name, graph in graphs.items()
        }
        handler = RequestHandler(
            runtime_graphs,
            data_profile_graphs,
            graphs,
            resolved_workflow_slices,
            request_id,
            config,
        )
        return await self._process_request(handler)

    def parse_query(self, query: dict[str, dict[str, Any]]) -> dict[str, CompiledGraph]:
        parsed_query = {
            name: Graph.from_json(graph["graph"]).compile(**graph["inputs"])
            for name, graph in query.items()
        }
        return parsed_query

    @classmethod
    def get_instance(cls, *args, **kwargs) -> "LumilakeServer":
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance

    @classmethod
    def get_started_instance(cls, *args, **kwargs) -> "LumilakeServer":
        """Asynchronously gets the server instance, starts it if not already

        It does not close the server instance after the context manager exits.
        """
        server = cls.get_instance(*args, **kwargs)
        if not server.is_started:
            server.start()
        return server

    @classmethod
    @contextmanager
    def serve_instance(cls, *args, **kwargs) -> Generator["LumilakeServer", None, None]:
        """Asynchronously gets the server instance, starts it if not already

        It closes the server instance after the context manager exits.
        """
        server = cls.get_started_instance(*args, **kwargs)
        try:
            yield server
        finally:
            if server.is_started:
                server.close()
