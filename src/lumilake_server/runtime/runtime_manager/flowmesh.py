"""
Flowmesh Submitter - Submit Lumilake logical plans to Flowmesh orchestrator.

This module handles the translation of Lumilake's optimized logical query plans
into Flowmesh task specifications and submits them via HTTP POST.
"""

import asyncio
import copy
import heapq
import json
import mimetypes
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from flowmesh.exceptions import APIError
from lumilake import envs
from lumilake.log import Logger, LogLevel, init_child_logger

from lumilake_server.runtime.flowmesh_client import (
    flowmesh_for_context,
    flowmesh_for_server,
)
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.protocol import RequestCancelledError
from lumilake_server.runtime.request import RequestInfo
from lumilake_server.runtime.runtime_graph import (
    Roles,
    RuntimeGraph,
    RuntimeGraphBuilder,
)
from lumilake_server.runtime.runtime_ops import RuntimeOp
from lumilake_server.utils.job_storage import get_job_storage

from .base import BaseRuntimeManager

TERMINAL_STATUSES = {"DONE", "FAILED"}


def _walk_output_path(
    item: Mapping[str, Any], parts: Sequence[str], output_op_id: str
) -> Any:
    """Walk a dotted ``items.<a>.<b>.<c>`` path through a result item.

    Supports nested ``dict`` traversal. For DataFrame-shaped fields that arrive
    as JSON-encoded strings (the canonical shape produced by ``mode: sql``
    retrievals when ``items.table`` is serialized for transport), the string is
    JSON-decoded once and traversal continues.
    """
    value: Any = item
    walked: list[str] = []
    for part in parts:
        walked.append(part)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"output node {output_op_id} cannot descend into non-JSON "
                    f"string at path 'items.{'.'.join(walked)}': {value!r}"
                ) from exc
        if not isinstance(value, Mapping) or part not in value:
            raise RuntimeError(
                f"output node {output_op_id} item missing field at path "
                f"'items.{'.'.join(walked)}': {item}"
            )
        value = value[part]
    return value


def _runtime_output_destination() -> dict[str, Any]:
    if envs.FLOWMESH_OUTPUT_DESTINATION == "http":
        return {"type": "http", "timeoutSec": 3600}
    return {"type": "local"}


@dataclass(slots=True)
class ShardRewriteResult:
    nodes: list[dict[str, Any]]
    worker_assignment: dict[str, list[str]]
    flowmesh_to_raw: dict[str, str]


class FlowmeshRuntimeManager(BaseRuntimeManager):
    """
    Submits Lumilake logical plans to Flowmesh orchestrator.
    """

    def __init__(
        self,
        logger: Logger | None = None,
        log_level: LogLevel | None = None,
    ) -> None:
        self.logger = init_child_logger("FlowmeshSubmitter", logger, log_level)
        self._schema_cache: dict[str, list[dict[str, Any]]] = (
            {}
        )  # Cache schema info by table name
        self._runtime_builder = RuntimeGraphBuilder(
            logger=self.logger, schema_cache=self._schema_cache
        )

        self._task_status_lock = asyncio.Lock()  # Used when modifying task status maps
        self._data_profile_task_status: dict[tuple[str, str], dict[str, str]] = (
            {}
        )  # (request_id, batch_id) -> dict of data_profile task IDs -> status
        self._execution_task_status: dict[tuple[str, str], dict[str, str]] = (
            {}
        )  # (request_id, batch_id) -> dict of execution task IDs -> status
        self._execution_output_tasks: dict[tuple[str, str], set[str]] = (
            {}
        )  # (request_id, batch_id) -> set of output task IDs
        self._task_node_map: dict[tuple[str, str], dict[str, str]] = (
            {}
        )  # (request_id, batch_id) -> task_id -> node_id

        # Batch-level tracking
        self._batch_status: dict[tuple[str, str], str] = (
            {}
        )  # (request_id, batch_id) -> "PENDING" | "RUNNING" | "COMPLETED" | "FAILED"
        self._batch_metadata: dict[tuple[str, str], dict[str, Any]] = (
            {}
        )  # (request_id, batch_id) -> {total_nodes, output_nodes, start_time, end_time}

        # FM workflow id per (request_id, batch_id); drives cancel + trace forwarding.
        self._batch_workflow_id: dict[tuple[str, str], str] = {}
        self._cancelled_requests: set[str] = set()

        # Cross-thread (FastAPI loop set/clear, _AsyncRunner loop get),
        # so threading lock, not asyncio.
        self._dispatch_tokens: dict[str, str | None] = {}
        self._dispatch_tokens_lock = threading.Lock()

    @property
    def fm(self):
        return flowmesh_for_context()

    def set_dispatch_token(self, request_id: str, token: str | None) -> None:
        with self._dispatch_tokens_lock:
            self._dispatch_tokens[request_id] = token

    def get_dispatch_token(self, request_id: str) -> str | None:
        with self._dispatch_tokens_lock:
            return self._dispatch_tokens.get(request_id)

    def clear_dispatch_token(self, request_id: str) -> None:
        with self._dispatch_tokens_lock:
            self._dispatch_tokens.pop(request_id, None)

    async def is_request_cancelled(self, request_id: str) -> bool:
        async with self._task_status_lock:
            return request_id in self._cancelled_requests

    def result_dir(self, request_info: RequestInfo) -> Path:
        return Path(request_info.request_id) / request_info.batch_id

    def _save_yaml_artifact(
        self,
        request_info: RequestInfo,
        filename: str,
        data: dict[str, Any] | str,
    ) -> str:
        yaml_data = (
            yaml.dump(data, default_flow_style=False, sort_keys=False)
            if isinstance(data, dict)
            else data
        )
        return self.save_runtime_artifact(
            request_info,
            filename,
            yaml_data.encode("utf-8"),
            "application/x-yaml",
        )

    def _save_json_artifact(
        self,
        request_info: RequestInfo,
        filename: str,
        data: Any,
    ) -> str:
        return self.save_runtime_artifact(
            request_info,
            filename,
            json.dumps(data, indent=2).encode("utf-8"),
            "application/json",
        )

    async def fetch_task_status(
        self,
        task_id: str,
    ) -> str:
        task_info = await self.fm.tasks.retrieve(task_id)
        return task_info.status

    async def fetch_task_description(
        self,
        task_id: str,
    ) -> dict[str, Any]:
        task_info = await self.fm.tasks.retrieve(task_id)
        return task_info.model_dump()

    async def _resolve_task_node_maps(
        self,
        task_ids: list[str],
        expected_node_names: list[str],
    ) -> tuple[dict[str, str], dict[str, str]]:
        expected_set = set(expected_node_names)
        task_to_node: dict[str, str] = {}
        node_to_task: dict[str, str] = {}

        descriptions = await asyncio.gather(
            *(self.fetch_task_description(task_id) for task_id in task_ids),
            return_exceptions=True,
        )
        for task_id, task_desc in zip(task_ids, descriptions):
            if isinstance(task_desc, BaseException):
                raise RuntimeError(
                    f"Failed to fetch FlowMesh task description for task_id={task_id}"
                ) from task_desc
            node_name = task_desc.get("graph_node_name")
            if not isinstance(node_name, str) or node_name not in expected_set:
                raise RuntimeError(
                    "FlowMesh task description missing valid graph_node_name for "
                    f"task_id={task_id}: value={node_name!r}"
                )
            existing_task = node_to_task.get(node_name)
            if existing_task is not None and existing_task != task_id:
                raise RuntimeError(
                    "Duplicate FlowMesh graph_node_name in task descriptions: "
                    f"node={node_name!r} task_ids=({existing_task}, {task_id})"
                )
            task_to_node[task_id] = node_name
            node_to_task[node_name] = task_id

        missing_task_ids = [
            task_id for task_id in task_ids if task_id not in task_to_node
        ]
        missing_node_names = [
            node_name
            for node_name in expected_node_names
            if node_name not in node_to_task
        ]
        if missing_task_ids or missing_node_names:
            raise RuntimeError(
                f"Task/node map is incomplete. Missing task_ids={missing_task_ids},"
                f" missing node_names={missing_node_names}, mapped={len(task_to_node)},"
                f" expected_tasks={len(task_ids)},"
                f" expected_nodes={len(expected_node_names)}"
            )
        return task_to_node, node_to_task

    async def update_task_status(
        self,
        task_status: dict[str, str],
    ) -> bool:
        completed = True
        for task_id, status in list(task_status.items()):
            if status not in TERMINAL_STATUSES:
                latest_status = await self.fetch_task_status(task_id)
                async with self._task_status_lock:
                    task_status[task_id] = latest_status
                if latest_status not in TERMINAL_STATUSES:
                    completed = False
        return completed

    def _formulate_details(
        self,
        task_status: dict[str, str],
    ) -> dict[str, int]:
        return {
            "succeeded": len([t for t in task_status.values() if t == "DONE"]),
            "failed": len([t for t in task_status.values() if t == "FAILED"]),
            "pending": len([t for t in task_status.values() if t == "PENDING"]),
            "dispatched": len([t for t in task_status.values() if t == "DISPATCHED"]),
        }

    def _output_task_status_aggregated(
        self, execution_batch_keys: list[tuple[str, str]]
    ):
        """Aggregate output task status across all batches for a request."""
        aggregated_output_status = {}
        for batch_key in execution_batch_keys:
            for task_id in self._execution_output_tasks[batch_key]:
                aggregated_output_status[task_id] = self._execution_task_status[
                    batch_key
                ][task_id]
        return aggregated_output_status

    def mark_batch_pending(
        self,
        request_id: str,
        batch_id: str,
        total_nodes: int,
        output_nodes: int,
    ) -> None:
        """Mark a batch as pending (not yet started)."""
        batch_key = (request_id, batch_id)
        self._batch_status[batch_key] = "PENDING"
        self._batch_metadata[batch_key] = {
            "total_nodes": total_nodes,
            "output_nodes": output_nodes,
            "start_time": None,
            "end_time": None,
            "raw_nodes": total_nodes,
            "flowmesh_nodes": total_nodes,
        }

    def mark_batch_running(self, request_id: str, batch_id: str) -> None:
        """Mark a batch as running (started processing)."""
        batch_key = (request_id, batch_id)
        self._batch_status[batch_key] = "RUNNING"
        metadata = self._batch_metadata.setdefault(
            batch_key,
            {
                "total_nodes": 0,
                "output_nodes": 0,
                "start_time": None,
                "end_time": None,
                "raw_nodes": 0,
                "flowmesh_nodes": 0,
            },
        )
        if metadata.get("start_time") is None:
            metadata["start_time"] = time.time()
        metadata["end_time"] = None

    def mark_batch_completed(self, request_id: str, batch_id: str) -> None:
        """Mark a batch as completed (all tasks finished successfully)."""
        batch_key = (request_id, batch_id)
        self._batch_status[batch_key] = "COMPLETED"
        metadata = self._batch_metadata.setdefault(
            batch_key,
            {
                "total_nodes": 0,
                "output_nodes": 0,
                "start_time": None,
                "end_time": None,
                "raw_nodes": 0,
                "flowmesh_nodes": 0,
            },
        )
        if metadata.get("start_time") is None:
            metadata["start_time"] = time.time()
        if metadata.get("end_time") is None:
            metadata["end_time"] = time.time()

    def mark_batch_failed(self, request_id: str, batch_id: str) -> None:
        """Mark a batch as failed (one or more tasks failed)."""
        batch_key = (request_id, batch_id)
        self._batch_status[batch_key] = "FAILED"
        metadata = self._batch_metadata.setdefault(
            batch_key,
            {
                "total_nodes": 0,
                "output_nodes": 0,
                "start_time": None,
                "end_time": None,
                "raw_nodes": 0,
                "flowmesh_nodes": 0,
            },
        )
        if metadata.get("start_time") is None:
            metadata["start_time"] = time.time()
        if metadata.get("end_time") is None:
            metadata["end_time"] = time.time()

    def get_task_node_map(self, request_id: str, batch_id: str) -> dict[str, str]:
        batch_key = (request_id, batch_id)
        return dict(self._task_node_map.get(batch_key, {}))

    def _build_batch_progress(
        self, batch_keys: list[tuple[str, str]]
    ) -> dict[str, Any]:
        """Build detailed batch progress information with ETA."""

        def _as_non_negative_int(value: Any) -> int:
            if isinstance(value, (int, float)):
                return max(0, int(value))
            return 0

        batches = []
        total_batches = len(batch_keys)
        completed_count = 0
        running_count = 0
        pending_count = 0
        failed_count = 0

        total_nodes_overall = 0
        completed_nodes_overall = 0
        raw_nodes_overall = 0
        flowmesh_nodes_overall = 0
        completed_batches_elapsed = 0.0

        for batch_key in batch_keys:
            batch_id = batch_key[1]
            status = self._batch_status.get(batch_key, "PENDING")
            metadata = self._batch_metadata.get(batch_key, {})

            flowmesh_nodes = _as_non_negative_int(metadata.get("flowmesh_nodes"))
            raw_nodes = _as_non_negative_int(metadata.get("raw_nodes"))
            start_time = metadata.get("start_time")
            end_time = metadata.get("end_time")
            raw_nodes_overall += raw_nodes
            flowmesh_nodes_overall += flowmesh_nodes

            # Count batch statuses
            if status == "COMPLETED":
                completed_count += 1
            elif status == "RUNNING":
                running_count += 1
            elif status == "FAILED":
                failed_count += 1
            else:  # PENDING
                pending_count += 1

            # Build batch info
            batch_info: dict[str, Any] = {
                "batch_id": batch_id,
                "status": status,
                "nodes": {},
            }

            # Add elapsed time if batch has started
            if start_time is not None:
                if isinstance(end_time, (int, float)) and status in {
                    "COMPLETED",
                    "FAILED",
                }:
                    elapsed = end_time - start_time
                else:
                    elapsed = time.time() - start_time
                if elapsed < 0:
                    elapsed = 0.0
                batch_info["elapsed_time"] = round(elapsed, 1)
                if status == "COMPLETED":
                    completed_batches_elapsed += elapsed

            # Add detailed node stats for execution tasks
            if batch_key in self._execution_task_status:
                task_status = self._execution_task_status[batch_key]
                node_details = self._formulate_details(task_status)
                batch_info["nodes"].update(node_details)
                batch_info["nodes"]["total"] = sum(node_details.values())

                # Track overall progress using displayed total
                total_nodes_overall += sum(node_details.values())
                completed_nodes_overall += (
                    node_details["succeeded"] + node_details["failed"]
                )
            else:
                batch_info["nodes"]["total"] = flowmesh_nodes
                total_nodes_overall += flowmesh_nodes

            batches.append(batch_info)

        # Calculate ETA based on completed batches' average time
        eta = None
        if completed_count > 0 and (running_count > 0 or pending_count > 0):
            avg_batch_time = completed_batches_elapsed / completed_count
            remaining_batches = running_count + pending_count
            eta = round(avg_batch_time * remaining_batches, 1)

        return {
            "total": total_batches,
            "completed": completed_count,
            "running": running_count,
            "pending": pending_count,
            "failed": failed_count,
            "batches": batches,
            "overall_progress": {
                "total_nodes": total_nodes_overall,
                "completed_nodes": completed_nodes_overall,
                "percentage": (
                    round(completed_nodes_overall / total_nodes_overall * 100, 1)
                    if total_nodes_overall > 0
                    else 0
                ),
                "raw_nodes": raw_nodes_overall,
                "flowmesh_nodes": flowmesh_nodes_overall,
            },
            "eta_seconds": eta,
        }

    async def get_workers(self) -> list[str]:
        """
        Fetch the list of available workers from the orchestrator.

        Returns
        -------
        list[str]
            List of worker IDs
        """
        workers = await flowmesh_for_server().workers.list(status="IDLE")
        return [w.id for w in workers]

    def count_runtime_nodes(self, graphs: dict[str, RuntimeGraph]) -> int:
        return sum(graph.node_count for graph in graphs.values())

    async def get_worker_profile(self, worker_id: str) -> dict[str, Any]:
        """
        Fetch the profile of a given worker.

        Parameters
        ----------
        worker_id : str
            The ID of the worker.

        Returns
        -------
        dict[str, Any]
            The worker profile including capabilities and current load.
        """
        worker = await flowmesh_for_server().workers.retrieve(worker_id)
        assert worker.id == worker_id
        return worker.hardware.model_dump() if worker.hardware else {}

    async def get_request_status(
        self,
        request_id: str,
    ) -> dict[str, Any]:
        # Find all batch keys for this request_id from all tracking dictionaries
        data_profile_batch_keys = [
            k for k in self._data_profile_task_status.keys() if k[0] == request_id
        ]
        execution_batch_keys = [
            k for k in self._execution_output_tasks.keys() if k[0] == request_id
        ]
        batch_status_keys = [k for k in self._batch_status.keys() if k[0] == request_id]

        # Combine all batch keys
        all_batch_keys = sorted(
            set(data_profile_batch_keys + execution_batch_keys + batch_status_keys),
            key=lambda x: x[1],
        )

        # If request_id not found in any batch, return error
        if not all_batch_keys:
            return {"error": "Request ID not found"}

        aggregated_status: dict[str, Any] = {}

        # Aggregate data_profile tasks status across all batches
        if data_profile_batch_keys:
            # Merge all batches' data_profile tasks
            all_data_profile_tasks: dict[str, str] = {}
            for batch_key in data_profile_batch_keys:
                all_data_profile_tasks.update(self._data_profile_task_status[batch_key])

            data_profile_completed = await self.update_task_status(
                all_data_profile_tasks
            )
            aggregated_status["data probing"] = {
                "completed": data_profile_completed,
                "details": self._formulate_details(all_data_profile_tasks),
            }
        else:  # data_profile is skipped
            aggregated_status["data probing"] = {"completed": True}

        # Aggregate execution tasks status across all batches
        if execution_batch_keys:
            # Merge all batches' execution tasks
            all_execution_tasks: dict[str, str] = {}
            for batch_key in execution_batch_keys:
                all_execution_tasks.update(self._execution_task_status[batch_key])

            execution_completed = await self.update_task_status(all_execution_tasks)
            aggregated_status["execution"] = {
                "completed": execution_completed,
                "details": self._formulate_details(all_execution_tasks),
            }

            output_task_status = self._output_task_status_aggregated(
                execution_batch_keys
            )
            output_completed = all(
                status in TERMINAL_STATUSES for status in output_task_status.values()
            )
            aggregated_status["outputs"] = {
                "completed": output_completed,
                "details": self._formulate_details(output_task_status),
            }

        # Add batch-level progress information (using all_batch_keys found earlier)
        if all_batch_keys:
            batch_progress = self._build_batch_progress(all_batch_keys)
            aggregated_status["batch_progress"] = batch_progress

        return aggregated_status

    async def process_request(
        self,
        request_info: RequestInfo,
        schedule: Schedule,
        worker_ids: list[str],
        data_profile_results: dict[str, list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """
        Submit a batch of graphs to Flowmesh with a schedule hint and worker assignment.

        Parameters
        ----------
        request_info : RequestInfo
            Contains the runtime graph and request metadata.
        schedule : Schedule
            Worker assignment.
        worker_ids : list[str]
            Workers selected to execute this batch.
        data_profile_results : dict[str, list[dict[str, Any]]], optional
            Data profile results, keyed by
            ``"data_profile::<node_id>::<query_name>"``.
        Returns
        -------
        dict
            - flat_outputs: dict[op_id, output_text] for optimizer remapping
            - prompts: dict[op_id, prompt_metadata] for history tracking
        """
        if await self.is_request_cancelled(request_info.request_id):
            self.logger.info(
                "Skipping request %s (cancelled before submission)",
                request_info.request_id,
            )
            raise RequestCancelledError(request_info.request_id)
        self.logger.info(
            f"Submitting batch {request_info.batch_id} in {request_info.request_id}"
            " with schedule hint"
            f" ({sum(len(nodes) for nodes in schedule.worker_assignment.values())}"
            f" nodes, workers: {worker_ids})"
        )
        # Update cache with data_profile results if provided
        if data_profile_results:
            self._schema_cache.update(data_profile_results)
            self.logger.info(
                f"Updated schema cache with {len(data_profile_results)} data profile"
                " results"
            )

        self.logger.info(f"Submitting request {request_info.request_id} to Flowmesh.")

        # Build task spec from the runtime graph.
        task_spec, output_node_indices, flowmesh_to_raw = await self._build_task_spec(
            request_info,
            request_info.runtime_graph,
            schedule=schedule,
        )
        flowmesh_node_count = len(task_spec["spec"]["graph"].get("nodes", []))
        raw_node_count = len(request_info.runtime_graph.node_order)
        task_yaml = yaml.dump(task_spec, default_flow_style=False, sort_keys=False)
        graph_uri = self._save_yaml_artifact(
            request_info,
            "lumilake-runtime-graph.yaml",
            {
                "node_order": request_info.runtime_graph.node_order,
                "output_nodes": sorted(
                    request_info.runtime_graph.output_node_map.keys()
                ),
                "raw_nodes": raw_node_count,
                "flowmesh_nodes": flowmesh_node_count,
            },
        )
        job_uri = self._save_yaml_artifact(
            request_info,
            "flowmesh_job.yaml",
            task_yaml,
        )
        self.logger.info(
            "Archived runtime graph to %s and FlowMesh job spec to %s",
            graph_uri,
            job_uri,
        )

        # Submit task
        if await self.is_request_cancelled(request_info.request_id):
            self.logger.info(
                "Skipping runtime submission for %s (cancelled)",
                request_info.request_id,
            )
            raise RequestCancelledError(request_info.request_id)
        try:
            submit_resp = await self.fm.workflows.submit(task_yaml)
        except APIError as e:
            body = str(e.body) if hasattr(e, "body") else str(e)
            if len(body) > 2000:
                body = body[:2000] + "...[truncated]"
            self.logger.error(
                "Flowmesh request failed: %s. Body: %s",
                e,
                body,
            )
            raise

        task_ids = [t.task_id for t in submit_resp.tasks]
        self.logger.info(f"Flowmesh accepted {len(task_ids)} tasks: {task_ids}")
        node_names = [
            node.get("name")
            for node in task_spec["spec"]["graph"].get("nodes", [])
            if node.get("name")
        ]
        task_node_map, node_task_map = await self._resolve_task_node_maps(
            task_ids, node_names
        )

        # Store FM workflow id for cancel + trace forwarding.
        async with self._task_status_lock:
            self._batch_workflow_id[
                (request_info.request_id, request_info.batch_id)
            ] = submit_resp.workflow_id

        total_nodes = len(task_ids)
        output_nodes = len(output_node_indices)
        raw_task_node_map: dict[str, str] = {}

        async with self._task_status_lock:
            batch_key = (request_info.request_id, request_info.batch_id)
            self._execution_output_tasks[batch_key] = set(task_ids)
            self._execution_task_status[batch_key] = {
                tid: "PENDING" for tid in task_ids
            }
            for task_id, node_name in task_node_map.items():
                if node_name not in flowmesh_to_raw:
                    raise ValueError(
                        "Missing raw-node mapping for FlowMesh node "
                        f"'{node_name}' (task_id={task_id})"
                    )
                raw_task_node_map[task_id] = flowmesh_to_raw[node_name]
            self._task_node_map[batch_key] = raw_task_node_map
            metadata = self._batch_metadata.setdefault(
                batch_key,
                {
                    "total_nodes": 0,
                    "output_nodes": 0,
                    "start_time": None,
                    "end_time": None,
                },
            )
            metadata["raw_nodes"] = raw_node_count
            metadata["flowmesh_nodes"] = flowmesh_node_count

        mapping_uri = self._save_json_artifact(
            request_info,
            "task-node-map.json",
            raw_task_node_map,
        )
        self.logger.info(
            "Archived task-node mapping (%d tasks) to %s",
            len(raw_task_node_map),
            mapping_uri,
        )

        # Poll for completion
        poll_timeout = envs.LUMILAKE_POLL_TIMEOUT_SECONDS
        poll_interval = envs.LUMILAKE_POLL_INTERVAL_SECONDS
        downloaded_tasks: set[str] = set()
        output_terminal = False
        output_task_status: dict[str, str] = {}
        start_time = time.time()
        batch_key = (request_info.request_id, request_info.batch_id)
        while (elapsed := time.time() - start_time) < poll_timeout:
            if await self.is_request_cancelled(request_info.request_id):
                self.logger.info(
                    "Stopping request %s (cancelled during execution)",
                    request_info.request_id,
                )
                raise RequestCancelledError(request_info.request_id)
            await self.update_task_status(self._execution_task_status[batch_key])

            # Download responses for newly completed tasks
            for tid in task_ids:
                status = self._execution_task_status[batch_key][tid]
                if status == "DONE" and tid not in downloaded_tasks:
                    node_id = task_node_map.get(tid)
                    if node_id is None:
                        raise RuntimeError(
                            "Missing task->node mapping for task "
                            f"{tid}. Known task ids={sorted(task_node_map.keys())}"
                        )
                    response_data = await self.fm.results.retrieve(tid)
                    response_uri = self._save_json_artifact(
                        request_info,
                        f"per-task-response/{tid}.json",
                        response_data,
                    )
                    self.logger.info(
                        "Archived response for task %s (%s) to %s",
                        tid,
                        node_id,
                        response_uri,
                    )
                    downloaded_tasks.add(tid)

            # Count statuses for all nodes and output nodes
            all_statuses = list(self._execution_task_status[batch_key].values())
            status_counts = {s: all_statuses.count(s) for s in set(all_statuses)}
            output_task_status = {
                task_id: self._execution_task_status[batch_key][task_id]
                for task_id in self._execution_output_tasks[batch_key]
            }
            output_status_counts = {
                s: list(output_task_status.values()).count(s)
                for s in output_task_status.values()
            }

            self.logger.info(
                f"Polling {total_nodes} nodes ({output_nodes} outputs):"
                f" all={status_counts}, outputs={output_status_counts} | elapsed:"
                f" {elapsed:.1f}s"
            )

            # Check if all output nodes are terminal
            output_terminal = all(
                status in TERMINAL_STATUSES for status in output_task_status.values()
            )

            if output_terminal:
                self.logger.info(
                    "All output nodes reached terminal status. Total elapsed:"
                    f" {elapsed:.1f}s"
                )
                break

            # Fail fast: any task failure terminates the workflow. Waiting
            # for output-node terminality alone leaves the user staring at
            # a "running" job for the full poll timeout when an upstream
            # task has already crashed the downstream chain.
            failed_tasks = [
                tid
                for tid, status in self._execution_task_status[batch_key].items()
                if status == "FAILED"
            ]
            if failed_tasks:
                self.logger.error(
                    "Fast-failing request %s: %d task(s) reported FAILED at"
                    " elapsed=%.1fs (first: %s)",
                    request_info.request_id,
                    len(failed_tasks),
                    elapsed,
                    failed_tasks[0],
                )
                raise RuntimeError(
                    f"Task {failed_tasks[0]} failed; aborting workflow"
                    f" ({len(failed_tasks)} task(s) failed in total)"
                )

            await asyncio.sleep(poll_interval)

        # Check timeout
        if not output_terminal:
            raise RuntimeError(
                f"Timeout waiting for output nodes to complete after {poll_timeout}s"
            )

        # Check for failures in output nodes
        for tid, status in output_task_status.items():
            if status == "FAILED":
                raise RuntimeError(f"Output task {tid} failed")

        # Aggregate results from output nodes
        flat_outputs: dict[str, Any] = {}
        prompts: dict[str, Any] = {}

        for _, output_op_id in output_node_indices:
            output_task_id = node_task_map.get(output_op_id)
            if not output_task_id:
                raise RuntimeError(
                    "Failed to resolve output node task mapping for "
                    f"node={output_op_id}. Known nodes={sorted(node_task_map.keys())}"
                )

            results_json = await self.fm.results.retrieve(output_task_id)
            items = results_json.get("items")
            if not isinstance(items, list) or not items:
                raise RuntimeError(f"output {output_op_id} produced no items")

            if any(isinstance(it.get("image"), dict) for it in items):
                job_storage = get_job_storage()
                if not envs.S3_ARCHIVE_PREFIX:
                    raise RuntimeError(
                        "S3_ARCHIVE_PREFIX is required for image outputs"
                    )
                archived: list[dict[str, Any]] = []
                for it in items:
                    image_ref = it["image"]
                    path = image_ref.get("path")
                    if not path:
                        raise RuntimeError(f"item missing image.path: {it}")
                    raw_name = image_ref.get("filename") or Path(path).name
                    filename = f"{output_op_id}-{Path(raw_name).name}"
                    try:
                        with tempfile.NamedTemporaryFile(delete=False) as tmp:
                            tmp_path = Path(tmp.name)
                        try:
                            await self.fm.results.download_file(
                                output_task_id, f"artifacts/{path}", tmp_path
                            )
                            data = tmp_path.read_bytes()
                        finally:
                            tmp_path.unlink(missing_ok=True)
                        content_type = (
                            mimetypes.guess_type(filename)[0]
                            or "application/octet-stream"
                        )
                        uri = job_storage.save_artifact(
                            request_info.request_id,
                            filename,
                            data,
                            content_type,
                        )
                        archived.append({"output": uri})
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to archive artifact for {output_op_id}: {e}"
                        )
                        archived.append({"output": "", "error": str(e)})
                items = archived
                # Image-archive items have shape ``{"output": <uri>}`` —
                # the user's path override (e.g. ``items.table``) is N/A here.
                output_field_parts: tuple[str, ...] = ("output",)
            else:
                output_field_parts = ("output",)
                output_path = request_info.runtime_graph.output_paths.get(output_op_id)
                if output_path is not None:
                    if (
                        not isinstance(output_path, str)
                        or not output_path.startswith("items.")
                        or output_path == "items."
                    ):
                        raise RuntimeError(
                            f"OutputOp {output_op_id!r} has malformed path "
                            f"{output_path!r}"
                        )
                    parts = tuple(
                        part for part in output_path[len("items.") :].split(".") if part
                    )
                    if not parts:
                        raise RuntimeError(
                            f"OutputOp {output_op_id!r} has malformed path "
                            f"{output_path!r}"
                        )
                    output_field_parts = parts
            outputs: list[str] = []
            for item in items:
                value: Any = _walk_output_path(item, output_field_parts, output_op_id)
                if isinstance(value, (dict, list)):
                    outputs.append(json.dumps(value))
                else:
                    outputs.append(value)
            flat_outputs[output_op_id] = outputs
            output_prompts: list[list[dict[str, str]]] = []
            for item, text in zip(items, outputs):
                try:
                    prompt = item["metadata"]["prompt"]
                except (KeyError, TypeError):
                    continue
                if not isinstance(prompt, list):
                    continue
                output_prompts.append(
                    prompt + [{"role": Roles.ASSISTANT.value, "content": text}]
                )
            if output_prompts:
                prompts[output_op_id] = output_prompts

        self.logger.info(f"Aggregated {len(flat_outputs)} output results.")
        return {
            "flat_outputs": flat_outputs,
            "chat_histories": prompts,
            "task_node_map": dict(raw_task_node_map),
        }

    async def _build_task_spec(
        self,
        request_info: RequestInfo,
        runtime_graph: RuntimeGraph,
        schedule: Schedule | None = None,
    ) -> tuple[dict[str, Any], list[tuple[int, str]], dict[str, str]]:
        """
        Build a Flowmesh-compatible task specification from Lumilake's runtime graph.

        Parameters
        ----------
        request_info : RequestInfo
            Request metadata including request_id and config flags
        runtime_graph : RuntimeGraph
            The runtime graph to submit
        schedule : Schedule, optional
            Worker assignment.

        Returns
        -------
        tuple
            - Flowmesh YAML task specification
            - Output node indices
            - Flowmesh node name -> raw runtime node ID mapping
        """
        nodes = runtime_graph.to_flowmesh_nodes()
        self._apply_per_node_resource_hints(nodes, runtime_graph)
        output_node_indices: list[tuple[int, str]] = []
        for idx, node_id in enumerate(runtime_graph.node_order):
            if node_id in runtime_graph.output_node_map:
                output_node_indices.append((idx, node_id))
        flowmesh_to_raw: dict[str, str] = {}
        for node in nodes:
            node_name = node.get("name")
            if not isinstance(node_name, str) or not node_name:
                raise ValueError("FlowMesh node is missing a valid 'name'")
            flowmesh_to_raw[node_name] = node_name

        active_worker_assignment = (
            {
                worker: list(node_ids)
                for worker, node_ids in schedule.worker_assignment.items()
            }
            if schedule is not None
            else {}
        )
        if schedule is not None:
            rewrite = self._rewrite_nodes_for_shard_intent(
                nodes=nodes,
                worker_assignment=active_worker_assignment,
                output_node_names=[node_id for _, node_id in output_node_indices],
            )
            nodes = rewrite.nodes
            active_worker_assignment = rewrite.worker_assignment
            flowmesh_to_raw = rewrite.flowmesh_to_raw

        spec: dict[str, Any] = {
            "apiVersion": "mloc/v1",
            "kind": "InferenceTask",
            "metadata": {
                "name": f"lumilake-{request_info.request_id}",
                "owner": "lumilake",
                "annotations": {
                    "description": (
                        "Lumilake logical plan execution for request"
                        f" {request_info.request_id}"
                    ),
                },
            },
            "spec": {
                "taskType": "inference",
                "resources": {
                    "replicas": 1,
                    "hardware": {
                        "cpu": envs.HARDWARE_CPU_REQUIREMENT,
                        "memory": envs.HARDWARE_MEMORY_REQUIREMENT,
                    },
                },
                "graph": {
                    "nodes": nodes,
                },
                "output": {
                    "destination": _runtime_output_destination(),
                    "artifacts": ["results.json", "logs"],
                },
            },
        }

        # Add schedule hint as annotation if provided.
        if schedule is not None:
            node_names = []
            for node in nodes:
                node_name = node.get("name")
                if not isinstance(node_name, str) or not node_name:
                    raise ValueError(
                        "FlowMesh rewritten node is missing a valid 'name'"
                    )
                node_names.append(node_name)
            (
                schedule_hint,
                per_node_selected_worker_candidates,
            ) = self._build_flat_schedule_hint(
                worker_assignment=active_worker_assignment,
                node_names=node_names,
                node_dependencies={
                    node_name: [
                        dep
                        for dep in node.get("dependsOn") or []
                        if isinstance(dep, str)
                    ]
                    for node_name, node in (
                        (str(item.get("name")), item) for item in nodes
                    )
                },
            )
            schedule_hint_annotation: dict[str, Any] = {
                "node_execution_order": schedule_hint,
            }
            if per_node_selected_worker_candidates:
                schedule_hint_annotation["selected_worker"] = {
                    "selected": per_node_selected_worker_candidates
                }
            spec["metadata"]["annotations"]["schedule_hint"] = schedule_hint_annotation

        return spec, output_node_indices, flowmesh_to_raw

    def _apply_per_node_resource_hints(
        self,
        nodes: list[dict[str, Any]],
        runtime_graph: RuntimeGraph,
    ) -> None:
        gpu_nodes = 0
        cpu_nodes = 0
        for node in nodes:
            node_name = node.get("name")
            if not isinstance(node_name, str):
                continue
            runtime_op = runtime_graph.nodes.get(node_name)
            if runtime_op is None:
                continue
            if self._runtime_op_requires_gpu(runtime_op):
                gpu_nodes += 1
                node_spec = node.setdefault("spec", {})
                if not isinstance(node_spec, dict):
                    continue
                node_resources = node_spec.setdefault("resources", {})
                if not isinstance(node_resources, dict):
                    continue
                node_hardware = node_resources.setdefault("hardware", {})
                if not isinstance(node_hardware, dict):
                    continue
                node_hardware["cpu"] = envs.HARDWARE_CPU_REQUIREMENT
                node_hardware["memory"] = envs.HARDWARE_MEMORY_REQUIREMENT
                node_hardware["gpu"] = {
                    "type": "any",
                    "count": max(1, int(envs.HARDWARE_GPU_REQUIREMENT)),
                    "memory": envs.HARDWARE_GPU_MEMORY_REQUIREMENT,
                }
            else:
                cpu_nodes += 1
        self.logger.info(
            "Applied per-node resource hints: gpu_required_nodes=%d"
            " cpu_eligible_nodes=%d",
            gpu_nodes,
            cpu_nodes,
        )

    @staticmethod
    def _runtime_op_requires_gpu(runtime_op: RuntimeOp) -> bool:
        backend = (runtime_op.backend or "").strip().lower()
        if backend in {"vllm", "transformers", "diffusers"}:
            return True
        task_type = (runtime_op.task_type or "").strip().lower()
        return task_type in {"inference", "embedding", "diffusion"}

    @staticmethod
    def _build_flat_schedule_hint(
        *,
        worker_assignment: Mapping[str, Sequence[str]],
        node_names: list[str],
        node_dependencies: Mapping[str, Sequence[str]],
    ) -> tuple[list[str], dict[str, list[str]]]:
        per_node_selected_worker_candidates: dict[str, list[str]] = {}
        node_set = set(node_names)
        node_to_workers: dict[str, list[str]] = {}
        seen_nodes: set[str] = set()
        first_seen_rank: dict[str, int] = {}
        next_rank = 0

        for worker, worker_nodes in worker_assignment.items():
            local_seen: set[str] = set()
            for node_name in worker_nodes:
                if node_name in local_seen:
                    raise ValueError(
                        "Invalid schedule hint: duplicate node found within worker "
                        f"assignment for worker '{worker}': {node_name}"
                    )
                local_seen.add(node_name)
                if node_name not in node_set:
                    raise ValueError(
                        "Invalid schedule hint: worker assignment references unknown"
                        f" node '{node_name}'"
                    )
                candidates = node_to_workers.setdefault(node_name, [])
                if worker not in candidates:
                    candidates.append(worker)
                if node_name not in seen_nodes:
                    seen_nodes.add(node_name)
                    first_seen_rank[node_name] = next_rank
                    next_rank += 1

        if seen_nodes != node_set:
            missing = sorted(node_set - seen_nodes)
            extra = sorted(seen_nodes - node_set)
            raise ValueError(
                "Invalid internal schedule: worker assignment does not cover graph"
                f" nodes. Missing: {missing}, extra: {extra}"
            )
        for node_name in node_names:
            per_node_selected_worker_candidates[node_name] = list(
                node_to_workers[node_name]
            )

        order_index = {node_id: idx for idx, node_id in enumerate(node_names)}
        in_degree: dict[str, int] = {node_id: 0 for node_id in node_names}
        children: dict[str, list[str]] = {node_id: [] for node_id in node_names}
        for node_id in node_names:
            for dep in node_dependencies.get(node_id, ()):
                if dep not in node_set:
                    continue
                if dep == node_id:
                    raise ValueError(
                        "Invalid schedule hint dependency graph: self dependency on"
                        f" '{node_id}'"
                    )
                in_degree[node_id] += 1
                children[dep].append(node_id)

        heap: list[tuple[int, int, str]] = []
        for node_id in node_names:
            if in_degree[node_id] != 0:
                continue
            heapq.heappush(
                heap,
                (
                    first_seen_rank.get(node_id, 10**9),
                    order_index[node_id],
                    node_id,
                ),
            )

        ordered_nodes: list[str] = []
        while heap:
            _, _, node_id = heapq.heappop(heap)
            ordered_nodes.append(node_id)
            for child in children[node_id]:
                in_degree[child] -= 1
                if in_degree[child] != 0:
                    continue
                heapq.heappush(
                    heap,
                    (
                        first_seen_rank.get(child, 10**9),
                        order_index[child],
                        child,
                    ),
                )

        if len(ordered_nodes) != len(node_names):
            raise ValueError(
                "Invalid schedule hint dependency graph: cycle detected in flowmesh"
                " nodes"
            )

        return ordered_nodes, per_node_selected_worker_candidates

    def _rewrite_nodes_for_shard_intent(
        self,
        *,
        nodes: list[dict[str, Any]],
        worker_assignment: Mapping[str, Sequence[str]],
        output_node_names: Sequence[str] = (),
    ) -> ShardRewriteResult:
        node_map: dict[str, dict[str, Any]] = {}
        node_order: list[str] = []
        for node in nodes:
            name = node.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("FlowMesh node is missing a valid 'name'")
            if name in node_map:
                raise ValueError(f"Duplicate FlowMesh node name: {name}")
            node_map[name] = node
            node_order.append(name)

        node_candidates, normalized_assignment = self._collect_node_worker_candidates(
            worker_assignment=worker_assignment,
            node_names=node_order,
        )
        baseline_node_candidates = {
            node_id: list(workers) for node_id, workers in node_candidates.items()
        }
        explicit_shard_intent_nodes = {
            node_id: workers
            for node_id, workers in baseline_node_candidates.items()
            if len(workers) > 1
        }
        if not explicit_shard_intent_nodes:
            return ShardRewriteResult(
                nodes=[copy.deepcopy(node_map[name]) for name in node_order],
                worker_assignment=normalized_assignment,
                flowmesh_to_raw={name: name for name in node_order},
            )
        shard_nodes_by_raw: dict[str, list[str]] = {}
        partitions_by_raw: dict[str, list[tuple[int, int]]] = {}
        effective_node_candidates = {
            node_id: list(workers) for node_id, workers in node_candidates.items()
        }
        rewrite_order = self._build_shard_rewrite_topological_order(
            node_order=node_order,
            node_map=node_map,
        )

        for node_id in rewrite_order:
            workers = explicit_shard_intent_nodes.get(node_id)
            if workers is None:
                continue
            shard_count = len(workers)
            fail_reason: tuple[str, str] | None = None
            try:
                spec = node_map[node_id].get("spec")
                if not isinstance(spec, dict):
                    fail_reason = (
                        "non-partitionable boundary",
                        "node spec is not a mapping",
                    )
                else:
                    spec_data = spec.get("data")
                    if not isinstance(spec_data, Mapping):
                        fail_reason = (
                            "non-partitionable boundary",
                            "spec.data is missing or invalid",
                        )
                    else:
                        deps = node_map[node_id].get("dependsOn")
                        dep_list = (
                            [dep for dep in deps if isinstance(dep, str)]
                            if isinstance(deps, list)
                            else []
                        )
                        local_partition_total = self._resolve_static_partition_total(
                            raw_node_id=node_id,
                            data_spec=spec_data,
                            context_node_id=node_id,
                        )
                        dep_shard_counts = [
                            len(shard_nodes_by_raw[dep])
                            for dep in dep_list
                            if dep in shard_nodes_by_raw
                        ]
                        if fail_reason is None and local_partition_total is None:
                            if not dep_list:
                                fail_reason = (
                                    "non-partitionable boundary",
                                    (
                                        "node has no partitionable local inputs or"
                                        " dependencies"
                                    ),
                                )
                            elif not dep_shard_counts:
                                inferred_dep_total = (
                                    self._infer_unsharded_dependency_total(
                                        raw_node_id=node_id,
                                        dependency_ids=dep_list,
                                        node_map=node_map,
                                    )
                                )
                                if (
                                    inferred_dep_total is None
                                    or inferred_dep_total <= 1
                                ):
                                    fail_reason = (
                                        "unresolved sharded deps",
                                        (
                                            "node has no local partition and no sharded"
                                            " dependencies"
                                        ),
                                    )
                        if fail_reason is None and dep_shard_counts:
                            expected_dep_shards = dep_shard_counts[0]
                            if any(
                                count != expected_dep_shards
                                for count in dep_shard_counts
                            ):
                                fail_reason = (
                                    "cardinality mismatch",
                                    "dependency shard counts do not align",
                                )
                            else:
                                shard_count = min(shard_count, expected_dep_shards)
                        if fail_reason is None and shard_count <= 0:
                            fail_reason = (
                                "non-partitionable boundary",
                                "no active shard worker is available",
                            )
                        if fail_reason is None:
                            total_items = self._infer_shard_input_size(
                                raw_node_id=node_id,
                                spec=spec,
                                shard_count=shard_count,
                                dependency_ids=dep_list,
                                partitions_by_raw=partitions_by_raw,
                                node_map=node_map,
                            )
                            shard_count = min(shard_count, max(1, total_items))
                            active_workers = workers[:shard_count]
                            if len(active_workers) != shard_count:
                                fail_reason = (
                                    "non-partitionable boundary",
                                    "failed to resolve active shard workers",
                                )
                            else:
                                effective_node_candidates[node_id] = active_workers
                                partitions_by_raw[node_id] = (
                                    self._build_index_partitions(
                                        total_items, shard_count
                                    )
                                )
                                shard_nodes_by_raw[node_id] = [
                                    f"{node_id}__shard_{idx}"
                                    for idx in range(shard_count)
                                ]
            except ValueError as err:
                fail_reason = (
                    self._classify_shard_boundary_error(str(err)),
                    str(err),
                )

            if fail_reason is None:
                continue
            reason_category, reason_detail = fail_reason
            raise ValueError(
                "Explicit shard-intent rewrite failed for raw node "
                f"'{node_id}' (reason: {reason_category}): {reason_detail}"
            )

        self._pin_non_explicit_unsharded_nodes(
            explicit_shard_intent_nodes=set(explicit_shard_intent_nodes),
            shard_nodes_by_raw=shard_nodes_by_raw,
            node_candidates=effective_node_candidates,
            baseline_node_candidates=baseline_node_candidates,
        )
        rewritten_input_assignment = self._prune_worker_assignment_to_candidates(
            worker_assignment=normalized_assignment,
            node_candidates=effective_node_candidates,
        )

        merge_required_nodes = self._compute_merge_required_nodes(
            node_order=node_order,
            node_map=node_map,
            shard_nodes_by_raw=shard_nodes_by_raw,
            output_node_names=set(output_node_names),
        )

        rewritten_nodes: list[dict[str, Any]] = []
        flowmesh_to_raw: dict[str, str] = {}
        for node_id in node_order:
            base_node = node_map[node_id]
            shard_names = shard_nodes_by_raw.get(node_id)
            if shard_names is None:
                rewritten_node = copy.deepcopy(base_node)
                rewritten_nodes.append(rewritten_node)
                flowmesh_to_raw[node_id] = node_id
                continue

            partitions = partitions_by_raw[node_id]
            if len(partitions) != len(shard_names):
                raise ValueError(
                    f"Internal shard rewrite mismatch for raw node '{node_id}'"
                )
            for shard_idx, shard_name in enumerate(shard_names):
                rewritten_node = copy.deepcopy(base_node)
                rewritten_node["name"] = shard_name
                rewritten_spec = rewritten_node.get("spec")
                if not isinstance(rewritten_spec, dict):
                    raise ValueError(
                        f"FlowMesh node '{node_id}' has invalid spec type for rewrite"
                    )
                spec_data = rewritten_spec.get("data")
                if not isinstance(spec_data, dict):
                    raise ValueError(
                        f"Unsupported shard boundary for raw node '{node_id}':"
                        " spec.data is missing or invalid"
                    )
                start, end = partitions[shard_idx]
                rewritten_spec["data"] = self._slice_static_lists(
                    value=spec_data,
                    raw_node_id=node_id,
                    start=start,
                    end=end,
                    total=partitions[-1][1] if partitions else 0,
                )
                self._rewrite_shard_node_references(
                    raw_node_id=node_id,
                    shard_idx=shard_idx,
                    spec=rewritten_spec,
                    shard_nodes_by_raw=shard_nodes_by_raw,
                )
                self._inline_unsharded_dependency_slices(
                    raw_node_id=node_id,
                    start=start,
                    end=end,
                    total=partitions[-1][1] if partitions else 0,
                    spec=rewritten_spec,
                    shard_nodes_by_raw=shard_nodes_by_raw,
                    node_map=node_map,
                )
                deps = rewritten_node.get("dependsOn")
                rewritten_node["dependsOn"] = self._rewrite_shard_dependencies(
                    deps=deps,
                    shard_idx=shard_idx,
                    shard_nodes_by_raw=shard_nodes_by_raw,
                )
                rewritten_nodes.append(rewritten_node)
                flowmesh_to_raw[shard_name] = node_id
            if node_id in merge_required_nodes:
                merge_node = self._build_merge_node(
                    raw_node_id=node_id,
                    base_node=base_node,
                    shard_names=shard_names,
                    partitions=partitions,
                )
                rewritten_nodes.append(merge_node)
                flowmesh_to_raw[node_id] = node_id

        rewritten_assignment = self._rewrite_worker_assignment_for_shards(
            original_assignment=rewritten_input_assignment,
            shard_nodes_by_raw=shard_nodes_by_raw,
            node_candidates=effective_node_candidates,
            merge_nodes=merge_required_nodes,
        )
        rewritten_names = [str(node.get("name")) for node in rewritten_nodes]
        assigned_names = {
            node_id
            for nodes_per_worker in rewritten_assignment.values()
            for node_id in nodes_per_worker
        }
        missing_assignment = sorted(set(rewritten_names) - assigned_names)
        if missing_assignment:
            raise ValueError(
                "Rewritten schedule does not cover all flowmesh nodes. "
                f"Missing: {missing_assignment}"
            )
        return ShardRewriteResult(
            nodes=rewritten_nodes,
            worker_assignment=rewritten_assignment,
            flowmesh_to_raw=flowmesh_to_raw,
        )

    @staticmethod
    def _classify_shard_boundary_error(message: str) -> str:
        message_lower = message.lower()
        if any(
            token in message_lower
            for token in (
                "cardinality",
                "mismatch",
                "inconsistent static list lengths",
                "do not align",
            )
        ):
            return "cardinality mismatch"
        if "unresolved sharded deps" in message_lower:
            return "unresolved sharded deps"
        return "non-partitionable boundary"

    @staticmethod
    def _build_shard_rewrite_topological_order(
        *,
        node_order: Sequence[str],
        node_map: Mapping[str, Mapping[str, Any]],
    ) -> list[str]:
        node_set = set(node_order)
        in_degree: dict[str, int] = {node_id: 0 for node_id in node_order}
        children: dict[str, list[str]] = {node_id: [] for node_id in node_order}
        for node_id in node_order:
            deps = node_map.get(node_id, {}).get("dependsOn")
            if not isinstance(deps, list):
                continue
            for dep in deps:
                if not isinstance(dep, str) or dep not in node_set:
                    continue
                if dep == node_id:
                    raise ValueError(
                        "Invalid shard rewrite dependency graph: "
                        f"self dependency on '{node_id}'"
                    )
                in_degree[node_id] += 1
                children[dep].append(node_id)

        order_index = {node_id: idx for idx, node_id in enumerate(node_order)}
        heap: list[tuple[int, str]] = []
        for node_id in node_order:
            if in_degree[node_id] == 0:
                heapq.heappush(heap, (order_index[node_id], node_id))

        ordered_nodes: list[str] = []
        while heap:
            _, node_id = heapq.heappop(heap)
            ordered_nodes.append(node_id)
            for child in children[node_id]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    heapq.heappush(heap, (order_index[child], child))

        if len(ordered_nodes) != len(node_order):
            raise ValueError(
                "Invalid shard rewrite dependency graph: cycle detected in flowmesh"
                " nodes"
            )
        return ordered_nodes

    @staticmethod
    def _pin_non_explicit_unsharded_nodes(
        *,
        explicit_shard_intent_nodes: set[str],
        shard_nodes_by_raw: Mapping[str, Sequence[str]],
        node_candidates: dict[str, list[str]],
        baseline_node_candidates: Mapping[str, Sequence[str]],
    ) -> None:
        for node_id, workers in list(node_candidates.items()):
            if node_id in explicit_shard_intent_nodes or node_id in shard_nodes_by_raw:
                continue
            baseline_workers = list(baseline_node_candidates.get(node_id, ()))
            if baseline_workers:
                node_candidates[node_id] = [baseline_workers[0]]
                continue
            if workers:
                node_candidates[node_id] = [workers[0]]

    @staticmethod
    def _prune_worker_assignment_to_candidates(
        *,
        worker_assignment: Mapping[str, Sequence[str]],
        node_candidates: Mapping[str, Sequence[str]],
    ) -> dict[str, list[str]]:
        allowed_workers = {
            node_id: set(workers) for node_id, workers in node_candidates.items()
        }
        pruned_assignment: dict[str, list[str]] = {}
        for worker, node_ids in worker_assignment.items():
            deduped_nodes: list[str] = []
            seen: set[str] = set()
            for node_id in node_ids:
                node_allowed_workers = allowed_workers.get(node_id)
                if (
                    node_allowed_workers is not None
                    and worker not in node_allowed_workers
                ):
                    continue
                if node_id in seen:
                    continue
                seen.add(node_id)
                deduped_nodes.append(node_id)
            pruned_assignment[worker] = deduped_nodes
        return pruned_assignment

    @staticmethod
    def _compute_merge_required_nodes(
        *,
        node_order: Sequence[str],
        node_map: Mapping[str, Mapping[str, Any]],
        shard_nodes_by_raw: Mapping[str, Sequence[str]],
        output_node_names: set[str],
    ) -> set[str]:
        sharded_nodes = set(shard_nodes_by_raw.keys())
        required = {
            node_id for node_id in output_node_names if node_id in sharded_nodes
        }
        for consumer_id in node_order:
            if consumer_id in sharded_nodes:
                continue
            deps = node_map.get(consumer_id, {}).get("dependsOn")
            if not isinstance(deps, list):
                continue
            for dep in deps:
                if isinstance(dep, str) and dep in sharded_nodes:
                    required.add(dep)
        return required

    def _build_merge_node(
        self,
        *,
        raw_node_id: str,
        base_node: Mapping[str, Any],
        shard_names: Sequence[str],
        partitions: Sequence[tuple[int, int]],
    ) -> dict[str, Any]:
        if len(shard_names) != len(partitions):
            raise ValueError(
                f"Internal shard merge mismatch for raw node '{raw_node_id}'"
            )
        base_spec = base_node.get("spec")
        if not isinstance(base_spec, Mapping):
            raise ValueError(
                f"Unsupported shard boundary for raw node '{raw_node_id}': node spec is"
                " not a mapping"
            )

        merge_items: list[dict[str, str]] = []
        for shard_idx, shard_name in enumerate(shard_names):
            start, end = partitions[shard_idx]
            if end <= start:
                continue
            merge_items.append({"node": shard_name, "path": "items.output"})

        merge_spec: dict[str, Any] = {
            "taskType": "echo",
            "data": {"type": "list", "items": merge_items},
        }
        return {
            "name": raw_node_id,
            "dependsOn": list(shard_names),
            "spec": merge_spec,
        }

    @staticmethod
    def _collect_node_worker_candidates(
        *,
        worker_assignment: Mapping[str, Sequence[str]],
        node_names: Sequence[str],
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        node_set = set(node_names)
        normalized_assignment = {
            worker: list(node_ids) for worker, node_ids in worker_assignment.items()
        }
        candidates: dict[str, list[str]] = {}
        seen_nodes: set[str] = set()
        for worker, node_ids in normalized_assignment.items():
            local_seen: set[str] = set()
            for node_id in node_ids:
                if node_id in local_seen:
                    raise ValueError(
                        "Worker assignment contains duplicate node IDs for worker "
                        f"'{worker}': {node_id}"
                    )
                local_seen.add(node_id)
                if node_id not in node_set:
                    raise ValueError(
                        f"Worker assignment references unknown node '{node_id}'"
                    )
                seen_nodes.add(node_id)
                worker_list = candidates.setdefault(node_id, [])
                if worker not in worker_list:
                    worker_list.append(worker)
        if seen_nodes != node_set:
            missing = sorted(node_set - seen_nodes)
            extra = sorted(seen_nodes - node_set)
            raise ValueError(
                "Worker assignment must cover all runtime nodes exactly at least once. "
                f"Missing: {missing}, extra: {extra}"
            )
        return candidates, normalized_assignment

    def _infer_shard_input_size(
        self,
        *,
        raw_node_id: str,
        spec: Mapping[str, Any],
        shard_count: int,
        dependency_ids: Sequence[str],
        partitions_by_raw: Mapping[str, Sequence[tuple[int, int]]],
        node_map: Mapping[str, Mapping[str, Any]],
    ) -> int:
        data_spec = spec.get("data")
        if not isinstance(data_spec, Mapping):
            raise ValueError(
                f"Unsupported shard boundary for raw node '{raw_node_id}': missing"
                " spec.data"
            )
        local_total = self._resolve_static_partition_total(
            raw_node_id=raw_node_id,
            data_spec=data_spec,
            context_node_id=raw_node_id,
        )
        if local_total is not None:
            return local_total
        dep_totals: list[int] = []
        for dep in dependency_ids:
            dep_partitions = partitions_by_raw.get(dep)
            if dep_partitions is None:
                continue
            if len(dep_partitions) != shard_count:
                raise ValueError(
                    "Unsupported shard boundary for raw node "
                    f"'{raw_node_id}': dependency '{dep}' shard count mismatch"
                )
            dep_totals.append(sum(end - start for start, end in dep_partitions))
        if dep_totals:
            first = dep_totals[0]
            if any(total != first for total in dep_totals):
                raise ValueError(
                    "Unsupported shard boundary for raw node "
                    f"'{raw_node_id}': dependency partition sizes do not align"
                )
            return max(1, first)

        # When direct dependency partitions are not available yet, infer from
        # ancestor nodes that carry static input lists.
        ancestor_totals = self._collect_ancestor_partition_totals(
            raw_node_id=raw_node_id,
            dependency_ids=dependency_ids,
            node_map=node_map,
        )
        if ancestor_totals:
            total = max(ancestor_totals)
            if any(value not in {1, total} for value in ancestor_totals):
                raise ValueError(
                    "Unsupported shard boundary for raw node "
                    f"'{raw_node_id}': ancestor static partition sizes do not align"
                )
            return max(1, total)
        raise ValueError(
            "Unsupported shard boundary for raw node "
            f"'{raw_node_id}': cannot infer static partition size"
        )

    def _collect_ancestor_partition_totals(
        self,
        *,
        raw_node_id: str,
        dependency_ids: Sequence[str],
        node_map: Mapping[str, Mapping[str, Any]],
    ) -> list[int]:
        totals: list[int] = []
        visited: set[str] = set()
        stack: list[str] = [dep for dep in dependency_ids if isinstance(dep, str)]
        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            node = node_map.get(node_id)
            if not isinstance(node, Mapping):
                continue
            spec = node.get("spec")
            if isinstance(spec, Mapping):
                data_spec = spec.get("data")
                if isinstance(data_spec, Mapping):
                    total = self._resolve_static_partition_total(
                        raw_node_id=raw_node_id,
                        data_spec=data_spec,
                        context_node_id=node_id,
                    )
                    if total is not None:
                        totals.append(total)
            deps = node.get("dependsOn")
            if isinstance(deps, list):
                for dep in deps:
                    if isinstance(dep, str) and dep not in visited:
                        stack.append(dep)
        return totals

    def _infer_unsharded_dependency_total(
        self,
        *,
        raw_node_id: str,
        dependency_ids: Sequence[str],
        node_map: Mapping[str, Mapping[str, Any]],
    ) -> int | None:
        direct_totals: list[int] = []
        for dep in dependency_ids:
            dep_node = node_map.get(dep)
            if not isinstance(dep_node, Mapping):
                continue
            dep_spec = dep_node.get("spec")
            if not isinstance(dep_spec, Mapping):
                continue
            dep_data = dep_spec.get("data")
            if not isinstance(dep_data, Mapping):
                continue
            dep_total = self._resolve_static_partition_total(
                raw_node_id=raw_node_id,
                data_spec=dep_data,
                context_node_id=dep,
            )
            if dep_total is not None:
                direct_totals.append(dep_total)
        if direct_totals:
            max_total = max(direct_totals)
            if any(total not in {1, max_total} for total in direct_totals):
                raise ValueError(
                    "Unsupported shard boundary for raw node "
                    f"'{raw_node_id}': dependency static partition sizes do not align"
                )
            return max_total
        ancestor_totals = self._collect_ancestor_partition_totals(
            raw_node_id=raw_node_id,
            dependency_ids=dependency_ids,
            node_map=node_map,
        )
        if not ancestor_totals:
            return None
        max_total = max(ancestor_totals)
        if any(total not in {1, max_total} for total in ancestor_totals):
            raise ValueError(
                "Unsupported shard boundary for raw node "
                f"'{raw_node_id}': ancestor static partition sizes do not align"
            )
        return max_total

    def _resolve_static_partition_total(
        self,
        *,
        raw_node_id: str,
        data_spec: Mapping[str, Any],
        context_node_id: str,
    ) -> int | None:
        lengths = self._collect_static_list_lengths(data_spec)
        if not lengths:
            return None
        total = max(lengths)
        for length in lengths:
            if length not in {1, total}:
                raise ValueError(
                    f"Unsupported shard boundary for raw node '{raw_node_id}': node"
                    f" '{context_node_id}' has inconsistent static list lengths"
                    f" {sorted(set(lengths))}"
                )
        return max(1, total)

    @staticmethod
    def _collect_static_list_lengths(value: Any) -> list[int]:
        lengths: list[int] = []
        if isinstance(value, Mapping):
            value_type = value.get("type")
            items = value.get("items")
            if value_type == "list" and isinstance(items, list):
                lengths.append(len(items))
            for item in value.values():
                lengths.extend(
                    FlowmeshRuntimeManager._collect_static_list_lengths(item)
                )
            return lengths
        if isinstance(value, list):
            for item in value:
                lengths.extend(
                    FlowmeshRuntimeManager._collect_static_list_lengths(item)
                )
        return lengths

    def _inline_unsharded_dependency_slices(
        self,
        *,
        raw_node_id: str,
        start: int,
        end: int,
        total: int,
        spec: dict[str, Any],
        shard_nodes_by_raw: Mapping[str, Sequence[str]],
        node_map: Mapping[str, Mapping[str, Any]],
    ) -> None:
        dep_total_cache: dict[str, int | None] = {}

        def dependency_static_total(dep_id: str) -> int | None:
            if dep_id in dep_total_cache:
                return dep_total_cache[dep_id]
            dep_node = node_map.get(dep_id)
            if not isinstance(dep_node, Mapping):
                dep_total_cache[dep_id] = None
                return None
            dep_spec = dep_node.get("spec")
            if not isinstance(dep_spec, Mapping):
                dep_total_cache[dep_id] = None
                return None
            dep_data = dep_spec.get("data")
            if not isinstance(dep_data, Mapping):
                dep_total_cache[dep_id] = None
                return None
            dep_total = self._resolve_static_partition_total(
                raw_node_id=raw_node_id,
                data_spec=dep_data,
                context_node_id=dep_id,
            )
            dep_total_cache[dep_id] = dep_total
            return dep_total

        def rewrite_inline_slice(
            column: dict[str, Any], dep_id: str, path: str
        ) -> None:
            dep_total = dependency_static_total(dep_id)
            if dep_total is not None and dep_total not in {1, total}:
                raise ValueError(
                    f"Unsupported shard boundary for raw node '{raw_node_id}':"
                    f" dependency '{dep_id}' static cardinality {dep_total} does not"
                    f" match shard source cardinality {total}"
                )
            suffix = self._extract_unindexed_items_suffix(path)
            items: list[dict[str, str]] = []
            for source_idx in range(start, end):
                resolved_idx = 0 if dep_total == 1 else source_idx
                items.append(
                    {
                        "node": dep_id,
                        "path": f"items[{resolved_idx}]{suffix}",
                    }
                )
            column.pop("node", None)
            column.pop("path", None)
            column["data"] = {"type": "list", "items": items}

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                ref_node = node.get("node")
                if (
                    isinstance(ref_node, str)
                    and ref_node not in shard_nodes_by_raw
                    and ref_node in node_map
                ):
                    ref_path = node.get("path")
                    if ref_path is None:
                        dep_total = dependency_static_total(ref_node)
                        if dep_total != 1:
                            raise ValueError(
                                "Unsupported shard boundary for raw node"
                                f" '{raw_node_id}': dependency '{ref_node}' missing"
                                " path for shard boundary rewrite"
                            )
                    elif isinstance(ref_path, str) and self._is_unindexed_items_path(
                        ref_path
                    ):
                        rewrite_inline_slice(node, ref_node, ref_path)
                for child in node.values():
                    visit(child)
                return
            if isinstance(node, list):
                for child in node:
                    visit(child)

        visit(spec)

    @staticmethod
    def _extract_unindexed_items_suffix(path: str) -> str:
        normalized = path.strip()
        prefix = "items"
        if not normalized.startswith(prefix):
            raise ValueError(
                "Invalid shard boundary rewrite path: "
                f"expected '{prefix}*', got '{path}'"
            )
        return normalized[len(prefix) :]

    @staticmethod
    def _is_unindexed_items_path(path: str) -> bool:
        normalized = path.strip()
        return normalized.startswith("items") and not normalized.startswith("items[")

    @staticmethod
    def _build_index_partitions(total: int, shard_count: int) -> list[tuple[int, int]]:
        safe_total = max(1, int(total))
        safe_count = max(1, int(shard_count))
        base = safe_total // safe_count
        remainder = safe_total % safe_count
        partitions: list[tuple[int, int]] = []
        start = 0
        for idx in range(safe_count):
            size = base + (1 if idx < remainder else 0)
            end = start + size
            partitions.append((start, end))
            start = end
        return partitions

    def _slice_static_lists(
        self,
        *,
        value: Any,
        raw_node_id: str,
        start: int,
        end: int,
        total: int,
    ) -> Any:
        if isinstance(value, dict):
            cloned = copy.deepcopy(value)
            if cloned.get("type") == "list" and isinstance(cloned.get("items"), list):
                items = list(cloned.get("items", []))
                if len(items) == total:
                    cloned["items"] = items[start:end]
                elif len(items) != 1:
                    raise ValueError(
                        f"Unsupported shard boundary for raw node '{raw_node_id}':"
                        f" static list length {len(items)} cannot be partitioned to"
                        f" total {total}"
                    )
            for key, item in list(cloned.items()):
                if key == "items" and isinstance(cloned.get("type"), str):
                    continue
                cloned[key] = self._slice_static_lists(
                    value=item,
                    raw_node_id=raw_node_id,
                    start=start,
                    end=end,
                    total=total,
                )
            return cloned
        if isinstance(value, list):
            return [
                self._slice_static_lists(
                    value=item,
                    raw_node_id=raw_node_id,
                    start=start,
                    end=end,
                    total=total,
                )
                for item in value
            ]
        return value

    def _rewrite_shard_node_references(
        self,
        *,
        raw_node_id: str,
        shard_idx: int,
        spec: dict[str, Any],
        shard_nodes_by_raw: Mapping[str, Sequence[str]],
    ) -> None:
        def visit(node: Any) -> None:
            if isinstance(node, dict):
                ref_node = node.get("node")
                if isinstance(ref_node, str):
                    if ref_node in shard_nodes_by_raw:
                        shards = shard_nodes_by_raw[ref_node]
                        if shard_idx >= len(shards):
                            raise ValueError(
                                "Unsupported shard boundary for raw node"
                                f" '{raw_node_id}': shard index mismatch for dependency"
                                f" '{ref_node}'"
                            )
                        node["node"] = shards[shard_idx]
                for item in node.values():
                    visit(item)
            elif isinstance(node, list):
                for item in node:
                    visit(item)

        visit(spec)

    @staticmethod
    def _rewrite_shard_dependencies(
        *,
        deps: Any,
        shard_idx: int,
        shard_nodes_by_raw: Mapping[str, Sequence[str]],
    ) -> list[str]:
        if deps is None:
            return []
        if not isinstance(deps, list):
            raise ValueError("Invalid shard boundary: 'dependsOn' must be a list")
        rewritten: list[str] = []
        for dep in deps:
            if not isinstance(dep, str):
                continue
            shard_nodes = shard_nodes_by_raw.get(dep)
            if shard_nodes is None:
                rewritten.append(dep)
                continue
            if shard_idx >= len(shard_nodes):
                raise ValueError(
                    f"Unsupported shard dependency rewrite for '{dep}': shard index out"
                    " of range"
                )
            rewritten.append(shard_nodes[shard_idx])
        return list(dict.fromkeys(rewritten))

    @staticmethod
    def _rewrite_worker_assignment_for_shards(
        *,
        original_assignment: Mapping[str, Sequence[str]],
        shard_nodes_by_raw: Mapping[str, Sequence[str]],
        node_candidates: Mapping[str, Sequence[str]],
        merge_nodes: set[str],
    ) -> dict[str, list[str]]:
        shard_index_by_worker: dict[str, dict[str, int]] = {}
        primary_worker_by_raw: dict[str, str] = {}
        for node_id, workers in node_candidates.items():
            if node_id not in shard_nodes_by_raw:
                continue
            if not workers:
                raise ValueError(
                    "Invalid shard rewrite assignment: missing worker candidates for "
                    f"node '{node_id}'"
                )
            expected_shard_nodes = shard_nodes_by_raw[node_id]
            if len(workers) != len(expected_shard_nodes):
                raise ValueError(
                    "Invalid shard rewrite assignment: shard worker count mismatch for "
                    f"node '{node_id}'"
                )
            primary_worker_by_raw[node_id] = workers[0]
            for idx, worker in enumerate(workers):
                shard_index_by_worker.setdefault(node_id, {})[worker] = idx

        merge_host_by_raw: dict[str, str] = {}
        worker_ids = list(original_assignment.keys())
        for node_id in merge_nodes:
            if node_id not in shard_nodes_by_raw:
                continue
            shard_workers = set(shard_index_by_worker.get(node_id, {}))
            preferred_workers = [
                worker for worker in worker_ids if worker not in shard_workers
            ]
            if preferred_workers:
                merge_host_by_raw[node_id] = preferred_workers[0]
                continue
            primary_worker = primary_worker_by_raw.get(node_id)
            if primary_worker is None:
                raise ValueError(
                    "Invalid shard rewrite assignment: missing primary worker for "
                    f"merge node '{node_id}'"
                )
            merge_host_by_raw[node_id] = primary_worker

        rewritten: dict[str, list[str]] = {}
        for worker, node_ids in original_assignment.items():
            out: list[str] = []
            seen: set[str] = set()
            for node_id in node_ids:
                shard_nodes = shard_nodes_by_raw.get(node_id)
                if shard_nodes is None:
                    mapped = node_id
                else:
                    shard_idx = shard_index_by_worker.get(node_id, {}).get(worker)
                    if shard_idx is None:
                        if node_id not in merge_nodes:
                            continue
                        merge_host = merge_host_by_raw.get(node_id)
                        if merge_host != worker:
                            continue
                        mapped = node_id
                    else:
                        mapped = shard_nodes[shard_idx]
                if mapped in seen:
                    continue
                seen.add(mapped)
                out.append(mapped)
                if (
                    node_id in merge_nodes
                    and merge_host_by_raw.get(node_id) == worker
                    and node_id not in seen
                ):
                    seen.add(node_id)
                    out.append(node_id)
            rewritten[worker] = out

        for node_id, worker in merge_host_by_raw.items():
            out = rewritten.setdefault(worker, [])
            if node_id not in out:
                out.append(node_id)

        return rewritten

    def release_executions(self, execution_ids: set[str]) -> None:
        """Drop ``_batch_workflow_id`` entries for the given executions.

        Called once a request's ``trace_ids`` have been persisted to
        ``JobStorage``; the in-memory tuple→workflow id map can then go.

        Snapshots the keys via ``.copy()`` before popping — entries can be
        added on the runtime async thread while we run on the FastAPI thread.
        """
        for key in self._batch_workflow_id.copy():
            if key[0] in execution_ids:
                self._batch_workflow_id.pop(key, None)

    async def cancel_request(self, request_id: str) -> None:
        """Cancel all FM workflows for a request. Uses the scheduler
        credential so shutdown cancels (no bearer in scope) still authenticate."""
        async with self._task_status_lock:
            self._cancelled_requests.add(request_id)
            workflow_ids = [
                wf
                for (rid, _), wf in self._batch_workflow_id.items()
                if rid == request_id
            ]
        if not workflow_ids:
            self.logger.info(
                "Cancellation requested for %s before workflow submission", request_id
            )
            return
        fm = flowmesh_for_server()
        for workflow_id in workflow_ids:
            try:
                await fm.workflows.cancel(workflow_id)
                self.logger.info(f"Successfully cancelled workflow {workflow_id}")
            except APIError as e:
                self.logger.warning(f"Failed to cancel workflow {workflow_id}: {e}")
