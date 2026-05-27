import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from support.runtime_graphs import build_dummy_runtime_graph

from lumilake_server.runtime.job_manager.base import BatchSelection, WorkflowItem
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.protocol import LumilakeRequestConfig
from lumilake_server.runtime.request import RequestHandler
from lumilake_server.runtime.runtime_ops import RuntimeOp
from lumilake_server.runtime.server import (
    LumilakeServer,
    LumilakeServerConfig,
    RequestState,
)
from lumilake_server.schemas.progress import JobProgress
from lumilake_server.utils.job_storage import get_job_storage


class FakeHandler:
    def __init__(self) -> None:
        self.results: list[Any] = []

    async def put_result(self, result: Any) -> None:
        self.results.append(result)


_RUNTIME_RESULT_DIRS: list[tempfile.TemporaryDirectory[str]] = []


def cleanup_runtime_result_dirs() -> None:
    while _RUNTIME_RESULT_DIRS:
        _RUNTIME_RESULT_DIRS.pop().cleanup()


class RecordingRuntimeManager:
    def __init__(
        self,
        *,
        cancelled: set[str] | None = None,
        status_by_request: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.cancelled = set() if cancelled is None else set(cancelled)
        self.status_by_request = {} if status_by_request is None else status_by_request
        self._dispatch_tokens: dict[str, str | None] = {}
        self.cancel_calls: list[str] = []
        self.mark_calls: list[tuple[str, str, str]] = []
        self._result_dir = tempfile.TemporaryDirectory(prefix="lumilake-runtime-test-")
        _RUNTIME_RESULT_DIRS.append(self._result_dir)

    async def get_workers(self) -> list[str]:
        return ["worker-1"]

    async def get_worker_profile(self, worker_id: str) -> dict[str, Any]:
        return {"gpu": {"count": 0}}

    def count_runtime_nodes(self, graphs: dict[str, Any]) -> int:
        return sum(graph.node_count for graph in graphs.values())

    def result_dir(self, request_info: Any) -> Path:
        return (
            Path(self._result_dir.name)
            / request_info.request_id
            / request_info.batch_id
        )

    def save_runtime_artifact(
        self,
        request_info: Any,
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

    def mark_batch_pending(
        self,
        request_id: str,
        batch_id: str,
        total_nodes: int,
        output_nodes: int,
    ) -> None:
        self.mark_calls.append(("pending", request_id, batch_id))

    def mark_batch_running(self, request_id: str, batch_id: str) -> None:
        self.mark_calls.append(("running", request_id, batch_id))

    def mark_batch_completed(self, request_id: str, batch_id: str) -> None:
        self.mark_calls.append(("completed", request_id, batch_id))

    def mark_batch_failed(self, request_id: str, batch_id: str) -> None:
        self.mark_calls.append(("failed", request_id, batch_id))

    async def process_request(
        self,
        request_info: Any,
        schedule: Schedule,
        worker_ids: list[str],
        data_profile_results: dict[str, list[dict[str, Any]]] | None,
    ) -> dict[str, Any]:
        return {"flat_outputs": {}, "chat_histories": {}, "task_node_map": {}}

    async def get_request_status(self, request_id: str) -> dict[str, Any]:
        return self.status_by_request.get(request_id, {"error": "Request ID not found"})

    async def cancel_request(self, request_id: str) -> None:
        self.cancel_calls.append(request_id)
        self.cancelled.add(request_id)

    def get_task_node_map(self, request_id: str, batch_id: str) -> dict[str, str]:
        return {}

    async def is_request_cancelled(self, request_id: str) -> bool:
        return request_id in self.cancelled

    def set_dispatch_token(self, request_id: str, token: str | None) -> None:
        self._dispatch_tokens[request_id] = token

    def get_dispatch_token(self, request_id: str) -> str | None:
        return self._dispatch_tokens.get(request_id)

    def clear_dispatch_token(self, request_id: str) -> None:
        self._dispatch_tokens.pop(request_id, None)


class ArtifactRuntimeManager(RecordingRuntimeManager):
    async def process_request(
        self,
        request_info: Any,
        schedule: Schedule,
        worker_ids: list[str],
        data_profile_results: dict[str, list[dict[str, Any]]] | None,
    ) -> dict[str, Any]:
        flat_outputs: dict[str, list[str]] = {}
        histories: dict[str, list[list[dict[str, str]]]] = {}
        storage = get_job_storage()
        for node_id in request_info.output_node_map:
            filename = f"{node_id}.bin"
            storage.save_artifact(
                request_info.request_id,
                filename,
                b"artifact-data",
                "application/octet-stream",
            )
            uri = f"memory://{request_info.request_id}/artifacts/{filename}"
            flat_outputs[node_id] = [uri]
            histories[node_id] = [[{"role": "assistant", "content": uri}]]
        return {
            "flat_outputs": flat_outputs,
            "chat_histories": histories,
            "task_node_map": {},
        }


def make_workflow(
    *,
    workflow_id: str,
    request_id: str,
    graph_name: str,
    public_graph_name: str,
    template_hash: str | None = None,
    dsl_inputs: dict[str, list[str]] | None = None,
    varying_input_keys: tuple[str, ...] = (),
    slice_index: int = 0,
    slice_start: int = 0,
    slice_length: int = 1,
    total_length: int = 1,
) -> WorkflowItem:
    runtime_graph = build_dummy_runtime_graph(graph_name)
    compiled_graph = SimpleNamespace(
        graph=object(),
        inputs={"input": [request_id]} if dsl_inputs is None else dsl_inputs,
    )
    return WorkflowItem(
        workflow_id=workflow_id,
        request_id=request_id,
        graph_name=graph_name,
        public_graph_name=public_graph_name,
        slice_index=slice_index,
        slice_start=slice_start,
        slice_length=slice_length,
        total_length=total_length,
        template_hash=(
            f"template-{graph_name}" if template_hash is None else template_hash
        ),
        varying_input_keys=varying_input_keys,
        runtime_graph=runtime_graph,
        data_profile_graph=runtime_graph,
        dsl_graph=cast(Any, compiled_graph),
        config=LumilakeRequestConfig(user_id=request_id, principal_id=request_id),
        enqueued_at=time.time(),
    )


def make_workflow_slices_from_inputs(
    *,
    request_id: str,
    public_graph_name: str,
    entities: list[str],
    template_hash: str = "template-shared",
) -> list[WorkflowItem]:
    total = len(entities)
    return [
        make_workflow(
            workflow_id=f"{request_id}-{idx}",
            request_id=request_id,
            graph_name=f"{request_id}-g{idx}",
            public_graph_name=public_graph_name,
            template_hash=template_hash,
            dsl_inputs={"entity": [entity]},
            varying_input_keys=("entity",),
            slice_index=idx,
            slice_start=idx,
            slice_length=1,
            total_length=total,
        )
        for idx, entity in enumerate(entities)
    ]


def make_batch(workflows: list[WorkflowItem]) -> BatchSelection:
    return BatchSelection(
        workflows=workflows,
        runtime_graphs={item.workflow_id: item.runtime_graph for item in workflows},
        data_profile_graphs={
            item.workflow_id: item.data_profile_graph for item in workflows
        },
        config=LumilakeRequestConfig(user_id="batch-user", principal_id="batch-user"),
    )


def make_server() -> LumilakeServer:
    LumilakeServer._instance = None
    return LumilakeServer(
        config=LumilakeServerConfig(
            is_local=True,
            runtime_url="http://localhost:18080",
            batch_size=4,
            cpu_worker_group_size=1,
            gpu_worker_group_size=0,
        ),
    )


def attach_request_states(
    server: LumilakeServer, workflows: list[WorkflowItem]
) -> dict[str, FakeHandler]:
    handlers: dict[str, FakeHandler] = {}
    for workflow in workflows:
        handler = handlers.get(workflow.request_id)
        if handler is None:
            handler = FakeHandler()
            handlers[workflow.request_id] = handler
            server._requests[workflow.request_id] = RequestState(
                handler=cast(RequestHandler, handler),
                config=LumilakeRequestConfig(
                    user_id=workflow.request_id, principal_id=workflow.request_id
                ),
                pending_workflows=set(),
                workflow_lengths={workflow.public_graph_name: 1},
                pending_runtime_nodes_raw=0,
                ready=True,
            )
            server._progress[workflow.request_id] = JobProgress()
        state = server._requests[workflow.request_id]
        state.pending_workflows.add(workflow.workflow_id)
        state.pending_runtime_nodes_raw += workflow.runtime_graph.node_count
    return handlers


def make_runtime_op(node_id: str) -> RuntimeOp:
    return RuntimeOp(
        node_id=node_id,
        task_type="inference",
        backend="dummy",
        model="dummy-model",
        data_spec={},
        model_spec={},
        inference_spec={},
    )
