import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from lumilake import envs
from support.runtime_server import (
    ArtifactRuntimeManager,
    RecordingRuntimeManager,
    attach_request_states,
    make_batch,
    make_runtime_op,
    make_workflow,
    make_workflow_slices_from_inputs,
)

import lumilake_server.runtime.runtime_manager.flowmesh as fm_mod
from lumilake_server.hooks.security import runtime_token_var
from lumilake_server.runtime.job_manager.base import BatchSelection
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager
from lumilake_server.runtime.runtime_ops import RuntimeOp
from lumilake_server.utils.job_storage import get_job_storage


@pytest.mark.asyncio
async def test_run_batch_uses_execution_request_id_for_multi_request_batch(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = server_factory()
    runtime_manager = RecordingRuntimeManager()
    server.runtime_manager = cast(Any, runtime_manager)

    workflows = [
        make_workflow(
            workflow_id="wf-a",
            request_id="req-a",
            graph_name="ga",
            public_graph_name="shared",
        ),
        make_workflow(
            workflow_id="wf-b",
            request_id="req-b",
            graph_name="gb",
            public_graph_name="shared",
        ),
    ]
    handlers = attach_request_states(server, workflows)
    batch = make_batch(workflows)
    seen: dict[str, Any] = {}

    async def _fake_process_batch(
        selected_batch: BatchSelection,
        batch_id: str,
        selected_workers: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        *,
        execution_request_id: str,
        member_request_ids: set[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        seen["execution_request_id"] = execution_request_id
        seen["member_request_ids"] = set(member_request_ids)
        outputs = {
            item.workflow_id: {"output": [f"value-{item.request_id}"]}
            for item in selected_batch.workflows
        }
        return outputs, {}

    monkeypatch.setattr(server, "_process_batch", _fake_process_batch)
    await server._run_batch(["worker-1"], batch)

    execution_request_id = cast(str, seen["execution_request_id"])
    assert execution_request_id.startswith("exec-")
    assert seen["member_request_ids"] == {"req-a", "req-b"}
    assert any(
        call[0] == "completed" and call[1] == execution_request_id
        for call in runtime_manager.mark_calls
    )
    assert execution_request_id not in server._execution_contexts
    assert "req-a" not in server._request_execution_ids
    assert "req-b" not in server._request_execution_ids
    assert handlers["req-a"].results[0].outputs["shared"]["output"] == ["value-req-a"]
    assert handlers["req-b"].results[0].outputs["shared"]["output"] == ["value-req-b"]


@pytest.mark.asyncio
async def test_run_batch_dispatch_token_none_when_all_unset(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = server_factory()
    runtime_manager = RecordingRuntimeManager()
    server.runtime_manager = cast(Any, runtime_manager)

    workflows = [
        make_workflow(
            workflow_id="wf-a",
            request_id="req-a",
            graph_name="ga",
            public_graph_name="shared",
        ),
    ]
    attach_request_states(server, workflows)
    batch = make_batch(workflows)
    seen_tokens: list[str | None] = []

    async def _fake_process_batch(
        selected_batch: BatchSelection,
        batch_id: str,
        selected_workers: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        *,
        execution_request_id: str,
        member_request_ids: set[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        seen_tokens.append(runtime_token_var.get())
        return {}, {}

    monkeypatch.setattr(server, "_process_batch", _fake_process_batch)
    await server._run_batch(["worker-1"], batch)

    assert seen_tokens == [None]


@pytest.mark.asyncio
async def test_run_batch_cancels_subset_and_continues_other_requests(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = server_factory()
    runtime_manager = RecordingRuntimeManager(cancelled={"req-a"})
    server.runtime_manager = cast(Any, runtime_manager)

    workflows = [
        make_workflow(
            workflow_id="wf-a",
            request_id="req-a",
            graph_name="ga",
            public_graph_name="shared",
        ),
        make_workflow(
            workflow_id="wf-b",
            request_id="req-b",
            graph_name="gb",
            public_graph_name="shared",
        ),
    ]
    handlers = attach_request_states(server, workflows)
    batch = make_batch(workflows)
    seen: dict[str, Any] = {}

    async def _fake_process_batch(
        selected_batch: BatchSelection,
        batch_id: str,
        selected_workers: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        *,
        execution_request_id: str,
        member_request_ids: set[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        seen["member_request_ids"] = set(member_request_ids)
        assert {item.request_id for item in selected_batch.workflows} == {"req-b"}
        return {"wf-b": {"output": ["value-req-b"]}}, {}

    monkeypatch.setattr(server, "_process_batch", _fake_process_batch)
    await server._run_batch(["worker-1"], batch)

    assert seen["member_request_ids"] == {"req-b"}
    cancelled_errors = handlers["req-a"].results[0].error_info
    assert cancelled_errors is not None
    assert any("request_cancelled" in item for item in cancelled_errors)
    assert handlers["req-b"].results[0].outputs["shared"]["output"] == ["value-req-b"]
    assert all(not call.startswith("exec-") for call in runtime_manager.cancel_calls)
    assert not server._execution_contexts
    assert not server._request_execution_ids


@pytest.mark.asyncio
async def test_run_batch_dispatch_token_excludes_cancelled_requests(
    server_factory,
) -> None:
    server = server_factory()
    runtime_manager = RecordingRuntimeManager(cancelled={"req-a"})
    server.runtime_manager = cast(Any, runtime_manager)

    workflows = [
        make_workflow(
            workflow_id="wf-a",
            request_id="req-a",
            graph_name="ga",
            public_graph_name="shared",
            dispatch_token="tok-cancelled",
        ),
        make_workflow(
            workflow_id="wf-b",
            request_id="req-b",
            graph_name="gb",
            public_graph_name="shared",
            dispatch_token="tok-active",
        ),
    ]
    attach_request_states(server, workflows)
    batch = make_batch(workflows)
    seen_tokens: list[str | None] = []

    async def _fake_process_batch(
        selected_batch: BatchSelection,
        batch_id: str,
        selected_workers: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        *,
        execution_request_id: str,
        member_request_ids: set[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        seen_tokens.append(runtime_token_var.get())
        return (
            {item.workflow_id: {"output": ["x"]} for item in selected_batch.workflows},
            {},
        )

    setattr(server, "_process_batch", _fake_process_batch)
    await server._run_batch(["worker-1"], batch)

    assert seen_tokens == ["tok-active"]


@pytest.mark.asyncio
async def test_run_batch_cancels_execution_if_all_member_requests_cancelled(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = server_factory()
    runtime_manager = RecordingRuntimeManager(cancelled={"req-a", "req-b"})
    server.runtime_manager = cast(Any, runtime_manager)

    workflows = [
        make_workflow(
            workflow_id="wf-a",
            request_id="req-a",
            graph_name="ga",
            public_graph_name="shared",
        ),
        make_workflow(
            workflow_id="wf-b",
            request_id="req-b",
            graph_name="gb",
            public_graph_name="shared",
        ),
    ]
    handlers = attach_request_states(server, workflows)
    batch = make_batch(workflows)

    async def _fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "_process_batch should not run when all requests are cancelled"
        )

    monkeypatch.setattr(server, "_process_batch", _fail_if_called)
    await server._run_batch(["worker-1"], batch)

    assert any(call.startswith("exec-") for call in runtime_manager.cancel_calls)
    for request_id in ("req-a", "req-b"):
        errors = handlers[request_id].results[0].error_info
        assert errors is not None
        assert any("request_cancelled" in item for item in errors)
    assert not server._execution_contexts
    assert not server._request_execution_ids


@pytest.mark.asyncio
async def test_run_batch_failure_does_not_fetch_task_node_map(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = server_factory()

    class StrictRuntimeManager(RecordingRuntimeManager):
        def get_task_node_map(self, request_id: str, batch_id: str) -> dict[str, str]:
            raise AssertionError(
                "get_task_node_map should not be called on batch failure"
            )

    runtime_manager = StrictRuntimeManager()
    server.runtime_manager = cast(Any, runtime_manager)

    workflows = [
        make_workflow(
            workflow_id="wf-a",
            request_id="req-a",
            graph_name="ga",
            public_graph_name="shared",
        )
    ]
    handlers = attach_request_states(server, workflows)
    batch = make_batch(workflows)

    async def _fail_process_batch(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("batch processing failed")

    monkeypatch.setattr(server, "_process_batch", _fail_process_batch)
    await server._run_batch(["worker-1"], batch)

    errors = handlers["req-a"].results[0].error_info
    assert errors is not None
    assert any("batch processing failed" in str(item) for item in errors)


@pytest.mark.asyncio
async def test_run_batch_failure_redacts_credential_from_batch_error(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch-processing exception can carry a credential FlowMesh echoed
    back in a rejection body (e.g. the Authorization header from the task
    spec we submitted). The persisted `batch_error` must not leak it."""
    server = server_factory()
    server.runtime_manager = cast(Any, RecordingRuntimeManager())

    workflows = [
        make_workflow(
            workflow_id="wf-a",
            request_id="req-a",
            graph_name="ga",
            public_graph_name="shared",
        )
    ]
    handlers = attach_request_states(server, workflows)
    batch = make_batch(workflows)

    async def _fail_process_batch(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            'submit rejected, echoed spec: {"Authorization": "Bearer sk-live-secret"}'
        )

    monkeypatch.setattr(server, "_process_batch", _fail_process_batch)
    await server._run_batch(["worker-1"], batch)

    errors = handlers["req-a"].results[0].error_info
    assert errors is not None
    serialized_errors = json.dumps(errors)
    assert "sk-live-secret" not in serialized_errors
    assert "***REDACTED***" in serialized_errors


@pytest.mark.asyncio
async def test_run_batch_tracks_success_only_completed_inputs(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = server_factory()
    runtime_manager = RecordingRuntimeManager()
    server.runtime_manager = cast(Any, runtime_manager)

    workflows = [
        make_workflow(
            workflow_id="wf-a",
            request_id="req-a",
            graph_name="ga",
            public_graph_name="shared",
        ),
        make_workflow(
            workflow_id="wf-b",
            request_id="req-a",
            graph_name="gb",
            public_graph_name="shared",
        ),
    ]
    attach_request_states(server, workflows)
    server._requests["req-a"].total_input_items = 2
    batch = make_batch(workflows)

    async def _fake_process_batch(
        selected_batch: BatchSelection,
        batch_id: str,
        selected_workers: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        *,
        execution_request_id: str,
        member_request_ids: set[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return {"wf-a": {"output": ["value-a"]}}, {}

    monkeypatch.setattr(server, "_process_batch", _fake_process_batch)
    await server._run_batch(["worker-1"], batch)

    state = server._requests["req-a"]
    assert state.completed_input_items_success == 1
    assert state.successful_workflow_ids == {"wf-a"}


@pytest.mark.asyncio
async def test_run_batch_slice_failure_does_not_roll_back_other_slice_results(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Characterizes the documented cross-slice contract: a later slice's
    failure does not erase an earlier slice's already-merged results."""
    server = server_factory()
    runtime_manager = RecordingRuntimeManager()
    server.runtime_manager = cast(Any, runtime_manager)

    slice0, slice1 = make_workflow_slices_from_inputs(
        request_id="req-slices",
        public_graph_name="shared",
        entities=["NVDA", "AAPL"],
    )
    handlers = attach_request_states(server, [slice0, slice1])

    async def _succeed_process_batch(
        selected_batch: BatchSelection,
        batch_id: str,
        selected_workers: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        *,
        execution_request_id: str,
        member_request_ids: set[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return {slice0.workflow_id: {"result": ["nvda-reply"]}}, {}

    monkeypatch.setattr(server, "_process_batch", _succeed_process_batch)
    await server._run_batch(["worker-1"], make_batch([slice0]))

    async def _fail_process_batch(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("api row failed for AAPL")

    monkeypatch.setattr(server, "_process_batch", _fail_process_batch)
    await server._run_batch(["worker-1"], make_batch([slice1]))

    assert len(handlers["req-slices"].results) == 1
    response = handlers["req-slices"].results[0]
    assert response.outputs["shared"]["result"] == ["nvda-reply", ""]
    assert response.error_info is not None
    assert any("api row failed for AAPL" in str(item) for item in response.error_info)


@pytest.mark.asyncio
async def test_process_batch_uses_parent_workflow_grouping_and_relocates_artifacts(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = server_factory()
    runtime_manager = ArtifactRuntimeManager()
    server.runtime_manager = cast(Any, runtime_manager)

    workflows = [
        make_workflow(
            workflow_id="wf-a",
            request_id="req-a",
            graph_name="ga",
            public_graph_name="shared",
        ),
        make_workflow(
            workflow_id="wf-b",
            request_id="req-b",
            graph_name="gb",
            public_graph_name="shared",
        ),
    ]
    handlers = attach_request_states(server, workflows)
    batch = make_batch(workflows)

    monkeypatch.setattr(
        server,
        "_merge_group_compiled_graph",
        lambda items: cast(Any, SimpleNamespace(_coalesce_rewrite_hits={})),
    )
    prefixes: list[str] = []

    def _fake_build(
        compiled_graph: Any,
        task_type_override: str | None = None,
        node_prefix: str | None = None,
    ) -> RuntimeGraph:
        assert node_prefix is not None
        prefixes.append(node_prefix)
        suffix = "data_profile" if task_type_override == "data_profile" else "runtime"
        node_id = f"{node_prefix}__{suffix}"
        op = make_runtime_op(node_id)
        output_node_map = (
            {} if task_type_override == "data_profile" else {node_id: "output"}
        )
        return RuntimeGraph(
            nodes={node_id: op},
            node_order=[node_id],
            output_node_map=output_node_map,
        )

    server._runtime_builder.build = _fake_build  # type: ignore[method-assign]

    async def _fake_schedule(
        *,
        request_id: str,
        batch_id: str,
        optimizer_type: str,
        runtime_graph: RuntimeGraph,
        selected_workers: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        data_profile_results: dict[str, list[dict[str, Any]]],
        member_request_ids: set[str] | None = None,
        bearer_token: str | None = None,
    ) -> Schedule:
        return Schedule(
            worker_assignment={selected_workers[0]: list(runtime_graph.node_order)}
        )

    server._generate_schedule_in_subprocess = _fake_schedule  # type: ignore[method-assign]

    await server._run_batch(["worker-1"], batch)

    assert any(prefix.startswith("request::req-a::shared::") for prefix in prefixes)
    assert any(prefix.startswith("request::req-b::shared::") for prefix in prefixes)

    resp_a = handlers["req-a"].results[0]
    resp_b = handlers["req-b"].results[0]
    uri_a = resp_a.outputs["shared"]["output"][0]
    uri_b = resp_b.outputs["shared"]["output"][0]
    assert "memory://req-a/artifacts/" in uri_a
    assert "memory://req-b/artifacts/" in uri_b
    assert "exec-" not in uri_a
    assert "exec-" not in uri_b
    assert uri_a != uri_b

    history_a = resp_a.chat_histories["shared"]["output"][0][0]["content"]
    history_b = resp_b.chat_histories["shared"]["output"][0][0]["content"]
    assert "memory://req-a/artifacts/" in history_a
    assert "memory://req-b/artifacts/" in history_b


@pytest.mark.asyncio
async def test_process_batch_uses_server_data_profile_collection(
    server_factory,
    monkeypatch,
) -> None:
    server = server_factory()

    class StrictRuntimeManager(RecordingRuntimeManager):
        def __init__(self) -> None:
            super().__init__()
            self.last_data_profile_results: dict[str, list[dict[str, Any]]] | None = (
                None
            )

        async def profile_data(
            self, request_info: Any
        ) -> dict[str, list[dict[str, Any]]]:
            raise AssertionError("runtime_manager.profile_data must not be called")

        async def process_request(
            self,
            request_info: Any,
            schedule: Schedule,
            worker_ids: list[str],
            data_profile_results: dict[str, list[dict[str, Any]]] | None,
        ) -> dict[str, Any]:
            self.last_data_profile_results = data_profile_results
            flat_outputs = {
                node_id: [f"out-{node_id}"] for node_id in request_info.output_node_map
            }
            return {
                "flat_outputs": flat_outputs,
                "chat_histories": {},
                "task_node_map": {},
            }

    runtime_manager = StrictRuntimeManager()
    server.runtime_manager = cast(Any, runtime_manager)

    workflows = [
        make_workflow(
            workflow_id="wf-a",
            request_id="req-a",
            graph_name="ga",
            public_graph_name="shared",
        )
    ]
    handlers = attach_request_states(server, workflows)
    batch = make_batch(workflows)

    monkeypatch.setattr(
        server,
        "_merge_group_compiled_graph",
        lambda items: cast(Any, SimpleNamespace(_coalesce_rewrite_hits={})),
    )

    def _fake_build(
        compiled_graph: Any,
        task_type_override: str | None = None,
        node_prefix: str | None = None,
    ) -> RuntimeGraph:
        assert node_prefix is not None
        suffix = "data_profile" if task_type_override == "data_profile" else "runtime"
        node_id = f"{node_prefix}__{suffix}"
        op = make_runtime_op(node_id)
        output_node_map = (
            {} if task_type_override == "data_profile" else {node_id: "output"}
        )
        return RuntimeGraph(
            nodes={node_id: op},
            node_order=[node_id],
            output_node_map=output_node_map,
        )

    server._runtime_builder.build = _fake_build  # type: ignore[method-assign]

    async def _fake_schedule(
        *,
        request_id: str,
        batch_id: str,
        optimizer_type: str,
        runtime_graph: RuntimeGraph,
        selected_workers: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        data_profile_results: dict[str, list[dict[str, Any]]],
        member_request_ids: set[str] | None = None,
        bearer_token: str | None = None,
    ) -> Schedule:
        return Schedule(
            worker_assignment={selected_workers[0]: list(runtime_graph.node_order)}
        )

    server._generate_schedule_in_subprocess = _fake_schedule  # type: ignore[method-assign]

    expected_profile: dict[str, list[dict[str, Any]]] = {
        "data_profile::node::node_query": [{"cost_estimates": []}]
    }
    observed_sources: dict[str, list[Any]] = {}

    async def _fake_collect_data_profile(
        **kwargs: Any,
    ) -> dict[str, list[dict[str, Any]]]:
        observed_sources.update(kwargs.get("data_profile_sources", {}))
        return expected_profile

    monkeypatch.setattr(
        "lumilake_server.runtime.server.collect_data_profile",
        _fake_collect_data_profile,
    )

    await server._run_batch(["worker-1"], batch)

    assert runtime_manager.last_data_profile_results == expected_profile
    assert len(observed_sources) == 1
    only_group_sources = next(iter(observed_sources.values()))
    assert only_group_sources
    assert only_group_sources[0].org_id == "default"
    resp = handlers["req-a"].results[0]
    assert resp.outputs["shared"]["output"][0].startswith("out-")


@pytest.mark.asyncio
async def test_run_batch_skips_data_profile_when_disabled(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = server_factory()
    runtime_manager = RecordingRuntimeManager()
    server.runtime_manager = cast(Any, runtime_manager)

    workflows = [
        make_workflow(
            workflow_id="wf-a",
            request_id="req-a",
            graph_name="ga",
            public_graph_name="shared",
        )
    ]
    attach_request_states(server, workflows)
    batch = make_batch(workflows)

    monkeypatch.setattr(
        server,
        "_merge_group_compiled_graph",
        lambda items: cast(Any, SimpleNamespace(_coalesce_rewrite_hits={})),
    )

    data_profile_builds: list[str] = []

    def _fake_build(
        compiled_graph: Any,
        task_type_override: str | None = None,
        node_prefix: str | None = None,
    ) -> RuntimeGraph:
        assert node_prefix is not None
        if task_type_override == "data_profile":
            data_profile_builds.append(node_prefix)
        node_id = f"{node_prefix}__runtime"
        op = make_runtime_op(node_id)
        return RuntimeGraph(
            nodes={node_id: op},
            node_order=[node_id],
            output_node_map={node_id: "output"},
        )

    server._runtime_builder.build = _fake_build  # type: ignore[method-assign]

    async def _fake_schedule(
        *,
        request_id: str,
        batch_id: str,
        optimizer_type: str,
        runtime_graph: RuntimeGraph,
        selected_workers: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        data_profile_results: dict[str, list[dict[str, Any]]],
        member_request_ids: set[str] | None = None,
        bearer_token: str | None = None,
    ) -> Schedule:
        return Schedule(
            worker_assignment={selected_workers[0]: list(runtime_graph.node_order)}
        )

    server._generate_schedule_in_subprocess = _fake_schedule  # type: ignore[method-assign]

    collect_called = False

    async def _fake_collect_data_profile(
        **kwargs: Any,
    ) -> dict[str, list[dict[str, Any]]]:
        nonlocal collect_called
        collect_called = True
        return {}

    monkeypatch.setattr(
        "lumilake_server.runtime.server.collect_data_profile",
        _fake_collect_data_profile,
    )

    monkeypatch.setattr(envs, "LUMILAKE_DISABLE_DATA_PROFILE", True)
    await server._run_batch(["worker-1"], batch)

    assert data_profile_builds == []
    assert collect_called is False


@pytest.mark.asyncio
async def test_generate_schedule_subprocess_timeout_terminates_and_kills(
    server_factory,
    monkeypatch,
) -> None:
    server = server_factory()

    class FakeQueue:
        def close(self) -> None:
            return

        def join_thread(self) -> None:
            return

        def get_nowait(self) -> Any:
            raise AssertionError("result queue should not be consumed on timeout")

    class FakeProcess:
        def __init__(self) -> None:
            self.pid = 43210
            self._alive = False
            self.terminate_calls = 0
            self.kill_calls = 0

        def start(self) -> None:
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1
            self._alive = False

        def join(self, timeout: float | None = None) -> None:
            return

    fake_process = FakeProcess()

    class FakeContext:
        def Queue(self, maxsize: int = 0) -> FakeQueue:
            return FakeQueue()

        def Process(self, target: Any, args: tuple[Any, ...]) -> FakeProcess:
            return fake_process

    monkeypatch.setattr(
        "lumilake_server.runtime.server.mp.get_context",
        lambda mode: FakeContext(),
    )
    monkeypatch.setattr(
        envs,
        "LUMILAKE_OPTIMIZER_SUBPROCESS_TIMEOUT_SECONDS",
        0.01,
    )

    runtime_graph = RuntimeGraph(nodes={}, node_order=[], output_node_map={})
    with pytest.raises(RuntimeError, match="timed out"):
        await server._generate_schedule_in_subprocess(
            request_id="req-timeout",
            batch_id="batch-timeout",
            optimizer_type="halo",
            runtime_graph=runtime_graph,
            selected_workers=["worker-1"],
            worker_profiles={},
            data_profile_results={},
        )

    assert fake_process.terminate_calls >= 1
    assert fake_process.kill_calls == 1


class _StubResults:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def download_file(
        self, task_id: str, remote_path: str, local_path: Path
    ) -> None:
        Path(local_path).write_bytes(self._payload)


class _StubFm:
    def __init__(self, payload: bytes) -> None:
        self.results = _StubResults(payload)


class _EmbeddingRuntimeManager(RecordingRuntimeManager):
    """Fakes FlowMesh dispatch but runs the real per-row embedding
    aggregation, proving the fix satisfies `_process_batch`'s per-slice
    output-length contract, not just the aggregation unit in isolation."""

    def __init__(self, *, row_count: int) -> None:
        super().__init__()
        self._row_count = row_count

    async def process_request(
        self,
        request_info: Any,
        schedule: Schedule,
        worker_ids: list[str],
        data_profile_results: dict[str, list[dict[str, Any]]] | None,
    ) -> dict[str, Any]:
        flowmesh_manager = FlowmeshRuntimeManager()
        items = [
            {
                "embedding_file": {"path": "embeddings.safetensors"},
                "model": "BAAI/bge-small-en-v1.5",
                "usage": {"num_requests": self._row_count, "embedding_dim": 4},
            }
        ]
        flat_outputs: dict[str, list[str]] = {}
        for node_id in request_info.output_node_map:
            flat_outputs[node_id] = await flowmesh_manager._aggregate_output_node(
                output_op_id=node_id,
                output_task_id="task-1",
                request_id=request_info.request_id,
                items=items,
                output_path=None,
                expected_row_count=self._row_count,
                list_lambda=False,
            )
        return {"flat_outputs": flat_outputs, "chat_histories": {}, "task_node_map": {}}


class _StaleSingleItemRuntimeManager(RecordingRuntimeManager):
    """Mimics the pre-fix behavior: one collapsed output item for the whole
    embedded slice instead of one per row."""

    async def process_request(
        self,
        request_info: Any,
        schedule: Schedule,
        worker_ids: list[str],
        data_profile_results: dict[str, list[dict[str, Any]]] | None,
    ) -> dict[str, Any]:
        flat_outputs = {
            node_id: ["single-collapsed-item"]
            for node_id in request_info.output_node_map
        }
        return {"flat_outputs": flat_outputs, "chat_histories": {}, "task_node_map": {}}


def _install_fake_build_and_schedule(server: Any, output_name: str) -> None:
    def _fake_build(
        compiled_graph: Any,
        task_type_override: str | None = None,
        node_prefix: str | None = None,
    ) -> RuntimeGraph:
        assert node_prefix is not None
        suffix = "data_profile" if task_type_override == "data_profile" else "runtime"
        node_id = f"{node_prefix}__{suffix}"
        op = make_runtime_op(node_id)
        output_node_map = (
            {} if task_type_override == "data_profile" else {node_id: output_name}
        )
        return RuntimeGraph(
            nodes={node_id: op}, node_order=[node_id], output_node_map=output_node_map
        )

    server._runtime_builder.build = _fake_build  # type: ignore[method-assign]

    async def _fake_schedule(
        *,
        request_id: str,
        batch_id: str,
        optimizer_type: str,
        runtime_graph: RuntimeGraph,
        selected_workers: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        data_profile_results: dict[str, list[dict[str, Any]]],
        member_request_ids: set[str] | None = None,
        bearer_token: str | None = None,
    ) -> Schedule:
        return Schedule(
            worker_assignment={selected_workers[0]: list(runtime_graph.node_order)}
        )

    server._generate_schedule_in_subprocess = _fake_schedule  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_embedding_per_row_outputs_reach_downstream_consumer_one_per_doc(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the per-row output fix: a 3-doc embedding batch must
    surface 3 `vectors` entries on the workflow's output, one per input doc,
    not 1 collapsed artifact — proving a downstream consumer reading the
    job result gets exactly as many embeddings as it submitted docs."""
    docs = ["doc one", "doc two", "doc three"]
    monkeypatch.setattr(
        fm_mod, "flowmesh_for_context", lambda: _StubFm(b"safetensors-bytes")
    )
    monkeypatch.setattr(fm_mod.envs, "S3_ARCHIVE_PREFIX", "s3://bucket/prefix")

    server = server_factory()
    server.runtime_manager = cast(Any, _EmbeddingRuntimeManager(row_count=len(docs)))

    workflows = [
        make_workflow(
            workflow_id="wf-embed",
            request_id="req-embed",
            graph_name="ga",
            public_graph_name="shared",
            slice_length=len(docs),
            total_length=len(docs),
        ),
    ]
    handlers = attach_request_states(server, workflows)
    batch = make_batch(workflows)

    monkeypatch.setattr(
        server,
        "_merge_group_compiled_graph",
        lambda items: cast(Any, SimpleNamespace(_coalesce_rewrite_hits={})),
    )
    _install_fake_build_and_schedule(server, "vectors")

    await server._run_batch(["worker-1"], batch)

    resp = handlers["req-embed"].results[0]
    assert resp.error_info is None
    vectors = resp.outputs["shared"]["vectors"]
    assert len(vectors) == len(docs)
    for idx, raw in enumerate(vectors):
        row = json.loads(raw)
        assert row["row"] == idx
        assert row["output"].endswith("embeddings.safetensors")


@pytest.mark.asyncio
async def test_stale_single_item_embedding_output_fails_downstream_length_check(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Documents the exact failure the fix prevents: collapsing a 3-row
    embedding slice into 1 output item trips the workflow's per-slice
    output-length contract instead of silently under-delivering rows."""
    server = server_factory()
    server.runtime_manager = cast(Any, _StaleSingleItemRuntimeManager())

    workflows = [
        make_workflow(
            workflow_id="wf-embed",
            request_id="req-embed",
            graph_name="ga",
            public_graph_name="shared",
            slice_length=3,
            total_length=3,
        ),
    ]
    handlers = attach_request_states(server, workflows)
    batch = make_batch(workflows)

    monkeypatch.setattr(
        server,
        "_merge_group_compiled_graph",
        lambda items: cast(Any, SimpleNamespace(_coalesce_rewrite_hits={})),
    )
    _install_fake_build_and_schedule(server, "vectors")

    await server._run_batch(["worker-1"], batch)

    errors = handlers["req-embed"].results[0].error_info
    assert errors is not None
    assert any("Output length mismatch" in str(item) for item in errors)


def test_relocate_artifacts_rewrites_nested_uri_in_json_encoded_output(
    server_factory,
) -> None:
    """Per-row artifact output is a JSON-encoded ref, not a bare uri;
    relocate must decode + recurse, not string-replace the raw blob."""
    server = server_factory()
    storage = get_job_storage()
    source_id = "exec-abc123"
    target_id = "req-xyz789"
    filename = "Embed-embeddings.safetensors"
    payload_bytes = b"fake-safetensors-bytes"
    source_uri = storage.save_artifact(
        source_id, filename, payload_bytes, "application/octet-stream"
    )
    value = json.dumps(
        {"output": source_uri, "model": "BAAI/bge-small-en-v1.5", "row": 0}
    )

    relocated = server._relocate_artifacts_for_request(
        value,
        source_request_id=source_id,
        target_request_id=target_id,
        cache={},
    )

    decoded = json.loads(relocated)
    assert decoded["model"] == "BAAI/bge-small-en-v1.5"
    assert decoded["row"] == 0
    assert target_id in decoded["output"]
    # Bytes must actually be copied to target, not just the uri string.
    data, _ = storage.get_artifact(target_id, filename)
    assert data == payload_bytes


class _ListLambdaRuntimeManager(RecordingRuntimeManager):
    """Fakes FlowMesh dispatch but runs the real list-Lambda aggregation,
    proving the whole-list output satisfies `_process_batch`'s demux."""

    def __init__(self, *, items: list[dict[str, Any]]) -> None:
        super().__init__()
        self._items = items

    async def process_request(
        self,
        request_info: Any,
        schedule: Schedule,
        worker_ids: list[str],
        data_profile_results: dict[str, list[dict[str, Any]]] | None,
    ) -> dict[str, Any]:
        flowmesh_manager = FlowmeshRuntimeManager()
        flat_outputs: dict[str, list[str]] = {}
        for node_id in request_info.output_node_map:
            flat_outputs[node_id] = await flowmesh_manager._aggregate_output_node(
                output_op_id=node_id,
                output_task_id="task-1",
                request_id=request_info.request_id,
                items=self._items,
                output_path="items.output",
                list_lambda=True,
            )
        return {"flat_outputs": flat_outputs, "chat_histories": {}, "task_node_map": {}}


def _install_echo_build_and_schedule(server: Any, output_name: str) -> None:
    """Fake build/schedule that marks the output node as a list-Lambda (echo)."""

    def _fake_build(
        compiled_graph: Any,
        task_type_override: str | None = None,
        node_prefix: str | None = None,
    ) -> RuntimeGraph:
        assert node_prefix is not None
        suffix = "data_profile" if task_type_override == "data_profile" else "runtime"
        node_id = f"{node_prefix}__{suffix}"
        op = RuntimeOp(
            node_id=node_id,
            task_type="echo",
            backend="echo",
            model="echo",
            data_spec={},
            model_spec={},
            inference_spec={},
        )
        output_node_map = (
            {} if task_type_override == "data_profile" else {node_id: output_name}
        )
        return RuntimeGraph(
            nodes={node_id: op}, node_order=[node_id], output_node_map=output_node_map
        )

    server._runtime_builder.build = _fake_build  # type: ignore[method-assign]

    async def _fake_schedule(
        *,
        request_id: str,
        batch_id: str,
        optimizer_type: str,
        runtime_graph: RuntimeGraph,
        selected_workers: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        data_profile_results: dict[str, list[dict[str, Any]]],
        member_request_ids: set[str] | None = None,
        bearer_token: str | None = None,
    ) -> Schedule:
        return Schedule(
            worker_assignment={selected_workers[0]: list(runtime_graph.node_order)}
        )

    server._generate_schedule_in_subprocess = _fake_schedule  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_list_lambda_output_single_slice_demux_accepts_one_value(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-row request whose list-Lambda output echoes several items must
    surface ONE output value holding the whole list, and the merged-workflow
    demux must accept it for the single slice."""
    whole_list = [
        {"fid": "f1", "statement": "s1", "quote": "q1"},
        {"fid": "f2", "statement": "s2", "quote": "q2"},
        {"fid": "f3", "statement": "s3", "quote": "q3"},
    ]
    server = server_factory()
    server.runtime_manager = cast(
        Any, _ListLambdaRuntimeManager(items=[{"output": it} for it in whole_list])
    )

    workflows = [
        make_workflow(
            workflow_id="wf-list",
            request_id="req-list",
            graph_name="ga",
            public_graph_name="shared",
            slice_length=1,
            total_length=1,
        ),
    ]
    handlers = attach_request_states(server, workflows)
    batch = make_batch(workflows)

    monkeypatch.setattr(
        server,
        "_merge_group_compiled_graph",
        lambda items: cast(Any, SimpleNamespace(_coalesce_rewrite_hits={})),
    )
    _install_echo_build_and_schedule(server, "observations")

    await server._run_batch(["worker-1"], batch)

    resp = handlers["req-list"].results[0]
    assert resp.error_info is None
    observations = resp.outputs["shared"]["observations"]
    assert len(observations) == 1
    assert json.loads(observations[0]) == whole_list


@pytest.mark.asyncio
async def test_list_lambda_output_multi_slice_run_fails_closed(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A list-mode Lambda runs once over the whole input lists of one run, so
    its single whole-list result cannot be split across slices; a multi-slice
    run must fail closed with a clear error naming the output and slice count."""
    server = server_factory()
    server.runtime_manager = cast(
        Any, _ListLambdaRuntimeManager(items=[{"output": {"fid": "f1"}}])
    )

    slice0, slice1 = make_workflow_slices_from_inputs(
        request_id="req-slices",
        public_graph_name="shared",
        entities=["NVDA", "AAPL"],
    )
    handlers = attach_request_states(server, [slice0, slice1])
    batch = make_batch([slice0, slice1])

    monkeypatch.setattr(
        server,
        "_merge_group_compiled_graph",
        lambda items: cast(Any, SimpleNamespace(_coalesce_rewrite_hits={})),
    )
    _install_echo_build_and_schedule(server, "observations")

    await server._run_batch(["worker-1"], batch)

    resp = handlers["req-slices"].results[0]
    assert resp.error_info is not None
    assert any(
        "List-Lambda output cannot be split across slices" in str(item)
        for item in resp.error_info
    )
