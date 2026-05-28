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
)

from lumilake_server.hooks.security import runtime_token_var
from lumilake_server.runtime.job_manager.base import BatchSelection
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.runtime_graph import RuntimeGraph


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
async def test_run_batch_dispatch_token_picks_first_non_empty(
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
    attach_request_states(server, workflows)
    runtime_manager.set_dispatch_token("req-a", "token-rotated")
    runtime_manager.set_dispatch_token("req-b", "token-fresh")
    batch = make_batch(workflows)
    seen_tokens: list[str | None] = []
    warnings: list[str] = []

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
        outputs = {
            item.workflow_id: {"output": [f"value-{item.request_id}"]}
            for item in selected_batch.workflows
        }
        return outputs, {}

    monkeypatch.setattr(server, "_process_batch", _fake_process_batch)
    original_warning = server.logger.warning

    def _capture_warning(msg: str, *args: Any, **kwargs: Any) -> None:
        warnings.append(msg % args if args else msg)
        original_warning(msg, *args, **kwargs)

    monkeypatch.setattr(server.logger, "warning", _capture_warning)
    await server._run_batch(["worker-1"], batch)

    assert seen_tokens == ["token-rotated"]
    assert any("differing dispatch tokens" in message for message in warnings)


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
        ),
        make_workflow(
            workflow_id="wf-b",
            request_id="req-b",
            graph_name="gb",
            public_graph_name="shared",
        ),
    ]
    attach_request_states(server, workflows)
    runtime_manager.set_dispatch_token("req-a", "tok-cancelled")
    runtime_manager.set_dispatch_token("req-b", "tok-active")
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
        runtime_graph: RuntimeGraph,
        selected_workers: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        data_profile_results: dict[str, list[dict[str, Any]]],
        member_request_ids: set[str] | None = None,
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
        runtime_graph: RuntimeGraph,
        selected_workers: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        data_profile_results: dict[str, list[dict[str, Any]]],
        member_request_ids: set[str] | None = None,
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
            runtime_graph=runtime_graph,
            selected_workers=["worker-1"],
            worker_profiles={},
            data_profile_results={},
        )

    assert fake_process.terminate_calls >= 1
    assert fake_process.kill_calls == 1
