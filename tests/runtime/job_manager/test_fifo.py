"""Tests for the fifo scheduling policy."""

from typing import Any, cast

from lumilake_server.runtime.job_manager.base import WorkflowItem
from lumilake_server.runtime.job_manager.policies import FifoSchedulingPolicy
from lumilake_server.runtime.protocol import LumilakeRequestConfig, Priority
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _graph(name: str) -> RuntimeGraph:
    op = RuntimeOp(
        node_id="a",
        task_type="http",
        backend="http",
        model="http",
        data_spec={},
        model_spec={},
        inference_spec={},
    )
    return RuntimeGraph(
        nodes={"a": op},
        node_order=["a"],
        output_node_map={"a": "output"},
    )


def _item(request_id: str, enqueued_at: float) -> WorkflowItem:
    return WorkflowItem(
        workflow_id=f"wf-{request_id}",
        request_id=request_id,
        graph_name="g",
        public_graph_name="g",
        slice_index=0,
        slice_start=0,
        slice_length=1,
        total_length=1,
        template_hash="hash",
        varying_input_keys=(),
        runtime_graph=_graph("g"),
        data_profile_graph=_graph("g"),
        dsl_graph=cast(Any, object()),
        config=LumilakeRequestConfig(
            priority=Priority.MEDIUM,
            user_id="u1",
            principal_id="u1",
        ),
        enqueued_at=enqueued_at,
    )


def _select(
    policy: FifoSchedulingPolicy,
    items: list[WorkflowItem],
    batch_size: int,
) -> list[str]:
    return policy.select_batch(items, {}, batch_size)


def test_fifo_selects_in_enqueue_order() -> None:
    policy = FifoSchedulingPolicy()
    items = [
        _item("r1", 1.0),
        _item("r2", 2.0),
        _item("r3", 3.0),
    ]
    assert _select(policy, items, 3) == ["wf-r1", "wf-r2", "wf-r3"]


def test_fifo_respects_batch_size_cap() -> None:
    policy = FifoSchedulingPolicy()
    items = [_item(f"r{i}", float(i)) for i in range(5)]
    assert _select(policy, items, 2) == ["wf-r0", "wf-r1"]


def test_fifo_ties_break_by_workflow_id() -> None:
    policy = FifoSchedulingPolicy()
    items = [
        _item("r-b", 1.0),
        _item("r-a", 1.0),
        _item("r-c", 1.0),
    ]
    assert _select(policy, items, 3) == ["wf-r-a", "wf-r-b", "wf-r-c"]
