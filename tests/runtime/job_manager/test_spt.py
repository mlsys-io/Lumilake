"""Tests for the spt scheduling policy."""

from typing import Any, cast

from lumilake_server.runtime.job_manager.base import WorkflowItem
from lumilake_server.runtime.job_manager.policies import SptSchedulingPolicy
from lumilake_server.runtime.protocol import LumilakeRequestConfig, Priority
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _graph(name: str, ops: list[RuntimeOp]) -> RuntimeGraph:
    nodes = {op.node_id: op for op in ops}
    return RuntimeGraph(
        nodes=nodes,
        node_order=[op.node_id for op in ops],
        output_node_map={ops[-1].node_id: "output"},
    )


def _gpu_op(node_id: str) -> RuntimeOp:
    return RuntimeOp(
        node_id=node_id,
        task_type="inference",
        backend="vllm",
        model="llama-7b",
        data_spec={},
        model_spec={},
        inference_spec={},
    )


def _cpu_op(node_id: str) -> RuntimeOp:
    return RuntimeOp(
        node_id=node_id,
        task_type="http",
        backend="http",
        model="http",
        data_spec={},
        model_spec={},
        inference_spec={},
    )


def _db_agent_op(node_id: str) -> RuntimeOp:
    return RuntimeOp(
        node_id=node_id,
        task_type="data_retrieval",
        backend="data_retrieval",
        model="data_retrieval",
        data_spec={"type": "lumid", "mode": "agent"},
        model_spec={},
        inference_spec={},
    )


def _item(request_id: str, graph: RuntimeGraph, enqueued_at: float) -> WorkflowItem:
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
        runtime_graph=graph,
        data_profile_graph=graph,
        dsl_graph=cast(Any, object()),
        config=LumilakeRequestConfig(
            priority=Priority.MEDIUM,
            user_id="u1",
            principal_id="u1",
        ),
        enqueued_at=enqueued_at,
    )


def test_spt_smallest_area_first() -> None:
    policy = SptSchedulingPolicy()
    # CPU op (small area) vs GPU op (large area).
    small = _item("small", _graph("s", [_cpu_op("a")]), 2.0)
    large = _item("large", _graph("l", [_gpu_op("a")]), 1.0)
    assert policy.select_batch([large, small], {}, 2) == ["wf-small", "wf-large"]


def test_spt_no_estimate_sorts_after_estimated() -> None:
    policy = SptSchedulingPolicy()
    estimated = _item("est", _graph("e", [_cpu_op("a")]), 1.0)
    unestimated = _item("agent", _graph("a", [_db_agent_op("a")]), 0.0)
    assert policy.select_batch([unestimated, estimated], {}, 2) == [
        "wf-est",
        "wf-agent",
    ]


def test_spt_ties_break_by_enqueue_order() -> None:
    policy = SptSchedulingPolicy()
    # Same area (both CPU), different enqueue times.
    first = _item("first", _graph("f", [_cpu_op("a")]), 1.0)
    second = _item("second", _graph("s", [_cpu_op("a")]), 2.0)
    assert policy.select_batch([second, first], {}, 2) == ["wf-first", "wf-second"]


def test_spt_respects_batch_size_cap() -> None:
    policy = SptSchedulingPolicy()
    items = [
        _item(f"r{i}", _graph(f"g{i}", [_cpu_op("a")]), float(i)) for i in range(5)
    ]
    assert policy.select_batch(items, {}, 2) == ["wf-r0", "wf-r1"]
