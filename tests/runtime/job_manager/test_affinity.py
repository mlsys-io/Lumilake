"""Tests for the affinity scheduling policy."""

from typing import Any, cast

from lumilake_server.runtime.job_manager.base import WorkflowItem
from lumilake_server.runtime.job_manager.policies import AffinitySchedulingPolicy
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


def test_affinity_follows_affinity_rank_order() -> None:
    policy = AffinitySchedulingPolicy()
    items = [_item("a", 1.0), _item("b", 2.0), _item("c", 3.0)]
    # affinity_rank orders c, a, b.
    rank = {"wf-c": 0, "wf-a": 1, "wf-b": 2}
    assert policy.select_batch(items, rank, 3) == ["wf-c", "wf-a", "wf-b"]


def test_affinity_fills_remaining_in_enqueue_order() -> None:
    policy = AffinitySchedulingPolicy()
    items = [_item("a", 1.0), _item("b", 2.0), _item("c", 3.0)]
    # affinity_rank only covers c; the rest fill by enqueue order.
    rank = {"wf-c": 0}
    assert policy.select_batch(items, rank, 3) == ["wf-c", "wf-a", "wf-b"]


def test_affinity_respects_batch_size_cap() -> None:
    policy = AffinitySchedulingPolicy()
    items = [_item("a", 1.0), _item("b", 2.0), _item("c", 3.0)]
    rank = {"wf-c": 0, "wf-a": 1, "wf-b": 2}
    assert policy.select_batch(items, rank, 2) == ["wf-c", "wf-a"]


def test_affinity_ignores_rank_ids_not_in_candidates() -> None:
    policy = AffinitySchedulingPolicy()
    items = [_item("a", 1.0), _item("b", 2.0)]
    rank = {"wf-x": 0, "wf-a": 1, "wf-b": 2}
    assert policy.select_batch(items, rank, 2) == ["wf-a", "wf-b"]
