"""Tests for the plas scheduling policy."""

from typing import Any, cast

from lumilake_server.runtime.job_manager.base import WorkflowItem
from lumilake_server.runtime.job_manager.policies import PlasSchedulingPolicy
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


def _item(
    request_id: str,
    graph: RuntimeGraph,
    enqueued_at: float,
    chain_id: str | None = None,
) -> WorkflowItem:
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
            chain_id=chain_id,
        ),
        enqueued_at=enqueued_at,
    )


def test_plas_least_attained_chain_first() -> None:
    policy = PlasSchedulingPolicy()
    # Chain A has already been charged (two rounds); chain B is fresh.
    chain_a_round1 = _item("a1", _graph("a1", [_cpu_op("a")]), 1.0, chain_id="chain-a")
    chain_a_round2 = _item("a2", _graph("a2", [_cpu_op("a")]), 2.0, chain_id="chain-a")
    chain_b = _item("b1", _graph("b1", [_cpu_op("a")]), 3.0, chain_id="chain-b")
    policy.on_commit([chain_a_round1, chain_a_round2])

    # chain-b has zero attained service, so it sorts before chain-a.
    assert policy.select_batch([chain_a_round2, chain_b], {}, 2) == [
        "wf-b1",
        "wf-a2",
    ]


def test_plas_charges_estimate_area_on_commit() -> None:
    policy = PlasSchedulingPolicy()
    # A GPU round is larger than a CPU round; both belong to the same chain.
    gpu = _item("g", _graph("g", [_gpu_op("a")]), 1.0, chain_id="chain-a")
    cpu = _item("c", _graph("c", [_cpu_op("a")]), 2.0, chain_id="chain-a")
    policy.on_commit([gpu, cpu])

    # A fresh chain sorts before the charged chain regardless of enqueue order.
    fresh = _item("f", _graph("f", [_cpu_op("a")]), 0.0, chain_id="chain-fresh")
    assert policy.select_batch([gpu, fresh], {}, 2) == ["wf-f", "wf-g"]


def test_plas_skips_unestimable_charge() -> None:
    policy = PlasSchedulingPolicy()
    agent = _item(
        "agent",
        _graph(
            "a",
            [
                RuntimeOp(
                    node_id="a",
                    task_type="data_retrieval",
                    backend="data_retrieval",
                    model="data_retrieval",
                    data_spec={"type": "lumid", "mode": "agent"},
                    model_spec={},
                    inference_spec={},
                )
            ],
        ),
        1.0,
        chain_id="chain-a",
    )
    policy.on_commit([agent])
    # No charge recorded: the chain's attained service stays zero.
    assert policy._attained == {}


def test_plas_respects_batch_size_cap() -> None:
    policy = PlasSchedulingPolicy()
    items = [
        _item(f"r{i}", _graph(f"g{i}", [_cpu_op("a")]), float(i)) for i in range(5)
    ]
    assert policy.select_batch(items, {}, 2) == ["wf-r0", "wf-r1"]
