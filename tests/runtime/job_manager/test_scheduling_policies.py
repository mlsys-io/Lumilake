"""Every registered scheduling policy against one shared workload."""

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from lumilake_server.runtime.job_manager.base import Job, WorkflowItem
from lumilake_server.runtime.job_manager.cost import estimate_area
from lumilake_server.runtime.job_manager.policies import (
    SCHEDULING_POLICIES,
    create_scheduling_policy,
)
from lumilake_server.runtime.job_manager.priority_queue import PriorityJobManager
from lumilake_server.runtime.optimizer.base import BaseOptimizer
from lumilake_server.runtime.protocol import LumilakeRequestConfig, Priority
from lumilake_server.runtime.request import WorkflowSliceMeta
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _op(node_id: str, kind: str, dependencies: tuple[str, ...]) -> RuntimeOp:
    if kind == "gpu":
        return RuntimeOp(
            node_id=node_id,
            dependencies=dependencies,
            task_type="inference",
            backend="vllm",
            model="llama-7b",
            data_spec={},
            model_spec={},
            inference_spec={},
        )
    if kind == "agent":
        return RuntimeOp(
            node_id=node_id,
            dependencies=dependencies,
            task_type="data_retrieval",
            backend="data_retrieval",
            model="data_retrieval",
            data_spec={"type": "lumid", "mode": "agent"},
            model_spec={},
            inference_spec={},
        )
    return RuntimeOp(
        node_id=node_id,
        dependencies=dependencies,
        task_type="http",
        backend="http",
        model="http",
        data_spec={},
        model_spec={},
        inference_spec={},
    )


def _graph(kinds: list[str]) -> RuntimeGraph:
    """A sequential graph with one op per entry of ``kinds``."""
    ops = [
        _op(f"n{i}", kind, (f"n{i - 1}",) if i else ()) for i, kind in enumerate(kinds)
    ]
    return RuntimeGraph(
        nodes={op.node_id: op for op in ops},
        node_order=[op.node_id for op in ops],
        output_node_map={ops[-1].node_id: "output"},
    )


def _config(user_id: str, chain_id: str | None) -> LumilakeRequestConfig:
    return LumilakeRequestConfig(
        priority=Priority.MEDIUM,
        user_id=user_id,
        principal_id=user_id,
        chain_id=chain_id,
    )


def _item(
    request_id: str,
    kinds: list[str],
    enqueued_at: float,
    user_id: str = "u1",
    chain_id: str | None = None,
) -> WorkflowItem:
    graph = _graph(kinds)
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
        config=_config(user_id, chain_id),
        enqueued_at=enqueued_at,
    )


def _job(request_id: str, user_id: str) -> Job:
    graph = _graph(["cpu"])
    return Job(
        request_id=request_id,
        runtime_graphs={"g": graph},
        data_profile_graphs={"g": graph},
        dsl_graphs={"g": cast(Any, object())},
        workflow_slices={
            "g": WorkflowSliceMeta(
                public_graph_name="g",
                slice_index=0,
                slice_start=0,
                slice_length=1,
                total_length=1,
                template_hash=f"hash-{request_id}",
                varying_input_keys=(),
            )
        },
        config=_config(user_id, None),
    )


# Candidates, in the round-robin order the job manager hands to the policy.
# Areas: cpu < gpu < gpu+gpu; agent retrieval has no estimate. gpu and gpu2
# share an enqueue time; cpu1 and cpu0 share an area.
CANDIDATES = [
    _item("cpu1", ["cpu"], 3.0),
    _item("gpu", ["gpu"], 1.0, chain_id="c1"),
    _item("gpu2", ["gpu", "gpu"], 1.0, user_id="u2"),
    _item("agent", ["agent"], 0.0, user_id="u2"),
    _item("cpu0", ["cpu"], 4.0, chain_id="c2"),
]
# Committed before selection: an earlier round of c1, and an unestimable
# round of c2 that charges nothing.
HISTORY = [
    _item("c1r0", ["gpu"], -2.0, chain_id="c1"),
    _item("c2r0", ["agent"], -1.0, chain_id="c2"),
]
AFFINITY_RANK = {"wf-gpu2": 0, "wf-cpu1": 1}

EXPECTED = {
    # One item per user in round-robin order (preferring affinity picks),
    # then the rest in candidate order.
    "default": ["cpu1", "gpu2", "gpu", "agent", "cpu0"],
    # Enqueue order; the gpu/gpu2 tie breaks by workflow id.
    "fifo": ["agent", "gpu", "gpu2", "cpu1", "cpu0"],
    # Smallest area first; the cpu tie breaks by enqueue order; no estimate last.
    "spt": ["cpu1", "cpu0", "gpu", "gpu2", "agent"],
    "lpt": ["gpu2", "gpu", "cpu1", "cpu0", "agent"],
    # Chain c1 has attained service; c2's only round charged nothing, so it
    # ties with the standalone jobs at zero and sorts by enqueue order.
    "plas": ["agent", "gpu2", "cpu1", "cpu0", "gpu"],
}


def _select(policy: str, batch_size: int) -> list[str]:
    impl = create_scheduling_policy(policy)
    impl.on_commit(HISTORY)
    selected = impl.select_batch(list(CANDIDATES), dict(AFFINITY_RANK), batch_size)
    return [wid.removeprefix("wf-") for wid in selected]


def test_workload_areas_are_as_described() -> None:
    area = {item.request_id: estimate_area(item) for item in CANDIDATES}
    assert area.pop("agent") is None
    estimated = {key: value for key, value in area.items() if value is not None}
    assert estimated.keys() == area.keys()
    assert estimated["cpu1"] == estimated["cpu0"]
    assert estimated["cpu1"] < estimated["gpu"] < estimated["gpu2"]


def test_every_policy_has_ground_truth() -> None:
    assert set(EXPECTED) == set(SCHEDULING_POLICIES)


@pytest.mark.parametrize("policy", sorted(EXPECTED))
@pytest.mark.parametrize("batch_size", [2, len(CANDIDATES)])
def test_policy_matches_ground_truth(policy: str, batch_size: int) -> None:
    assert _select(policy, batch_size) == EXPECTED[policy][:batch_size]


@pytest.mark.parametrize("policy", sorted(SCHEDULING_POLICIES))
@pytest.mark.parametrize("batch_size", range(1, len(CANDIDATES) + 2))
def test_policy_returns_distinct_candidates_within_cap(
    policy: str, batch_size: int
) -> None:
    selected = _select(policy, batch_size)
    assert len(selected) == len(set(selected)) <= batch_size
    assert set(selected) <= {item.request_id for item in CANDIDATES}


@pytest.mark.parametrize("policy", sorted(SCHEDULING_POLICIES))
@pytest.mark.asyncio
async def test_policy_selects_through_job_manager(policy: str) -> None:
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        policy=policy,
    )
    await manager.enqueue(_job("r1", "u1"))
    await manager.enqueue(_job("r2", "u1"))

    batch = await manager.select_batch(2)
    assert batch is not None
    assert {i.request_id for i in batch.workflows} == {"r1", "r2"}
