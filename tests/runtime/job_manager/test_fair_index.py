"""Tests for the analytic cost model, attained service, and fair_index policy."""

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from lumilake_server.runtime.job_manager.attained import AttainedService
from lumilake_server.runtime.job_manager.base import Job, WorkflowItem
from lumilake_server.runtime.job_manager.cost import CostParams, estimate_area
from lumilake_server.runtime.job_manager.priority_queue import PriorityJobManager
from lumilake_server.runtime.optimizer.base import BaseOptimizer
from lumilake_server.runtime.protocol import (
    HardwareRequirements,
    LumilakeRequestConfig,
    Priority,
)
from lumilake_server.runtime.request import WorkflowSliceMeta
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp
from tests.support.clock import VirtualClock


def _slice_meta(graph_name: str) -> WorkflowSliceMeta:
    return WorkflowSliceMeta(
        public_graph_name=graph_name,
        slice_index=0,
        slice_start=0,
        slice_length=1,
        total_length=1,
        template_hash=f"hash-{graph_name}",
        varying_input_keys=(),
    )


def _graph_with_ops(name: str, ops: list[RuntimeOp]) -> RuntimeGraph:
    nodes = {op.node_id: op for op in ops}
    return RuntimeGraph(
        nodes=nodes,
        node_order=[op.node_id for op in ops],
        output_node_map={ops[-1].node_id: "output"},
    )


def _gpu_op(node_id: str, model: str = "llama-7b") -> RuntimeOp:
    return RuntimeOp(
        node_id=node_id,
        task_type="inference",
        backend="vllm",
        model=model,
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


def _db_op(node_id: str, mode: str = "sql") -> RuntimeOp:
    return RuntimeOp(
        node_id=node_id,
        task_type="data_retrieval",
        backend="data_retrieval",
        model="data_retrieval",
        data_spec={"type": "lumid", "mode": mode},
        model_spec={},
        inference_spec={},
    )


def _build_job(
    request_id: str,
    graph_name: str,
    runtime_graph: RuntimeGraph,
    user_id: str,
    hardware: HardwareRequirements | None = None,
    principal_id: str | None = None,
    priority: Priority = Priority.MEDIUM,
) -> Job:
    return Job(
        request_id=request_id,
        runtime_graphs={graph_name: runtime_graph},
        data_profile_graphs={graph_name: runtime_graph},
        dsl_graphs={graph_name: cast(Any, object())},
        workflow_slices={graph_name: _slice_meta(graph_name)},
        config=LumilakeRequestConfig(
            priority=priority,
            user_id=user_id,
            principal_id=principal_id or user_id,
            hardware_requirements=hardware,
        ),
    )


def _build_item(
    request_id: str,
    graph_name: str,
    runtime_graph: RuntimeGraph,
    user_id: str,
    hardware: HardwareRequirements | None = None,
) -> WorkflowItem:
    return WorkflowItem(
        workflow_id=f"wf-{request_id}",
        request_id=request_id,
        graph_name=graph_name,
        public_graph_name=graph_name,
        slice_index=0,
        slice_start=0,
        slice_length=1,
        total_length=1,
        template_hash=f"hash-{graph_name}",
        varying_input_keys=(),
        runtime_graph=runtime_graph,
        data_profile_graph=runtime_graph,
        dsl_graph=cast(Any, object()),
        config=LumilakeRequestConfig(
            priority=Priority.MEDIUM,
            user_id=user_id,
            principal_id=user_id,
            hardware_requirements=hardware,
        ),
        enqueued_at=0.0,
    )


def _manager(policy: str = "legacy", **kwargs: Any) -> PriorityJobManager:
    return PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        policy=policy,
        **kwargs,
    )


# -- cost model --------------------------------------------------------------


def test_estimate_area_more_nodes_is_larger() -> None:
    params = CostParams()
    single = _build_item("r1", "g1", _graph_with_ops("g1", [_cpu_op("a")]), "u1")
    # Two chained CPU ops: critical path grows with node count.
    chained = _graph_with_ops(
        "g2",
        [
            RuntimeOp(
                node_id="a",
                task_type="http",
                backend="http",
                model="http",
                data_spec={},
                model_spec={},
                inference_spec={},
            ),
            RuntimeOp(
                node_id="b",
                task_type="http",
                backend="http",
                model="http",
                data_spec={},
                model_spec={},
                inference_spec={},
                dependencies=("a",),
            ),
        ],
    )
    multi = _build_item("r2", "g2", chained, "u1")
    single_area = estimate_area(single, params)
    multi_area = estimate_area(multi, params)
    assert single_area is not None and multi_area is not None
    assert multi_area > single_area


def test_estimate_area_gpu_larger_than_cpu_same_shape() -> None:
    params = CostParams()
    cpu = _build_item("r1", "g1", _graph_with_ops("g1", [_cpu_op("a")]), "u1")
    gpu = _build_item("r2", "g2", _graph_with_ops("g2", [_gpu_op("a")]), "u1")
    cpu_area = estimate_area(cpu, params)
    gpu_area = estimate_area(gpu, params)
    assert cpu_area is not None and gpu_area is not None
    assert gpu_area > cpu_area


def test_estimate_area_wider_hardware_is_larger() -> None:
    params = CostParams()
    small = _build_item(
        "r1",
        "g1",
        _graph_with_ops("g1", [_gpu_op("a")]),
        "u1",
        hardware=HardwareRequirements(cpu=1, gpu=1),
    )
    wide = _build_item(
        "r2",
        "g2",
        _graph_with_ops("g2", [_gpu_op("a")]),
        "u1",
        hardware=HardwareRequirements(cpu=8, gpu=4),
    )
    small_area = estimate_area(small, params)
    wide_area = estimate_area(wide, params)
    assert small_area is not None and wide_area is not None
    assert wide_area > small_area


def test_estimate_area_critical_path_not_sum() -> None:
    params = CostParams()
    # N independent parallel ops: critical path is a single op.
    parallel = _graph_with_ops("p", [_cpu_op("a"), _cpu_op("b"), _cpu_op("c")])
    # N chained ops: critical path is the sum.
    chained = _graph_with_ops(
        "c",
        [
            RuntimeOp(
                node_id="a",
                task_type="http",
                backend="http",
                model="http",
                data_spec={},
                model_spec={},
                inference_spec={},
                dependencies=(),
            ),
            RuntimeOp(
                node_id="b",
                task_type="http",
                backend="http",
                model="http",
                data_spec={},
                model_spec={},
                inference_spec={},
                dependencies=("a",),
            ),
            RuntimeOp(
                node_id="c",
                task_type="http",
                backend="http",
                model="http",
                data_spec={},
                model_spec={},
                inference_spec={},
                dependencies=("b",),
            ),
        ],
    )
    parallel_job = _build_item("r1", "p", parallel, "u1")
    chained_job = _build_item("r2", "c", chained, "u1")
    parallel_area = estimate_area(parallel_job, params)
    chained_area = estimate_area(chained_job, params)
    assert parallel_area is not None and chained_area is not None
    assert chained_area > parallel_area


def test_estimate_area_agent_mode_returns_none() -> None:
    params = CostParams()
    agent = _build_item(
        "r1", "g1", _graph_with_ops("g1", [_db_op("a", mode="agent")]), "u1"
    )
    assert estimate_area(agent, params) is None


# -- attained service --------------------------------------------------------


def test_attained_charge_then_read_at_t0_is_exact() -> None:
    clock = VirtualClock()
    service = AttainedService(half_life_seconds=600.0, clock=clock.now)
    service.charge("u1", 5.0)
    assert service.get("u1") == pytest.approx(5.0)


def test_attained_decays_to_half_after_one_half_life() -> None:
    clock = VirtualClock()
    service = AttainedService(half_life_seconds=10.0, clock=clock.now)
    service.charge("u1", 8.0)
    clock.advance(10.0)
    assert service.get("u1") == pytest.approx(4.0)


def test_attained_decay_is_lazy_and_accumulates() -> None:
    clock = VirtualClock()
    service = AttainedService(half_life_seconds=10.0, clock=clock.now)
    service.charge("u1", 4.0)
    clock.advance(10.0)
    service.charge("u1", 4.0)
    # After one half-life the first 4.0 has decayed to 2.0, then +4.0 = 6.0.
    assert service.get("u1") == pytest.approx(6.0)


# -- fair_index policy -------------------------------------------------------


def test_legacy_policy_is_default() -> None:
    manager = _manager()
    assert manager._policy == "legacy"


@pytest.mark.asyncio
async def test_legacy_selection_preserves_round_robin_and_priority() -> None:
    """The legacy policy preserves per-user round-robin and priority quantums.

    Capacity-aware selection and dispatch are unconditional and not gated
    behind the flag; what ``legacy`` preserves is the ordering policy: users
    within a partition rotate round-robin, and higher-priority queues are
    drained first.
    """
    manager = _manager(policy="legacy")
    # Two users, one item each, same priority and principal (same partition).
    # Round-robin must alternate between them.
    await manager.enqueue(
        _build_job(
            "r1",
            "g1",
            _graph_with_ops("g1", [_cpu_op("a")]),
            "u1",
            principal_id="shared",
        )
    )
    await manager.enqueue(
        _build_job(
            "r2",
            "g2",
            _graph_with_ops("g2", [_cpu_op("b")]),
            "u2",
            principal_id="shared",
        )
    )
    first = await manager.select_batch(1)
    assert first is not None
    assert first.workflows[0].config.user_id == "u1"
    second = await manager.select_batch(1)
    assert second is not None
    assert second.workflows[0].config.user_id == "u2"

    # Priority: a HIGH item queued behind a MEDIUM item is picked first. The
    # candidate pool is built HIGH-first, so a batch of size 2 returns the
    # HIGH item before the MEDIUM one.
    manager2 = _manager(policy="legacy")
    await manager2.enqueue(
        _build_job(
            "m1",
            "g1",
            _graph_with_ops("g1", [_cpu_op("a")]),
            "u1",
            principal_id="shared",
            priority=Priority.MEDIUM,
        )
    )
    await manager2.enqueue(
        _build_job(
            "h1",
            "g2",
            _graph_with_ops("g2", [_cpu_op("b")]),
            "u2",
            principal_id="shared",
            priority=Priority.HIGH,
        )
    )
    picked = await manager2.select_batch(2)
    assert picked is not None
    assert [i.request_id for i in picked.workflows] == ["h1", "m1"]


@pytest.mark.asyncio
async def test_legacy_selection_pins_starvation() -> None:
    """Starvation pinning is preserved: a repeatedly-missed item is forced in."""
    manager = _manager(policy="legacy", starvation_limit=2)
    await manager.enqueue(
        _build_job(
            "starved",
            "g1",
            _graph_with_ops("g1", [_cpu_op("a")]),
            "u1",
            principal_id="shared",
        )
    )
    starved_items = await manager.enqueue(
        _build_job(
            "other",
            "g2",
            _graph_with_ops("g2", [_cpu_op("b")]),
            "u2",
            principal_id="shared",
        )
    )
    starved_items[0].miss_count = 2

    captured: dict[str, list[str]] = {}

    def _capture_pins(
        runtime_graphs: dict[str, RuntimeGraph],
        _enqueued_at: dict[str, float],
        _batch_size: int,
        *,
        pinned_ids: list[str] | None = None,
        **_kwargs: Any,
    ) -> list[str]:
        captured["pinned_ids"] = [] if pinned_ids is None else list(pinned_ids)
        return [next(iter(runtime_graphs))]

    with patch(
        "lumilake_server.runtime.job_manager.priority_queue.select_affinity_batch_ids",
        side_effect=_capture_pins,
    ):
        batch = await manager.select_batch(1)
        assert batch is not None

    assert starved_items[0].workflow_id in captured["pinned_ids"]
    assert batch.workflows[0].workflow_id == starved_items[0].workflow_id


@pytest.mark.asyncio
async def test_fair_index_heavy_user_selected_less() -> None:
    """Two users with equal item counts but 10x different area: the heavy user
    is selected less often over many rounds."""
    clock = VirtualClock()
    manager = _manager(
        policy="fair_index",
        fair_share_target=10.0,
        fairness_half_life_seconds=600.0,
        clock=clock.now,
        starvation_limit=1000,
    )
    # Heavy user: GPU ops (large area). Light user: CPU ops (small area).
    # Both share a principal so they land in the same partition / candidate pool.
    # Equal item counts (10 each) so the difference is purely area.
    for i in range(10):
        await manager.enqueue(
            _build_job(
                f"heavy-{i}",
                f"h-{i}",
                _graph_with_ops(f"h-{i}", [_gpu_op("a")]),
                "heavy",
                principal_id="shared",
            )
        )
        await manager.enqueue(
            _build_job(
                f"light-{i}",
                f"l-{i}",
                _graph_with_ops(f"l-{i}", [_cpu_op("a")]),
                "light",
                principal_id="shared",
            )
        )

    heavy_picks = 0
    light_picks = 0
    # Only the first 10 rounds matter: both users have 10 items, so after the
    # light user's items are consumed first the remaining picks are all heavy.
    for _ in range(10):
        batch = await manager.select_batch(1)
        assert batch is not None
        if batch.workflows[0].config.user_id == "heavy":
            heavy_picks += 1
        else:
            light_picks += 1
        clock.advance(1.0)
    assert light_picks > heavy_picks


@pytest.mark.asyncio
async def test_fair_index_las_fallback_does_not_crash() -> None:
    """An unestimable item (agent-mode retrieval) engages the LAS fallback
    rather than crashing."""
    clock = VirtualClock()
    manager = _manager(
        policy="fair_index",
        fair_share_target=10.0,
        fairness_half_life_seconds=600.0,
        clock=clock.now,
    )
    agent_graph = _graph_with_ops("a", [_db_op("a", mode="agent")])
    await manager.enqueue(_build_job("agent", "a", agent_graph, "u1"))
    batch = await manager.select_batch(1)
    assert batch is not None
    assert len(batch.workflows) == 1


@pytest.mark.asyncio
async def test_fair_index_starvation_pinning_still_fires() -> None:
    """Starvation pinning must still fire under fair_index."""
    quantums = {priority: 0 for priority in Priority}
    quantums[Priority.HIGH] = 1
    quantums[Priority.LOW] = 1
    clock = VirtualClock()
    manager = _manager(
        policy="fair_index",
        fair_share_target=10.0,
        fairness_half_life_seconds=600.0,
        clock=clock.now,
        quantums=quantums,
        starvation_limit=2,
    )
    await manager.enqueue(
        _build_job(
            "high",
            "h",
            _graph_with_ops("h", [_gpu_op("a")]),
            "u-high",
            principal_id="shared",
            priority=Priority.HIGH,
        )
    )
    low_items = await manager.enqueue(
        _build_job(
            "low",
            "l",
            _graph_with_ops("l", [_cpu_op("a")]),
            "u-low",
            principal_id="shared",
            priority=Priority.LOW,
        )
    )
    low_items[0].miss_count = 2

    captured: dict[str, list[str]] = {}

    def _capture_pins(
        runtime_graphs: dict[str, RuntimeGraph],
        _enqueued_at: dict[str, float],
        _batch_size: int,
        *,
        pinned_ids: list[str] | None = None,
        **_kwargs: Any,
    ) -> list[str]:
        captured["pinned_ids"] = [] if pinned_ids is None else list(pinned_ids)
        return [next(iter(runtime_graphs))]

    with patch(
        "lumilake_server.runtime.job_manager.priority_queue.select_affinity_batch_ids",
        side_effect=_capture_pins,
    ):
        batch = await manager.select_batch(1)
        assert batch is not None

    assert low_items[0].workflow_id in captured["pinned_ids"]
    assert batch.workflows[0].workflow_id == low_items[0].workflow_id
