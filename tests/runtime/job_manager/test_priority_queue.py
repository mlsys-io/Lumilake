from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from support.runtime_graphs import build_dummy_runtime_graph

from lumilake.runtime.job_manager.base import Job
from lumilake.runtime.job_manager.priority_queue import PriorityJobManager
from lumilake.runtime.optimizer.base import BaseOptimizer
from lumilake.runtime.protocol import LumilakeRequestConfig, Priority
from lumilake.runtime.request import WorkflowSliceMeta
from lumilake.runtime.runtime_graph import RuntimeGraph


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


def _primary_priority() -> Priority:
    return next(iter(Priority))


def _priority_quantums(primary: int) -> dict[Priority, int]:
    quantums = {priority: 0 for priority in Priority}
    quantums[_primary_priority()] = primary
    return quantums


def _build_job(
    request_id: str,
    graph_name: str,
    user_id: str | None = None,
    priority: Priority | None = None,
) -> Job:
    runtime_graph = build_dummy_runtime_graph(graph_name)
    owner = request_id if user_id is None else user_id
    selected_priority = _primary_priority() if priority is None else priority
    return Job(
        request_id=request_id,
        runtime_graphs={graph_name: runtime_graph},
        data_profile_graphs={graph_name: runtime_graph},
        dsl_graphs={graph_name: cast(Any, object())},
        workflow_slices={graph_name: _slice_meta(graph_name)},
        config=LumilakeRequestConfig(priority=selected_priority, user_id=owner),
    )


def _build_multi_graph_job(
    request_id: str,
    graph_names: list[str],
    user_id: str | None = None,
    priority: Priority | None = None,
) -> Job:
    runtime_graphs = {name: build_dummy_runtime_graph(name) for name in graph_names}
    owner = request_id if user_id is None else user_id
    selected_priority = _primary_priority() if priority is None else priority
    return Job(
        request_id=request_id,
        runtime_graphs=runtime_graphs,
        data_profile_graphs=runtime_graphs,
        dsl_graphs={name: cast(Any, object()) for name in graph_names},
        workflow_slices={name: _slice_meta(name) for name in graph_names},
        config=LumilakeRequestConfig(priority=selected_priority, user_id=owner),
    )


@pytest.mark.asyncio
async def test_select_batch_can_mix_multiple_requests() -> None:
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        quantums=_priority_quantums(8),
    )

    await manager.enqueue(_build_job("req-a", "graph-a"))
    await manager.enqueue(_build_job("req-b", "graph-b"))

    batch = await manager.select_batch(2)

    assert batch is not None
    assert len(batch.workflows) == 2
    assert {item.request_id for item in batch.workflows} == {"req-a", "req-b"}


@pytest.mark.asyncio
async def test_select_batch_round_robin_fairness_across_same_priority_requests() -> (
    None
):
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        quantums=_priority_quantums(8),
    )

    await manager.enqueue(_build_multi_graph_job("req-a", ["a-1", "a-2"]))
    await manager.enqueue(_build_multi_graph_job("req-b", ["b-1", "b-2"]))

    batch = await manager.select_batch(2)
    assert batch is not None
    assert len(batch.workflows) == 2
    assert {item.request_id for item in batch.workflows} == {"req-a", "req-b"}


@pytest.mark.asyncio
async def test_select_batch_round_robin_fairness_across_users() -> None:
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        quantums=_priority_quantums(8),
    )

    await manager.enqueue(
        _build_multi_graph_job("req-a", ["a-1", "a-2"], user_id="user-a")
    )
    await manager.enqueue(_build_job("req-b", "b-1", user_id="user-b"))

    batch = await manager.select_batch(2)
    assert batch is not None
    assert len(batch.workflows) == 2
    assert {item.config.user_id for item in batch.workflows} == {"user-a", "user-b"}


@pytest.mark.asyncio
async def test_starvation_override_allows_low_priority_progress() -> None:
    quantums = {priority: 0 for priority in Priority}
    quantums[Priority.HIGH] = 8
    quantums[Priority.LOW] = 1
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        quantums=quantums,
        starvation_limit=2,
    )

    await manager.enqueue(
        _build_job(
            "req-low",
            "l-1",
            user_id="user-low",
            priority=Priority.LOW,
        )
    )

    def _prefer_high(
        runtime_graphs: dict[str, RuntimeGraph],
        _enqueued_at: dict[str, float],
        _batch_size: int,
        **_kwargs: Any,
    ) -> list[str]:
        for workflow_id in runtime_graphs:
            if manager.get_workflow(workflow_id).config.priority == Priority.HIGH:
                return [workflow_id]
        return [next(iter(runtime_graphs))]

    selected_priorities: list[Priority] = []
    with patch(
        "lumilake.runtime.job_manager.priority_queue.select_affinity_batch_ids",
        side_effect=_prefer_high,
    ):
        for round_idx in range(3):
            await manager.enqueue(
                _build_job(
                    f"req-high-{round_idx}",
                    f"h-{round_idx}",
                    user_id="user-high",
                    priority=Priority.HIGH,
                )
            )
            batch = await manager.select_batch(1)
            assert batch is not None
            selected_priorities.append(batch.workflows[0].config.priority)

    assert selected_priorities[0] == Priority.HIGH
    assert selected_priorities[1] == Priority.HIGH
    assert selected_priorities[2] == Priority.LOW


@pytest.mark.asyncio
async def test_starved_ids_are_pinned_for_affinity_selection() -> None:
    quantums = {priority: 0 for priority in Priority}
    quantums[Priority.HIGH] = 1
    quantums[Priority.LOW] = 1
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        quantums=quantums,
        starvation_limit=2,
    )

    await manager.enqueue(
        _build_job(
            "req-high",
            "h-1",
            user_id="user-high",
            priority=Priority.HIGH,
        )
    )
    low_items = await manager.enqueue(
        _build_job(
            "req-low",
            "l-1",
            user_id="user-low",
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
        "lumilake.runtime.job_manager.priority_queue.select_affinity_batch_ids",
        side_effect=_capture_pins,
    ):
        batch = await manager.select_batch(1)
        assert batch is not None

    assert low_items[0].workflow_id in captured["pinned_ids"]
    assert batch.workflows[0].workflow_id == low_items[0].workflow_id


@pytest.mark.asyncio
async def test_multi_tenant_case_07_priority_with_starvation_progression() -> None:
    quantums = {priority: 0 for priority in Priority}
    # One visible candidate per priority per round keeps medium slices moving while
    # allowing low slices to accumulate misses and trigger starvation relief.
    quantums[Priority.MEDIUM] = 1
    quantums[Priority.LOW] = 1
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        quantums=quantums,
        starvation_limit=3,
    )

    medium_graphs = [f"07A-text-analysis__slice_{idx}" for idx in range(10)]
    low_graphs = [f"07B-ETL__slice_{idx}" for idx in range(6)]
    await manager.enqueue(
        _build_multi_graph_job(
            "req-07-medium",
            medium_graphs,
            user_id="user8",
            priority=Priority.MEDIUM,
        )
    )
    await manager.enqueue(
        _build_multi_graph_job(
            "req-07-low",
            low_graphs,
            user_id="user4",
            priority=Priority.LOW,
        )
    )

    priority_rank = {
        Priority.HIGH: 0,
        Priority.MEDIUM: 1,
        Priority.LOW: 2,
    }

    def _prefer_higher_priority(
        runtime_graphs: dict[str, RuntimeGraph],
        _enqueued_at: dict[str, float],
        _batch_size: int,
        **_kwargs: Any,
    ) -> list[str]:
        selected = min(
            runtime_graphs,
            key=lambda workflow_id: priority_rank[
                manager.get_workflow(workflow_id).config.priority
            ],
        )
        return [selected]

    picked_priorities: list[Priority] = []
    with patch(
        "lumilake.runtime.job_manager.priority_queue.select_affinity_batch_ids",
        side_effect=_prefer_higher_priority,
    ):
        while await manager.has_work():
            batch = await manager.select_batch(1)
            assert batch is not None
            picked_priorities.append(batch.workflows[0].config.priority)

    assert len(picked_priorities) == len(medium_graphs) + len(low_graphs)
    last_medium_idx = max(
        idx
        for idx, priority in enumerate(picked_priorities)
        if priority == Priority.MEDIUM
    )
    assert all(
        priority == Priority.LOW
        for priority in picked_priorities[last_medium_idx + 1 :]
    )

    before_medium_finishes = picked_priorities[: last_medium_idx + 1]
    low_indices_before_finish = [
        idx
        for idx, priority in enumerate(before_medium_finishes)
        if priority == Priority.LOW
    ]
    assert len(low_indices_before_finish) >= 2
    assert low_indices_before_finish[0] <= 3
    assert all(
        low_indices_before_finish[idx] - low_indices_before_finish[idx - 1] <= 4
        for idx in range(1, len(low_indices_before_finish))
    )
    assert before_medium_finishes.count(Priority.MEDIUM) > before_medium_finishes.count(
        Priority.LOW
    )


@pytest.mark.asyncio
async def test_multi_tenant_case_08_two_jobs_can_share_one_batch() -> None:
    quantums = {priority: 0 for priority in Priority}
    quantums[Priority.MEDIUM] = 8
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        quantums=quantums,
    )

    await manager.enqueue(
        _build_job(
            "req-08-social",
            "08A-text-analysis",
            user_id="user8",
            priority=Priority.MEDIUM,
        )
    )
    await manager.enqueue(
        _build_job(
            "req-08-trading",
            "08B-multi-modal-analysis",
            user_id="user4",
            priority=Priority.MEDIUM,
        )
    )

    def _prefer_single_id(
        runtime_graphs: dict[str, RuntimeGraph],
        _enqueued_at: dict[str, float],
        _batch_size: int,
        **_kwargs: Any,
    ) -> list[str]:
        return [next(iter(runtime_graphs))]

    with patch(
        "lumilake.runtime.job_manager.priority_queue.select_affinity_batch_ids",
        side_effect=_prefer_single_id,
    ):
        batch = await manager.select_batch(2)

    assert batch is not None
    assert len(batch.workflows) == 2
    assert {item.request_id for item in batch.workflows} == {
        "req-08-social",
        "req-08-trading",
    }
    assert {item.config.user_id for item in batch.workflows} == {"user8", "user4"}
