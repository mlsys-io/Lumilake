from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from support.runtime_graphs import build_dummy_runtime_graph

from lumilake_server.runtime.job_manager.base import Job
from lumilake_server.runtime.job_manager.priority_queue import PriorityJobManager
from lumilake_server.runtime.optimizer.base import BaseOptimizer
from lumilake_server.runtime.protocol import LumilakeRequestConfig, Priority
from lumilake_server.runtime.request import WorkflowSliceMeta
from lumilake_server.runtime.runtime_graph import RuntimeGraph


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
    principal_id: str = "p",
    dispatch_token: str | None = None,
    optimizer_type: str | None = None,
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
        config=LumilakeRequestConfig(
            priority=selected_priority,
            user_id=owner,
            principal_id=principal_id,
            optimizer_type=optimizer_type,
        ),
        dispatch_token=dispatch_token,
    )


def _build_multi_graph_job(
    request_id: str,
    graph_names: list[str],
    user_id: str | None = None,
    priority: Priority | None = None,
    principal_id: str = "p",
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
        config=LumilakeRequestConfig(
            priority=selected_priority, user_id=owner, principal_id=principal_id
        ),
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
async def test_select_batch_partitions_by_principal_id() -> None:
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        quantums=_priority_quantums(8),
    )

    await manager.enqueue(_build_job("req-a", "graph-a", principal_id="p-1"))
    await manager.enqueue(_build_job("req-b", "graph-b", principal_id="p-2"))

    first = await manager.select_batch(2)
    assert first is not None
    assert {item.config.principal_id for item in first.workflows} == {
        first.workflows[0].config.principal_id
    }
    selected_in_first = {item.workflow_id for item in first.workflows}

    second = await manager.select_batch(2)
    assert second is not None
    selected_in_second = {item.workflow_id for item in second.workflows}
    assert selected_in_first.isdisjoint(selected_in_second)
    assert {item.config.principal_id for item in second.workflows} == {
        second.workflows[0].config.principal_id
    }
    assert (
        first.workflows[0].config.principal_id
        != second.workflows[0].config.principal_id
    )


@pytest.mark.asyncio
async def test_select_batch_partitions_same_principal_by_dispatch_token() -> None:
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        quantums=_priority_quantums(8),
    )

    await manager.enqueue(
        _build_job("req-a", "graph-a", principal_id="p-1", dispatch_token="tok-old")
    )
    await manager.enqueue(
        _build_job("req-b", "graph-b", principal_id="p-1", dispatch_token="tok-new")
    )

    first = await manager.select_batch(2)
    assert first is not None
    assert len({item.dispatch_token for item in first.workflows}) == 1

    second = await manager.select_batch(2)
    assert second is not None
    assert len({item.dispatch_token for item in second.workflows}) == 1

    assert first.workflows[0].dispatch_token != second.workflows[0].dispatch_token
    assert {first.workflows[0].dispatch_token, second.workflows[0].dispatch_token} == {
        "tok-old",
        "tok-new",
    }


@pytest.mark.asyncio
async def test_select_batch_fills_to_size_within_anchor_principal() -> None:
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        quantums=_priority_quantums(6),
    )

    # Three principals, six workflows each.
    for principal in ("p-1", "p-2", "p-3"):
        for index in range(6):
            await manager.enqueue(
                _build_job(
                    f"req-{principal}-{index}",
                    f"g-{principal}-{index}",
                    principal_id=principal,
                )
            )

    batch = await manager.select_batch(4)
    assert batch is not None
    assert len(batch.workflows) == 4
    assert len({item.config.principal_id for item in batch.workflows}) == 1


@pytest.mark.asyncio
async def test_select_batch_round_robins_across_principals() -> None:
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        quantums=_priority_quantums(8),
    )

    await manager.enqueue(_build_job("req-a", "graph-a", principal_id="p-1"))
    await manager.enqueue(_build_job("req-b", "graph-b", principal_id="p-2"))
    await manager.enqueue(_build_job("req-c", "graph-c", principal_id="p-3"))

    chosen: list[str] = []
    for _ in range(3):
        batch = await manager.select_batch(1)
        assert batch is not None
        assert len({item.config.principal_id for item in batch.workflows}) == 1
        chosen.append(batch.workflows[0].config.principal_id)

    assert sorted(chosen) == ["p-1", "p-2", "p-3"]


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
        "lumilake_server.runtime.job_manager.priority_queue.select_affinity_batch_ids",
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
        "lumilake_server.runtime.job_manager.priority_queue.select_affinity_batch_ids",
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
        "lumilake_server.runtime.job_manager.priority_queue.select_affinity_batch_ids",
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
        "lumilake_server.runtime.job_manager.priority_queue.select_affinity_batch_ids",
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


@pytest.mark.asyncio
async def test_mixed_optimizer_types_split_into_separate_batches() -> None:
    """Two requests from the same principal+token with DIFFERENT optimizer types
    must land in separate batches.  Under the pre-fix code both requests shared
    the same (principal_id, dispatch_token) partition key, so select_batch would
    have returned them together and silently routed both to whichever optimizer
    appeared first in the batch — a security leak for remote optimizer types."""
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        quantums=_priority_quantums(8),
        default_optimizer_type="halo",
    )

    await manager.enqueue(
        _build_job(
            "req-halo",
            "graph-halo",
            principal_id="p-shared",
            dispatch_token="tok-shared",
            optimizer_type="halo",
        )
    )
    await manager.enqueue(
        _build_job(
            "req-topo",
            "graph-topo",
            principal_id="p-shared",
            dispatch_token="tok-shared",
            optimizer_type="topological-sort",
        )
    )

    first = await manager.select_batch(2)
    assert first is not None
    # The batch must be homogeneous: all items share one optimizer_type.
    first_optimizers = {item.config.optimizer_type for item in first.workflows}
    assert len(first_optimizers) == 1

    second = await manager.select_batch(2)
    assert second is not None
    second_optimizers = {item.config.optimizer_type for item in second.workflows}
    assert len(second_optimizers) == 1

    # The two batches carry distinct optimizer types.
    assert first_optimizers != second_optimizers
    assert first_optimizers | second_optimizers == {"halo", "topological-sort"}


@pytest.mark.asyncio
async def test_none_optimizer_cobatches_with_explicit_default() -> None:
    """A request with optimizer=None and one with optimizer=<default> are
    semantically equivalent and must co-batch (None is normalized to the server
    default before partitioning).  This verifies the normalization choice:
    keeping them together avoids an unnecessary partition split for the common
    case where the per-job override happens to match the server default."""
    default = "halo"
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        quantums=_priority_quantums(8),
        default_optimizer_type=default,
    )

    await manager.enqueue(
        _build_job(
            "req-none",
            "graph-none",
            principal_id="p-shared",
            dispatch_token="tok-shared",
            optimizer_type=None,
        )
    )
    await manager.enqueue(
        _build_job(
            "req-explicit",
            "graph-explicit",
            principal_id="p-shared",
            dispatch_token="tok-shared",
            optimizer_type=default,
        )
    )

    batch = await manager.select_batch(2)
    assert batch is not None
    # Both requests must appear in a single batch because they resolve to the
    # same effective optimizer type after None-normalization.
    assert len(batch.workflows) == 2
    assert {item.request_id for item in batch.workflows} == {"req-none", "req-explicit"}


@pytest.mark.asyncio
async def test_reserve_then_abort_does_not_drift_user_rr_pointer() -> None:
    """Aborting a reservation must leave the user-level RR pointer untouched.

    Directly inspects ``_rr_user_order`` before/after an abort cycle and
    also asserts that two consecutive ``reserve+abort`` calls with
    ``batch_size=1`` return the same single item — both would break if
    ``reserve_batch`` rotated the user-RR pointer on the peek path.
    """
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        quantums=_priority_quantums(8),
    )
    primary = _primary_priority()

    await manager.enqueue(_build_job("req-a", "graph-a", user_id="user-a"))
    await manager.enqueue(_build_job("req-b", "graph-b", user_id="user-b"))
    await manager.enqueue(_build_job("req-c", "graph-c", user_id="user-c"))

    order_before = list(manager._rr_user_order[primary])

    first = await manager.reserve_batch(1)
    assert first is not None
    first_ids = [item.workflow_id for item in first.selection.workflows]
    await manager.abort_reservation(first)

    order_after_abort = list(manager._rr_user_order[primary])
    assert order_after_abort == order_before

    second = await manager.reserve_batch(1)
    assert second is not None
    second_ids = [item.workflow_id for item in second.selection.workflows]
    await manager.abort_reservation(second)

    assert order_before == list(manager._rr_user_order[primary])
    assert first_ids == second_ids


@pytest.mark.asyncio
async def test_commit_reservation_rejects_stale_after_abort() -> None:
    """``commit_reservation`` raises if the reservation was already aborted."""
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        quantums=_priority_quantums(8),
    )
    await manager.enqueue(_build_job("req-a", "graph-a", user_id="user-a"))

    first = await manager.reserve_batch(1)
    assert first is not None
    await manager.abort_reservation(first)

    with pytest.raises(RuntimeError, match="active"):
        await manager.commit_reservation(first)


@pytest.mark.asyncio
async def test_abort_reservation_rejects_stale_after_commit() -> None:
    """``abort_reservation`` raises if the reservation was already committed."""
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        quantums=_priority_quantums(8),
    )
    await manager.enqueue(_build_job("req-a", "graph-a", user_id="user-a"))

    first = await manager.reserve_batch(1)
    assert first is not None
    await manager.commit_reservation(first)

    with pytest.raises(RuntimeError, match="active"):
        await manager.abort_reservation(first)
