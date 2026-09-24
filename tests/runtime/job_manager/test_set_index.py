"""Tests for the set_index scheduling policy."""

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from lumilake_server.runtime.job_manager.base import Job, WorkflowItem
from lumilake_server.runtime.job_manager.policies import (
    SCHEDULING_POLICIES,
    create_scheduling_policy,
)
from lumilake_server.runtime.job_manager.policies.set_index import (
    MODEL_SET_CAP,
    SetIndexSchedulingPolicy,
)
from lumilake_server.runtime.job_manager.priority_queue import PriorityJobManager
from lumilake_server.runtime.optimizer.base import BaseOptimizer
from lumilake_server.runtime.protocol import LumilakeRequestConfig, Priority
from lumilake_server.runtime.request import WorkflowSliceMeta
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _gpu_op(node_id: str, model: str) -> RuntimeOp:
    return RuntimeOp(
        node_id=node_id,
        task_type="inference",
        backend="vllm",
        model=model,
        data_spec={},
        model_spec={},
        inference_spec={},
    )


def _graph(model: str) -> RuntimeGraph:
    op = _gpu_op("a", model)
    return RuntimeGraph(
        nodes={"a": op},
        node_order=["a"],
        output_node_map={"a": "output"},
    )


def _cpu_graph() -> RuntimeGraph:
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


def _item(wid: str, model: str, weight: float) -> WorkflowItem:
    return WorkflowItem(
        workflow_id=wid,
        request_id=wid,
        graph_name="g",
        public_graph_name="g",
        slice_index=0,
        slice_start=0,
        slice_length=1,
        total_length=1,
        template_hash="h",
        varying_input_keys=(),
        runtime_graph=_graph(model),
        data_profile_graph=_graph(model),
        dsl_graph=cast(Any, object()),
        config=LumilakeRequestConfig(
            priority=Priority.MEDIUM,
            user_id="u",
            principal_id="u",
        ),
        enqueued_at=0.0,
    )


def _cpu_item(wid: str, weight: float) -> WorkflowItem:
    """A model-agnostic item (CPU/DB backend): no required model."""
    graph = _cpu_graph()
    return WorkflowItem(
        workflow_id=wid,
        request_id=wid,
        graph_name="g",
        public_graph_name="g",
        slice_index=0,
        slice_start=0,
        slice_length=1,
        total_length=1,
        template_hash="h",
        varying_input_keys=(),
        runtime_graph=graph,
        data_profile_graph=graph,
        dsl_graph=cast(Any, object()),
        config=LumilakeRequestConfig(
            priority=Priority.MEDIUM,
            user_id="u",
            principal_id="u",
        ),
        enqueued_at=0.0,
    )


def _policy(
    setup_cost_sigma: float,
    *,
    resident_model: str | None = None,
    model_set_cap: int = MODEL_SET_CAP,
    service_fn: Any = None,
) -> SetIndexSchedulingPolicy:
    return SetIndexSchedulingPolicy(
        setup_cost_sigma=setup_cost_sigma,
        fair_share_target=10.0,
        fairness_half_life_seconds=600.0,
        resident_model=resident_model,
        model_set_cap=model_set_cap,
        weight_fn=lambda item: float(item.workflow_id.rsplit("-", 1)[-1]),
        service_fn=service_fn,
    )


def _select(
    policy: SetIndexSchedulingPolicy,
    items: list[WorkflowItem],
    batch_size: int,
) -> list[str]:
    return policy.select_batch(items, {}, batch_size)


def test_prefers_same_model_when_setup_outweighs_weight() -> None:
    """A same-model batch beats a higher-weight off-model batch when the setup
    charge outweighs the weight difference."""
    # Resident model "A". Two A items (weights 5, 4) and one B item (weight 9).
    # batch_size=2. Same-model batch: W=9, T=base+0. Off-model batch (A+B):
    # W=14, T=base+sigma. With sigma large, same-model wins.
    policy = _policy(setup_cost_sigma=100.0, resident_model="A")
    items = [
        _item("w-5", "A", 5.0),
        _item("w-4", "A", 4.0),
        _item("w-9", "B", 9.0),
    ]
    picked = _select(policy, items, 2)
    assert set(picked) == {"w-5", "w-4"}


def test_prefers_off_model_when_weight_outweighs_setup() -> None:
    """The reverse: with a small setup charge, the higher-weight off-model
    batch wins."""
    policy = _policy(setup_cost_sigma=0.1, resident_model="A")
    items = [
        _item("w-5", "A", 5.0),
        _item("w-4", "A", 4.0),
        _item("w-9", "B", 9.0),
    ]
    picked = _select(policy, items, 2)
    assert set(picked) == {"w-9", "w-5"}


def test_sigma_zero_reduces_to_weight_ranked_selection() -> None:
    """With sigma=0 the setup term vanishes; selection is purely by weight."""
    policy = _policy(setup_cost_sigma=0.0, resident_model="A")
    items = [
        _item("w-1", "B", 1.0),
        _item("w-9", "C", 9.0),
        _item("w-5", "A", 5.0),
    ]
    picked = _select(policy, items, 2)
    assert picked == ["w-9", "w-5"]


def test_model_set_enumeration_cap_is_respected() -> None:
    """The cap bounds the size of distinct-model sets enumerated per call."""
    # Resident A. Two non-resident models B and C, each with one item.
    # With cap=1 only singletons {A}, {B}, {C} are enumerated, so a 2-item
    # batch cannot combine B and C. With cap=2 the pair {B, C} is enumerated
    # and both are selected.
    items = [
        _item("w-5", "B", 5.0),
        _item("w-4", "C", 4.0),
    ]
    capped = _policy(setup_cost_sigma=0.0, resident_model="A", model_set_cap=1)
    assert len(_select(capped, items, 2)) == 1
    uncapped = _policy(setup_cost_sigma=0.0, resident_model="A", model_set_cap=2)
    assert set(_select(uncapped, items, 2)) == {"w-5", "w-4"}


def test_registered_as_set_index() -> None:
    assert "set_index" in SCHEDULING_POLICIES
    policy = create_scheduling_policy(
        "set_index",
        setup_cost_sigma=1.0,
        fair_share_target=10.0,
        fairness_half_life_seconds=600.0,
    )
    assert isinstance(policy, SetIndexSchedulingPolicy)


def test_fresh_group_charges_every_model() -> None:
    """With no resident model, every distinct model in the batch is charged."""
    policy = _policy(setup_cost_sigma=1000.0, resident_model=None)
    items = [
        _item("w-15", "A", 15.0),
        _item("w-14", "A", 14.0),
        _item("w-20", "B", 20.0),
    ]
    # No resident model: a same-model pair (A+A) charges one setup, a mixed
    # pair (A+B) charges two. At large sigma the same-model pair wins despite
    # the lower total weight.
    picked = _select(policy, items, 2)
    assert set(picked) == {"w-15", "w-14"}


def test_high_weight_long_service_excluded_when_not_worth_it() -> None:
    """A high-weight item with a very long service time is excluded when
    including it would inflate the block beyond what its weight is worth.

    This is the max-cost correction: base(S) = max of member service times, so
    one long item sets the block length for everyone. Top-B by weight alone
    would wrongly include it.
    """
    # Resident model A. Three A items, batch_size=2.
    #   w-9:  weight 9,  service 100 (long)
    #   w-5:  weight 5,  service 1
    #   w-4:  weight 4,  service 1
    # Including w-9: block = 100, W = 9+5 = 14, rho = 14/100 = 0.14.
    # Excluding w-9: block = 1,   W = 5+4 = 9,  rho = 9/1   = 9.0.
    # The long item is excluded.
    policy = _policy(
        setup_cost_sigma=0.0,
        resident_model="A",
        service_fn=lambda item: {
            "w-9": 100.0,
            "w-5": 1.0,
            "w-4": 1.0,
        }[item.workflow_id],
    )
    items = [
        _item("w-9", "A", 9.0),
        _item("w-5", "A", 5.0),
        _item("w-4", "A", 4.0),
    ]
    picked = _select(policy, items, 2)
    assert set(picked) == {"w-5", "w-4"}


def test_high_weight_long_service_included_when_worth_it() -> None:
    """The same long item is included when its weight makes the inflated block
    worthwhile."""
    # Same setup but the long item's weight is much higher.
    #   w-90: weight 90, service 10
    #   w-5:  weight 5,  service 1
    #   w-4:  weight 4,  service 1
    # Including w-90: block = 10, W = 90+5 = 95, rho = 95/10 = 9.5.
    # Excluding w-90: block = 1,  W = 5+4 = 9,  rho = 9/1  = 9.0.
    # Including wins.
    policy = _policy(
        setup_cost_sigma=0.0,
        resident_model="A",
        service_fn=lambda item: {
            "w-90": 10.0,
            "w-5": 1.0,
            "w-4": 1.0,
        }[item.workflow_id],
    )
    items = [
        _item("w-90", "A", 90.0),
        _item("w-5", "A", 5.0),
        _item("w-4", "A", 4.0),
    ]
    # Including w-90: block = 10, W = 90+5 = 95, rho = 95/10 = 9.5.
    # Excluding w-90: block = 1,  W = 5+4 = 9,  rho = 9/1  = 9.0.
    # Including wins.
    picked = _select(policy, items, 2)
    assert set(picked) == {"w-90", "w-5"}


def _job(wid: str, model: str) -> Job:
    graph = _graph(model)
    return Job(
        request_id=wid,
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
                template_hash="h",
                varying_input_keys=(),
            )
        },
        config=LumilakeRequestConfig(
            priority=Priority.MEDIUM,
            user_id="u",
            principal_id="u",
        ),
    )


def _manager(policy: str) -> PriorityJobManager:
    return PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        policy=policy,
    )


@pytest.mark.asyncio
async def test_set_index_constructs_and_selects_via_manager() -> None:
    """The production construction path: PriorityJobManager(policy='set_index')
    must come up and select a batch. This is the path the matrix run found
    broken — the manager passes setup_cost_sigma=, which the policy must
    accept."""
    manager = _manager("set_index")
    await manager.enqueue(_job("r1", "A"))
    batch = await manager.select_batch(1)
    assert batch is not None
    assert batch.workflows[0].request_id == "r1"


@pytest.mark.asyncio
async def test_legacy_constructs_via_manager() -> None:
    """legacy must also be constructible through the manager's
    parameter names, so a mismatch like set_index's is caught for all
    policies."""
    for policy in ("legacy",):
        manager = _manager(policy)
        await manager.enqueue(_job("r1", "A"))
        batch = await manager.select_batch(1)
        assert batch is not None
        assert batch.workflows[0].request_id == "r1"


def test_agnostic_only_pool_returns_nonempty_batch() -> None:
    """A candidate pool of only model-agnostic items dispatches: the empty
    model set is always enumerated, so the agnostic items form a batch with
    zero setup charge."""
    policy = _policy(setup_cost_sigma=1.0, resident_model="A")
    items = [
        _cpu_item("c-5", 5.0),
        _cpu_item("c-4", 4.0),
    ]
    picked = _select(policy, items, 2)
    assert set(picked) == {"c-5", "c-4"}


def test_agnostic_only_pool_with_no_resident_model() -> None:
    """The exact path that produced the empty universe: no resident model and
    only agnostic candidates. Must still dispatch."""
    policy = _policy(setup_cost_sigma=1.0, resident_model=None)
    items = [
        _cpu_item("c-5", 5.0),
        _cpu_item("c-4", 4.0),
    ]
    picked = _select(policy, items, 2)
    assert set(picked) == {"c-5", "c-4"}


def test_all_agnostic_batch_beats_model_set_batch() -> None:
    """An all-agnostic batch pays no setup, so it can beat a model-set batch
    that does. With a large sigma, the agnostic batch is selected over a
    same-model batch."""
    policy = _policy(setup_cost_sigma=1000.0, resident_model="A")
    items = [
        _cpu_item("c-5", 5.0),
        _cpu_item("c-4", 4.0),
        _item("w-9", "A", 9.0),
        _item("w-8", "A", 8.0),
    ]
    # Agnostic pair: W=9, T=base+0. Same-model A pair: W=17, T=base+sigma.
    # With sigma large the agnostic pair wins despite the lower weight.
    picked = _select(policy, items, 2)
    assert set(picked) == {"c-5", "c-4"}
