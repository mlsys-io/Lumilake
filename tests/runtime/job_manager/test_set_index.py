"""Tests for the set_index scheduling policy."""

from typing import Any, cast

from lumilake_server.runtime.job_manager.base import WorkflowItem
from lumilake_server.runtime.job_manager.policies import (
    SCHEDULING_POLICIES,
    create_scheduling_policy,
)
from lumilake_server.runtime.job_manager.policies.set_index import (
    MODEL_SET_CAP,
    SetIndexSchedulingPolicy,
)
from lumilake_server.runtime.protocol import LumilakeRequestConfig, Priority
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


def _policy(
    sigma: float,
    *,
    resident_model: str | None = None,
    model_set_cap: int = MODEL_SET_CAP,
) -> SetIndexSchedulingPolicy:
    return SetIndexSchedulingPolicy(
        sigma=sigma,
        fair_share_target=10.0,
        fairness_half_life_seconds=600.0,
        resident_model=resident_model,
        model_set_cap=model_set_cap,
        weight_fn=lambda item: float(item.workflow_id.rsplit("-", 1)[-1]),
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
    policy = _policy(sigma=100.0, resident_model="A")
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
    policy = _policy(sigma=0.1, resident_model="A")
    items = [
        _item("w-5", "A", 5.0),
        _item("w-4", "A", 4.0),
        _item("w-9", "B", 9.0),
    ]
    picked = _select(policy, items, 2)
    assert set(picked) == {"w-9", "w-5"}


def test_sigma_zero_reduces_to_weight_ranked_selection() -> None:
    """With sigma=0 the setup term vanishes; selection is purely by weight."""
    policy = _policy(sigma=0.0, resident_model="A")
    items = [
        _item("w-1", "B", 1.0),
        _item("w-9", "C", 9.0),
        _item("w-5", "A", 5.0),
    ]
    picked = _select(policy, items, 2)
    assert picked == ["w-9", "w-5"]


def test_model_set_enumeration_cap_is_respected() -> None:
    """The cap bounds the distinct-model sets enumerated per call."""
    policy = _policy(sigma=1.0, resident_model="A", model_set_cap=1)
    items = [
        _item("w-1", "A", 1.0),
        _item("w-2", "B", 2.0),
    ]
    # With cap=1 only the resident model alone is enumerated, so only A items
    # are selectable even though B has higher weight.
    picked = _select(policy, items, 1)
    assert picked == ["w-1"]


def test_registered_as_set_index() -> None:
    assert "set_index" in SCHEDULING_POLICIES
    policy = create_scheduling_policy(
        "set_index",
        sigma=1.0,
        fair_share_target=10.0,
        fairness_half_life_seconds=600.0,
    )
    assert isinstance(policy, SetIndexSchedulingPolicy)


def test_fresh_group_charges_every_model() -> None:
    """With no resident model, every distinct model in the batch is charged."""
    policy = _policy(sigma=1000.0, resident_model=None)
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
