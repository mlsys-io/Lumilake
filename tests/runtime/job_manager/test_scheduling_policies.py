"""Every registered scheduling policy against one shared workload."""

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from lumilake_server.runtime.capacity import FreeCapacity
from lumilake_server.runtime.job_manager.base import Job, WorkflowItem
from lumilake_server.runtime.job_manager.cost import estimate_area
from lumilake_server.runtime.job_manager.policies import (
    SCHEDULING_POLICIES,
    CohortPackSchedulingPolicy,
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
    if kind == "python":
        return RuntimeOp(
            node_id=node_id,
            dependencies=dependencies,
            task_type="python",
            backend="python",
            model="python",
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


def _config(
    user_id: str, chain_id: str | None, chain_round: int = 0
) -> LumilakeRequestConfig:
    return LumilakeRequestConfig(
        priority=Priority.MEDIUM,
        user_id=user_id,
        principal_id=user_id,
        chain_id=chain_id,
        chain_round=chain_round,
    )


def _item(
    request_id: str,
    kinds: list[str],
    enqueued_at: float,
    user_id: str = "u1",
    chain_id: str | None = None,
    chain_round: int = 0,
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
        template_hash="-".join(kinds),
        varying_input_keys=(),
        runtime_graph=graph,
        data_profile_graph=graph,
        dsl_graph=cast(Any, object()),
        config=_config(user_id, chain_id, chain_round),
        enqueued_at=enqueued_at,
    )


def _job(
    request_id: str,
    user_id: str,
    requires_gpu: bool = False,
    kinds: list[str] | None = None,
    chain_id: str | None = None,
    chain_round: int = 0,
    template_hash: str | None = None,
) -> Job:
    graph = _graph(kinds or ["cpu"])
    return Job(
        requires_gpu={"g": requires_gpu},
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
                template_hash=template_hash or f"hash-{request_id}",
                varying_input_keys=(),
            )
        },
        config=_config(user_id, chain_id, chain_round),
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
    # Every group is template-pure and the groups sit in different units; the
    # oldest unit goes alone, and with no cost constants the fill never merges
    # another unit, whatever the batch size.
    "cohort_pack": ["agent"],
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
    # cohort_pack keys templates by template hash, which differs between the two
    # requests, and with no cost constants it does not mix templates in a batch.
    if policy == "cohort_pack":
        assert {i.request_id for i in batch.workflows} == {"r1"}
    else:
        assert {i.request_id for i in batch.workflows} == {"r1", "r2"}


def _round_item(
    request_id: str,
    kinds: list[str],
    enqueued_at: float,
    chain_id: str,
    chain_round: int,
) -> WorkflowItem:
    return _item(
        request_id,
        kinds,
        enqueued_at,
        chain_id=chain_id,
        chain_round=chain_round,
    )


def _cohort_select(
    candidates: list[WorkflowItem],
    batch_size: int,
    *,
    history: list[WorkflowItem] | None = None,
    free_lanes: int | None = None,
    fixed_seconds: float = 0.0,
    mix_seconds: float = 0.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    patience: int = 8,
) -> list[str]:
    impl = CohortPackSchedulingPolicy(
        fixed_seconds=fixed_seconds,
        mix_seconds=mix_seconds,
        beta=beta,
        gamma=gamma,
        patience=patience,
    )
    impl.on_commit(history or [])
    impl.observe_free_lanes(free_lanes)
    selected = impl.select_batch(candidates, {}, batch_size)
    return [wid.removeprefix("wf-") for wid in selected]


def test_cohort_pack_keys_templates_by_template_hash() -> None:
    # Two slices of one round differ in runtime-graph node ids but share the
    # template hash, so the round is one template and one unit; a round with a
    # different template hash is a different template and is not merged in.
    first = _round_item("a0", ["cpu"], 1.0, "c1", 0)
    second = _round_item("a1", ["cpu"], 1.0, "c1", 0)
    second.runtime_graph = RuntimeGraph(
        nodes={
            f"round_0__slice_1__{op.node_id}": op
            for op in second.runtime_graph.nodes.values()
        },
        node_order=[f"round_0__slice_1__{n}" for n in second.runtime_graph.node_order],
        output_node_map=second.runtime_graph.output_node_map,
    )
    other = _round_item("b0", ["cpu"], 2.0, "c2", 0)
    other.template_hash = "other"
    assert set(first.runtime_graph.nodes) != set(second.runtime_graph.nodes)
    assert _cohort_select([first, second, other], 10) == ["a0", "a1"]
    assert _cohort_select([other, first, second], 10, history=[]) == ["a0", "a1"]
    # With a large fixed cost the other template merges in; with a large mixing
    # cost it does not.
    assert _cohort_select(
        [first, second, other], 10, fixed_seconds=100.0, mix_seconds=1.0
    ) == ["a0", "a1", "b0"]
    assert _cohort_select(
        [first, second, other], 10, fixed_seconds=1.0, mix_seconds=100.0
    ) == ["a0", "a1"]


def test_cohort_pack_first_round_cuts_a_template_into_balanced_units() -> None:
    # Four same-template chains of width 2 with batch size 4: two units of two
    # chains; the anchor is the unit of the oldest chain.
    candidates = [
        _round_item(f"{name}{i}", ["cpu"], float(order), name, 0)
        for order, name in enumerate(["c1", "c2", "c3", "c4"])
        for i in range(2)
    ]
    assert _cohort_select(candidates, 4) == ["c10", "c11", "c20", "c21"]


def test_cohort_pack_units_stay_template_pure_without_a_cost_gain() -> None:
    cpu = [_round_item("a0", ["cpu"], 1.0, "c1", 0)]
    gpu = [_round_item("b0", ["gpu"], 2.0, "c2", 0)]
    assert _cohort_select(cpu + gpu, 10) == ["a0"]
    assert _cohort_select(gpu + cpu, 10) == ["a0"]


def test_cohort_pack_fill_merges_rounds_when_fixed_cost_dominates() -> None:
    cpu = [_round_item("a0", ["cpu"], 1.0, "c1", 0)]
    gpu = [_round_item("b0", ["gpu"], 2.0, "c2", 0)]
    # Fixed cost per batch far above the mixing cost: one batch of both.
    assert _cohort_select(cpu + gpu, 10, fixed_seconds=100.0, mix_seconds=1.0) == [
        "a0",
        "b0",
    ]
    # Mixing dearer than a separate batch: the anchor goes alone.
    assert _cohort_select(cpu + gpu, 10, fixed_seconds=1.0, mix_seconds=100.0) == ["a0"]


def test_cohort_pack_waits_for_mates_then_releases_after_patience() -> None:
    history = [
        _round_item("a", ["cpu"], 0.0, "c1", 0),
        _round_item("b", ["cpu"], 0.0, "c2", 0),
    ]
    ready = _round_item("a1", ["cpu"], 5.0, "c1", 1)
    other = _round_item("z0", ["gpu"], 6.0, "c9", 0)
    # c2 went through the same template and is one round behind, so c1's unit is
    # incomplete: the complete unit of the other chain goes first.
    assert _cohort_select([ready, other], 10, history=history) == ["z0"]
    ready.miss_count = 9
    assert _cohort_select([ready, other], 10, history=history, patience=8) == ["a1"]


def test_cohort_pack_spreads_a_unit_over_idle_lanes() -> None:
    members = [_round_item(f"a{i}", ["cpu"], 1.0, "c1", 0) for i in range(6)]
    assert len(_cohort_select(members, 10, fixed_seconds=0.5, gamma=0.0)) == 6
    assert (
        len(_cohort_select(members, 10, free_lanes=1, fixed_seconds=0.5, gamma=0.0))
        == 6
    )
    assert (
        len(_cohort_select(members, 10, free_lanes=3, fixed_seconds=0.5, gamma=0.0))
        == 2
    )
    # A large fixed cost makes the extra batches not worth their lane-time.
    assert (
        len(_cohort_select(members, 10, free_lanes=3, fixed_seconds=500.0, gamma=1.0))
        == 6
    )


def test_cohort_pack_does_not_spread_onto_lanes_other_units_will_take() -> None:
    a = [_round_item(f"a{i}", ["cpu"], 1.0, "c1", 0) for i in range(6)]
    b = [_round_item("b0", ["gpu"], 2.0, "c2", 0)]
    # The other unit takes the second lane, so the first unit is not split.
    assert (
        len(
            _cohort_select(
                a + b, 10, free_lanes=2, fixed_seconds=0.5, mix_seconds=100.0, gamma=0.0
            )
        )
        == 6
    )
    assert (
        len(
            _cohort_select(
                a, 10, free_lanes=2, fixed_seconds=0.5, mix_seconds=100.0, gamma=0.0
            )
        )
        == 3
    )


def test_free_lane_hook_defaults_to_a_no_op() -> None:
    impl = create_scheduling_policy("fifo")
    impl.observe_free_lanes(3)
    assert impl.select_batch(list(CANDIDATES), {}, 2) == ["wf-agent", "wf-gpu"]


class _LaneRecorder(CohortPackSchedulingPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[int | None] = []

    def observe_free_lanes(self, free_lanes: int | None) -> None:
        self.seen.append(free_lanes)
        super().observe_free_lanes(free_lanes)


async def _lanes_seen(
    *,
    requires_gpu: bool,
    cpu_workers: int,
    gpu_workers: int,
    cpu_group_size: int = 1,
    gpu_group_size: int = 1,
    ok_workers: frozenset[str] | None = None,
    python_workers: frozenset[str] | None = None,
    kinds: list[str] | None = None,
) -> int | None:
    """What the policy is told when a one-job partition is reserved."""
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        policy="cohort_pack",
        cpu_worker_group_size=cpu_group_size,
        gpu_worker_group_size=gpu_group_size,
        worker_meets_hardware=lambda profile, _hardware: bool(profile["ok"]),
    )
    recorder = _LaneRecorder()
    manager._policy_impl = recorder
    await manager.enqueue(_job("r1", "u1", requires_gpu, kinds))
    cpu = tuple(f"c{i}" for i in range(cpu_workers))
    gpu = tuple(f"g{i}" for i in range(gpu_workers))
    profiles = (
        {
            w: {
                "ok": ok_workers is None or w in ok_workers,
                "supported_task_types": (
                    ["python"] if python_workers and w in python_workers else []
                ),
            }
            for w in (*cpu, *gpu)
        }
        if ok_workers is not None or python_workers is not None
        else {}
    )
    reservation = await manager.reserve_batch(
        4,
        capacity=FreeCapacity(
            cpu_worker_ids=cpu, gpu_worker_ids=gpu, profiles=profiles
        ),
    )
    assert reservation is not None
    assert len(recorder.seen) == 1
    return recorder.seen[0]


@pytest.mark.asyncio
async def test_free_lanes_of_a_cpu_only_partition_count_cpu_groups() -> None:
    lanes = await _lanes_seen(
        requires_gpu=False, cpu_workers=5, gpu_workers=9, cpu_group_size=2
    )
    assert lanes == 2


@pytest.mark.asyncio
async def test_free_lanes_of_a_gpu_partition_are_bounded_by_its_cpu_group() -> None:
    # A GPU dispatch also claims a CPU group: one CPU worker, one lane.
    assert await _lanes_seen(requires_gpu=True, cpu_workers=1, gpu_workers=4) == 1
    # Enough CPU workers: the GPU groups bound it instead.
    assert (
        await _lanes_seen(
            requires_gpu=True, cpu_workers=6, gpu_workers=5, gpu_group_size=2
        )
        == 2
    )


@pytest.mark.asyncio
async def test_free_lanes_count_only_workers_meeting_the_hardware() -> None:
    ok = frozenset({"c0", "c1", "g0", "g1", "g2"})
    # Four GPU workers idle but only three meet the hardware; two CPU workers.
    assert (
        await _lanes_seen(
            requires_gpu=True, cpu_workers=3, gpu_workers=4, ok_workers=ok
        )
        == 2
    )
    # CPU-only: two of the three CPU workers qualify.
    assert (
        await _lanes_seen(
            requires_gpu=False, cpu_workers=3, gpu_workers=4, ok_workers=ok
        )
        == 2
    )


@pytest.mark.asyncio
async def test_free_lanes_count_only_cpu_workers_advertising_the_task_type() -> None:
    # Three CPU groups are free and meet the hardware, but only one advertises
    # python, so only one dispatch of a python job can start.
    assert (
        await _lanes_seen(
            requires_gpu=False,
            cpu_workers=3,
            gpu_workers=0,
            python_workers=frozenset({"c1"}),
            kinds=["python"],
        )
        == 1
    )
    # The same capacity serves three dispatches of a job with no gated type.
    assert (
        await _lanes_seen(
            requires_gpu=False,
            cpu_workers=3,
            gpu_workers=0,
            python_workers=frozenset({"c1"}),
        )
        == 3
    )
    # No worker advertises it: no dispatch can start.
    assert (
        await _lanes_seen(
            requires_gpu=False,
            cpu_workers=3,
            gpu_workers=0,
            python_workers=frozenset(),
            kinds=["python"],
        )
        == 0
    )


@pytest.mark.asyncio
async def test_free_lanes_with_a_zero_cpu_group_follow_the_cpu_only_nodes() -> None:
    # (a) GPU-only work claims no CPU worker: the GPU groups bound the count.
    assert (
        await _lanes_seen(
            requires_gpu=True,
            cpu_workers=0,
            gpu_workers=4,
            cpu_group_size=0,
            kinds=["gpu"],
        )
        == 4
    )
    # (b) A data_retrieval node still claims one CPU worker.
    assert (
        await _lanes_seen(
            requires_gpu=True,
            cpu_workers=1,
            gpu_workers=4,
            cpu_group_size=0,
            kinds=["gpu", "agent"],
        )
        == 1
    )
    # (c) ... so with no free CPU worker no dispatch can start.
    assert (
        await _lanes_seen(
            requires_gpu=True,
            cpu_workers=0,
            gpu_workers=4,
            cpu_group_size=0,
            kinds=["gpu", "agent"],
        )
        == 0
    )


@pytest.mark.asyncio
async def test_free_lanes_are_unknown_without_capacity() -> None:
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer), policy="cohort_pack"
    )
    recorder = _LaneRecorder()
    manager._policy_impl = recorder
    await manager.enqueue(_job("r1", "u1"))
    assert await manager.reserve_batch(4) is not None
    assert recorder.seen == [None]


def test_cohort_pack_recuts_the_cohort_when_a_chain_joins_late() -> None:
    # Units are balanced cuts of the current cohort in chain order, not
    # remembered: chain b, arriving after a and c were first seen together,
    # sorts between them and moves the cut.
    a, c, d = (
        _round_item(f"{name}0", ["cpu"], float(order), name, 0)
        for order, name in enumerate(["a", "c", "d"], start=1)
    )
    b = _round_item("b0", ["cpu"], 4.0, "b", 0)
    assert _cohort_select([a, c, d], 2) == ["a0", "c0"]
    assert _cohort_select([a, b, c, d], 2) == ["a0", "b0"]


def test_cohort_pack_never_mixes_templates_within_one_request() -> None:
    # One ordinary request (no chain id) with two workflow entries of different
    # templates is one (chain, round) but two template groups.
    first = _item("r", ["cpu"], 1.0)
    second = _item("r", ["gpu"], 1.0)
    second.workflow_id = "wf-r2"
    assert first.template_hash != second.template_hash
    assert _cohort_select([first, second], 10) == ["r"]
    assert _cohort_select([second, first], 10) == ["r"]


async def _starved_unit_dispatch(*, starvation_limit: int, misses: int) -> set[str]:
    """Request ids dispatched when an incomplete unit's member has missed
    ``misses`` dispatches and a complete unit of another template is waiting."""
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        policy="cohort_pack",
        starvation_limit=starvation_limit,
    )
    manager._policy_impl = CohortPackSchedulingPolicy(patience=8)
    # Chains a and b both ran round 0; only a is ready for round 1, so its unit
    # is waiting for b.
    manager._policy_impl.on_commit(
        [
            _round_item("a0", ["cpu"], 0.0, "a", 0),
            _round_item("b0", ["cpu"], 0.0, "b", 0),
        ]
    )
    (waiting,) = await manager.enqueue(
        _job("a1", "u1", chain_id="a", chain_round=1, template_hash="round1")
    )
    for request_id in ("z0", "z1"):
        await manager.enqueue(
            _job(request_id, "u1", chain_id="z", chain_round=0, template_hash="other")
        )
    waiting.miss_count = misses
    reservation = await manager.reserve_batch(2)
    assert reservation is not None
    return {item.request_id for item in reservation.selection.workflows}


@pytest.mark.asyncio
async def test_global_starvation_limit_overrides_cohort_patience() -> None:
    # Patience is 8, but the global limit of 3 pins the starved member first.
    assert "a1" in await _starved_unit_dispatch(starvation_limit=3, misses=3)
    # Below the global limit the complete unit goes first and a1 keeps waiting.
    assert await _starved_unit_dispatch(starvation_limit=3, misses=2) == {"z0", "z1"}


@pytest.mark.asyncio
async def test_cohort_patience_governs_below_a_high_global_limit() -> None:
    for misses in (3, 8):
        dispatched = await _starved_unit_dispatch(starvation_limit=1000, misses=misses)
        assert dispatched == {"z0", "z1"}
    # More than PATIENCE misses releases the incomplete unit's member.
    assert "a1" in await _starved_unit_dispatch(starvation_limit=1000, misses=9)


def test_cohort_pack_waits_on_an_ended_chain_only_while_other_work_is_ready() -> None:
    # Chain b ended after round 0 but still counts as a mate for chain a.
    history = [
        _round_item("a0", ["cpu"], 0.0, "a", 0),
        _round_item("b0", ["cpu"], 0.0, "b", 0),
    ]
    waiting = _round_item("a1", ["gpu"], 5.0, "a", 1)
    other = _round_item("z0", ["cpu", "cpu"], 6.0, "z", 0)
    # (a) The only ready unit is incomplete, yet it is dispatched at once.
    assert _cohort_select([waiting], 10, history=history) == ["a1"]
    # (b) A complete unit of another template goes first until patience passes.
    assert _cohort_select([waiting, other], 10, history=history) == ["z0"]
    waiting.miss_count = 9
    assert _cohort_select([waiting, other], 10, history=history) == ["a1"]


def test_cohort_pack_late_older_commit_keeps_the_last_round_width() -> None:
    impl = CohortPackSchedulingPolicy()
    impl.on_commit([_round_item(f"a1{i}", ["cpu"], 0.0, "a", 1) for i in range(3)])
    impl.on_commit([_round_item("a0", ["cpu"], 0.0, "a", 0)])
    assert impl._width["a"] == 3
    assert impl._last_round["a"] == 1
    # Another commit of the same latest round still takes the widest count.
    impl.on_commit([_round_item("a1x", ["cpu"], 0.0, "a", 1)])
    assert impl._width["a"] == 3
    impl.on_commit([_round_item(f"z{r}", ["cpu"], 0.0, "z", r) for r in (0, 1)])
    # Chain z is ready on round 2 with a's template history; a is its mate with
    # its last round's 3 slices, so with batch size 3 the cohort is cut into
    # [a] and [z] and z's unit is complete. The older z2 beats the complete
    # unit y0; with a's width corrupted to 1 the cohort [a, z] would be one
    # incomplete unit and y0 would win.
    z2 = _round_item("z2", ["cpu", "cpu"], 1.0, "z", 2)
    y0 = _round_item("y0", ["gpu"], 2.0, "y", 0)
    assert [w.removeprefix("wf-") for w in impl.select_batch([z2, y0], {}, 3)] == ["z2"]


def test_cohort_pack_mate_width_is_the_largest_batch_of_its_last_round() -> None:
    impl = CohortPackSchedulingPolicy()
    impl.on_commit([_round_item(f"a1{i}", ["cpu"], 0.0, "a", 1) for i in range(2)])
    impl.on_commit([_round_item(f"a1{i}", ["cpu"], 0.0, "a", 1) for i in range(2, 5)])
    assert impl._width["a"] == 3
    # A new round replaces the width.
    impl.on_commit([_round_item("a2", ["cpu"], 0.0, "a", 2)])
    assert impl._width["a"] == 1


def test_cohort_pack_oversized_unit_goes_out_over_several_dispatches() -> None:
    # One chain's ready round has 5 slices but a dispatch takes at most 3: the
    # unit's first 3 slices in (enqueue time, slice index, workflow id) order go
    # first, and the remaining 2 are selected next.
    impl = CohortPackSchedulingPolicy()
    slices = [
        _round_item(f"s{i}", ["cpu"], enqueued, "c", 0)
        for i, enqueued in enumerate([3.0, 1.0, 5.0, 2.0, 4.0])
    ]
    first = impl.select_batch(list(slices), {}, 3)
    assert [w.removeprefix("wf-") for w in first] == ["s1", "s3", "s0"]
    left = [item for item in slices if item.workflow_id not in set(first)]
    second = impl.select_batch(left, {}, 3)
    assert [w.removeprefix("wf-") for w in second] == ["s4", "s2"]


def test_cohort_pack_indexes_chains_by_last_committed_round() -> None:
    impl = CohortPackSchedulingPolicy()
    impl.on_commit([_round_item("a0", ["cpu"], 0.0, "a", 0)])
    assert impl._chains_at == {0: {"a"}}
    impl.on_commit([_round_item("a1", ["cpu"], 1.0, "a", 1)])
    assert impl._chains_at == {0: set(), 1: {"a"}}
    # An older round committed late does not move the chain back.
    impl.on_commit([_round_item("a0b", ["cpu"], 2.0, "a", 0)])
    assert impl._chains_at == {0: set(), 1: {"a"}}
    impl.on_commit([_round_item("b0", ["cpu"], 3.0, "b", 0)])
    assert impl._chains_at == {0: {"b"}, 1: {"a"}}


def test_cohort_pack_does_not_merge_a_set_equal_to_the_unit_alone() -> None:
    # A combo whose value equals the unit's after rounding is not a gain.
    cpu = _round_item("a0", ["cpu"], 1.0, "c1", 0)
    gpu = _round_item("b0", ["gpu"], 2.0, "c2", 0)
    assert _cohort_select([cpu, gpu], 10, beta=1.0 - 1e-12) == ["a0"]


def test_cohort_pack_overdue_tie_is_independent_of_candidate_order() -> None:
    # One chain and round with two template groups, both incomplete (chain b is
    # a mate one round behind) and equally overdue.
    history = [
        _round_item("a0", ["cpu"], 0.0, "a", 0),
        _round_item("b0", ["cpu"], 0.0, "b", 0),
    ]
    first = _round_item("a1c", ["cpu"], 5.0, "a", 1)
    second = _round_item("a1g", ["gpu"], 5.0, "a", 1)
    first.miss_count = second.miss_count = 9
    assert _cohort_select([first, second], 10, history=history) == ["a1c"]
    assert _cohort_select([second, first], 10, history=history) == ["a1c"]


def test_cohort_pack_overdue_units_of_equal_misses_rank_by_chain_id() -> None:
    # Chains c and d are one round behind and not ready, so the ready units of
    # a and b (different templates) are both incomplete; both are overdue with
    # equal misses. The lexically smaller chain wins although b is older.
    history = [
        _round_item(f"{name}0", ["cpu"], 0.0, name, 0) for name in ("a", "b", "c", "d")
    ]
    a1 = _round_item("a1", ["gpu"], 6.0, "a", 1)
    b1 = _round_item("b1", ["cpu", "cpu"], 5.0, "b", 1)
    a1.miss_count = b1.miss_count = 9
    assert _cohort_select([a1, b1], 10, history=history) == ["a1"]
    assert _cohort_select([b1, a1], 10, history=history) == ["a1"]


def test_cohort_pack_overdue_chain_id_prefix_ranks_behind_its_extension() -> None:
    # Same construction with chains a and aa: a longer id ranks ahead of an id it
    # extends, although a is older.
    history = [
        _round_item(f"{name}0", ["cpu"], 0.0, name, 0) for name in ("a", "aa", "c", "d")
    ]
    a1 = _round_item("a1", ["gpu"], 5.0, "a", 1)
    aa1 = _round_item("aa1", ["cpu", "cpu"], 6.0, "aa", 1)
    a1.miss_count = aa1.miss_count = 9
    assert _cohort_select([a1, aa1], 10, history=history) == ["aa1"]
    assert _cohort_select([aa1, a1], 10, history=history) == ["aa1"]


def test_cohort_pack_anchor_is_the_most_passed_over_complete_unit() -> None:
    old = _round_item("o0", ["cpu"], 1.0, "c1", 0)
    young = _round_item("y0", ["gpu"], 2.0, "c2", 0)
    assert _cohort_select([old, young], 10) == ["o0"]
    young.miss_count = 2
    assert _cohort_select([old, young], 10) == ["y0"]


def test_cohort_pack_merges_templates_for_economy_of_scale_alone() -> None:
    cpu = [_round_item("a0", ["cpu"], 1.0, "c1", 0)]
    gpu = [_round_item("b0", ["gpu"], 2.0, "c2", 0)]
    assert _cohort_select(cpu + gpu, 10, beta=1.0) == ["a0"]
    assert _cohort_select(cpu + gpu, 10, beta=1.5) == ["a0"]
    assert _cohort_select(cpu + gpu, 10, beta=0.5) == ["a0", "b0"]
