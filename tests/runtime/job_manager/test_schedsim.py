"""Tests of the scheduling-simulation harness itself (not of the scheduler).

These verify the harness's own correctness: the clock is deterministic and
never sleeps on wall time, the same seed reproduces identical metrics, the
metrics are exact on a tiny hand-computable fixture, Jain's index behaves, and
bubble-seconds is zero when no idle worker has a runnable queued item.
"""

from unittest.mock import MagicMock

import pytest

from lumilake_server.runtime.job_manager.priority_queue import PriorityJobManager
from lumilake_server.runtime.optimizer.base import BaseOptimizer
from tests.support.schedsim.clock import VirtualClock
from tests.support.schedsim.metrics import (
    _percentile,
    compute_metrics,
    jains_index,
)
from tests.support.schedsim.runner import SimulationConfig, run_simulation
from tests.support.schedsim.scenarios import (
    many_short_chains_one_long,
    mixed_gpu_cpu,
    two_users_uneven_graph_size,
)
from tests.support.schedsim.workload import (
    ChainSpec,
    OpKind,
    OpMix,
    Workload,
    WorkloadParams,
    generate_workload,
)


def _make_manager(config: SimulationConfig) -> PriorityJobManager:
    return PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        cpu_worker_group_size=config.cpu_group_size,
        gpu_worker_group_size=config.gpu_group_size,
    )


# -- clock ------------------------------------------------------------------


def test_clock_is_deterministic_and_never_sleeps() -> None:
    clock = VirtualClock()
    assert clock.now() == 0.0
    clock.advance(1.5)
    clock.advance(2.5)
    assert clock.now() == 4.0
    assert clock.monotonic() == 4.0
    # Advancing is pure arithmetic; it must not depend on wall time.
    before = clock.now()
    clock.advance(0.0)
    assert clock.now() == before


def test_clock_rejects_backwards_advance() -> None:
    clock = VirtualClock()
    clock.advance(5.0)
    with pytest.raises(ValueError):
        clock.advance(-1.0)


# -- reproducibility --------------------------------------------------------


@pytest.mark.asyncio
async def test_same_seed_reproduces_identical_metrics() -> None:
    scenario = mixed_gpu_cpu(seed=42)
    first = await run_simulation(
        scenario.workload, _make_manager(scenario.config), scenario.config
    )
    second = await run_simulation(
        scenario.workload, _make_manager(scenario.config), scenario.config
    )
    assert compute_metrics(first) == compute_metrics(second)


# -- hand-computable fixture -------------------------------------------------


def _hand_fixture_workload() -> Workload:
    """3 chains, 2 CPU workers, batch_size=1, all CPU, service time 1.0 each.

    Chain A: 1 round, arrives t=0.
    Chain B: 1 round, arrives t=0.
    Chain C: 1 round, arrives t=0.

    With 2 CPU workers and batch_size=1, the scheduler dispatches A and B at
    t=0 (one per worker), both finishing at t=1. C waits until t=1, then runs
    on a freed worker from t=1 to t=2.
    """
    chains = [
        ChainSpec(
            chain_id="A",
            user_id="u0",
            cpu_units=1,
            gpu_units=0,
            round_services=(1.0,),
            round_kinds=(OpKind.CPU,),
        ),
        ChainSpec(
            chain_id="B",
            user_id="u0",
            cpu_units=1,
            gpu_units=0,
            round_services=(1.0,),
            round_kinds=(OpKind.CPU,),
        ),
        ChainSpec(
            chain_id="C",
            user_id="u0",
            cpu_units=1,
            gpu_units=0,
            round_services=(1.0,),
            round_kinds=(OpKind.CPU,),
        ),
    ]
    return Workload(
        chains=chains,
        arrival_times=[0.0, 0.0, 0.0],
        params=WorkloadParams(
            num_users=1,
            chains_per_user=3,
            op_mix=OpMix(gpu=0.0, db=0.0, cpu=1.0),
        ),
    )


@pytest.mark.asyncio
async def test_metrics_exact_on_hand_computable_fixture() -> None:
    workload = _hand_fixture_workload()
    config = SimulationConfig(
        batch_size=1,
        cpu_groups=2,
        gpu_groups=0,
        cpu_units_per_worker=1,
        gpu_units_per_worker=1,
    )
    result = await run_simulation(workload, _make_manager(config), config)
    metrics = compute_metrics(result)

    # Latencies: A=1, B=1, C=2. p50 = 1, p95 = 2 (nearest-rank).
    assert metrics.p50_latency == pytest.approx(1.0)
    assert metrics.p95_latency == pytest.approx(2.0)

    # Slowdown: A=1/1=1, B=1/1=1, C=2/1=2.
    assert sorted(metrics.per_chain_slowdown) == pytest.approx([1.0, 1.0, 2.0])

    # Utilization: 3 worker-seconds busy / (2 workers * 2 seconds) = 0.75.
    assert metrics.worker_utilization == pytest.approx(0.75)

    # Bubble-seconds: between t=0 and t=1, C is queued and 0 CPU workers are
    # idle (both busy), so no bubble. Between t=1 and t=2, C runs on one
    # worker and the other is idle with no queued item, so no bubble. Total 0.
    assert metrics.bubble_seconds == pytest.approx(0.0)

    # Jain over chains: all three chains have equal area (share 1/2 each,
    # service 1.0), so Jain = 1.0.
    assert metrics.jains_chains == pytest.approx(1.0)


# -- percentiles ------------------------------------------------------------


def test_percentile_is_nearest_rank() -> None:
    """Nearest-rank percentiles on a known list: p50=3, p95=5 for [1..5]."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(values, 50) == 3.0
    assert _percentile(values, 95) == 5.0
    # p0 is the minimum, p100 the maximum.
    assert _percentile(values, 0) == 1.0
    assert _percentile(values, 100) == 5.0


def test_percentile_empty_returns_zero() -> None:
    assert _percentile([], 50) == 0.0


# -- Jain's index -----------------------------------------------------------


def test_jains_index_perfect_split_is_one() -> None:
    assert jains_index([1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_jains_index_skewed_split_drops() -> None:
    assert jains_index([1.0, 1.0, 1.0, 1.0]) == pytest.approx(1.0)
    skewed = jains_index([1.0, 1.0, 1.0, 10.0])
    assert skewed < 1.0
    assert skewed > 0.0


@pytest.mark.asyncio
async def test_jains_users_skewed_when_areas_uneven() -> None:
    scenario = two_users_uneven_graph_size(seed=7)
    result = await run_simulation(
        scenario.workload, _make_manager(scenario.config), scenario.config
    )
    metrics = compute_metrics(result)
    # user-1 declares 8x the GPU demand of user-0, so resource-area is skewed
    # and Jain's index over users must be strictly below 1.0.
    assert metrics.jains_users < 1.0


# -- bubble-seconds ---------------------------------------------------------


@pytest.mark.asyncio
async def test_bubble_seconds_zero_when_no_idle_worker_has_runnable_item() -> None:
    # All-CPU workload with enough workers: every idle worker always has a
    # runnable queued item only while the queue is non-empty, and the queue
    # drains without idle workers sitting on runnable work.
    workload = generate_workload(
        seed=3,
        params=WorkloadParams(
            num_users=1,
            chains_per_user=2,
            p_stop=0.5,
            max_rounds=3,
            op_mix=OpMix(gpu=0.0, db=0.0, cpu=1.0),
            service_mean=1.0,
            service_scale=0.0,
            arrival_times=[0.0, 0.0],
        ),
    )
    config = SimulationConfig(
        batch_size=1,
        cpu_groups=4,
        gpu_groups=0,
    )
    result = await run_simulation(workload, _make_manager(config), config)
    assert result.bubble_seconds == pytest.approx(0.0)


# -- scenarios smoke --------------------------------------------------------


@pytest.mark.asyncio
async def test_mixed_gpu_cpu_skips_gpu_partition_for_cpu_work() -> None:
    scenario = mixed_gpu_cpu(seed=0)
    result = await run_simulation(
        scenario.workload, _make_manager(scenario.config), scenario.config
    )
    # A long GPU chain occupies the only GPU group from t=0 to t=3 while a CPU
    # chain arrives at t=0. Capacity-aware selection must skip the ineligible
    # GPU partition and run the CPU work on the idle CPU group concurrently,
    # so the CPU chain finishes before the GPU chain's service completes.
    assert result.chain_latencies["cpu"] < 3.0
    assert result.chain_latencies["gpu-long"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_many_short_chains_one_long_runs() -> None:
    scenario = many_short_chains_one_long(seed=0)
    result = await run_simulation(
        scenario.workload, _make_manager(scenario.config), scenario.config
    )
    assert len(result.chain_latencies) == len(scenario.workload.chains)


@pytest.mark.asyncio
async def test_run_drains_manager() -> None:
    scenario = mixed_gpu_cpu(seed=0)
    manager = _make_manager(scenario.config)
    await run_simulation(scenario.workload, manager, scenario.config)
    assert await manager.has_work() is False
