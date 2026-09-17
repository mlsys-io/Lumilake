"""Metrics computed from a completed simulation run.

The primary objective is chain end-to-end latency (first round enqueued to last
round finished). Fairness is measured with Jain's index in resource-area
(demand x duration), where the 4-D ``HardwareRequirements`` vector is collapsed
to a scalar via dominant-resource share.
"""

import math
from dataclasses import dataclass, field


def _percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile of an ascending-sorted list."""
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(p / 100.0 * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]


def jains_index(values: list[float]) -> float:
    """Jain's fairness index over ``values`` (1.0 = perfectly fair)."""
    if not values:
        return 1.0
    total = sum(values)
    if total <= 0.0:
        return 1.0
    n = len(values)
    return (total * total) / (n * sum(v * v for v in values))


@dataclass
class SimulationResult:
    """Raw per-chain and per-user observations from a run."""

    chain_latencies: dict[str, float] = field(default_factory=dict)
    chain_service_times: dict[str, float] = field(default_factory=dict)
    chain_areas: dict[str, float] = field(default_factory=dict)
    user_areas: dict[str, float] = field(default_factory=dict)
    worker_busy_seconds: float = 0.0
    total_worker_seconds: float = 0.0
    bubble_seconds: float = 0.0


@dataclass
class SimulationMetrics:
    """Computed metrics for a simulation run."""

    p50_latency: float
    p95_latency: float
    per_chain_slowdown: list[float]
    worker_utilization: float
    bubble_seconds: float
    jains_users: float
    jains_chains: float


def compute_metrics(result: SimulationResult) -> SimulationMetrics:
    """Compute all metrics from a :class:`SimulationResult`."""
    latencies = sorted(result.chain_latencies.values())
    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)

    slowdown = [
        result.chain_latencies[chain_id] / service
        for chain_id, service in result.chain_service_times.items()
        if service > 0.0
    ]

    utilization = (
        result.worker_busy_seconds / result.total_worker_seconds
        if result.total_worker_seconds > 0.0
        else 0.0
    )

    return SimulationMetrics(
        p50_latency=p50,
        p95_latency=p95,
        per_chain_slowdown=slowdown,
        worker_utilization=utilization,
        bubble_seconds=result.bubble_seconds,
        jains_users=jains_index(list(result.user_areas.values())),
        jains_chains=jains_index(list(result.chain_areas.values())),
    )
