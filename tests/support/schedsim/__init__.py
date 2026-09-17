"""Discrete-event scheduling simulation harness for Lumilake policies.

The harness drives the real :class:`PriorityJobManager` selection path against
a synthetic workload on a simulated worker pool, under a deterministic virtual
clock. It exists to evaluate scheduling policies; the metrics it computes are
the primary output.
"""

from tests.support.schedsim.clock import VirtualClock
from tests.support.schedsim.metrics import (
    SimulationMetrics,
    compute_metrics,
    jains_index,
)
from tests.support.schedsim.pool import WorkerPool
from tests.support.schedsim.runner import SimulationResult, run_simulation
from tests.support.schedsim.scenarios import (
    many_short_chains_one_long,
    mixed_gpu_cpu,
    two_users_uneven_graph_size,
)
from tests.support.schedsim.workload import (
    ChainSpec,
    OpMix,
    Workload,
    WorkloadParams,
    generate_workload,
)

__all__ = [
    "ChainSpec",
    "OpMix",
    "SimulationMetrics",
    "SimulationResult",
    "VirtualClock",
    "Workload",
    "WorkloadParams",
    "WorkerPool",
    "compute_metrics",
    "generate_workload",
    "jains_index",
    "many_short_chains_one_long",
    "mixed_gpu_cpu",
    "run_simulation",
    "two_users_uneven_graph_size",
]
