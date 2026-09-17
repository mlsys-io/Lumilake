"""Named, committed, reproducible simulation scenarios.

Each scenario is a pair of a :class:`Workload` and a :class:`SimulationConfig`
that together expose a specific scheduling behaviour. They are reproducible
from a fixed seed.
"""

from dataclasses import dataclass

from tests.support.schedsim.runner import SimulationConfig
from tests.support.schedsim.workload import (
    ChainSpec,
    OpKind,
    OpMix,
    Workload,
    WorkloadParams,
    generate_workload,
)


@dataclass(frozen=True)
class Scenario:
    """A named workload + cluster configuration."""

    name: str
    workload: Workload
    config: SimulationConfig


def _workload_from_chains(
    chains: list[ChainSpec],
    arrival_times: list[float],
    params: WorkloadParams,
) -> Workload:
    return Workload(chains=chains, arrival_times=arrival_times, params=params)


def mixed_gpu_cpu(*, seed: int = 0) -> Scenario:
    """GPU-scarce cluster with concurrent CPU-only work.

    A long GPU chain occupies the only GPU group from t=0 to t=3; a second GPU
    chain arrives at t=1 and queues behind it. A CPU-only chain arrives at t=0
    and must be dispatched on the idle CPU group *while* the GPU is busy —
    capacity-aware selection skips the ineligible GPU partition and runs the
    CPU work instead of head-of-line blocking behind it.
    """
    chains = [
        ChainSpec(
            chain_id="gpu-long",
            user_id="u0",
            cpu_units=1,
            gpu_units=1,
            round_services=(3.0,),
            round_kinds=(OpKind.GPU,),
        ),
        ChainSpec(
            chain_id="gpu-queued",
            user_id="u0",
            cpu_units=1,
            gpu_units=1,
            round_services=(1.0,),
            round_kinds=(OpKind.GPU,),
        ),
        ChainSpec(
            chain_id="cpu",
            user_id="u1",
            cpu_units=1,
            gpu_units=0,
            round_services=(1.0, 1.0),
            round_kinds=(OpKind.CPU, OpKind.CPU),
        ),
    ]
    workload = _workload_from_chains(
        chains,
        [0.0, 1.0, 0.0],
        WorkloadParams(
            num_users=2,
            chains_per_user=2,
            op_mix=OpMix(gpu=0.5, db=0.0, cpu=0.5),
        ),
    )
    config = SimulationConfig(
        batch_size=1,
        cpu_groups=1,
        gpu_groups=1,
    )
    return Scenario(name="mixed_gpu_cpu", workload=workload, config=config)


def many_short_chains_one_long(*, seed: int = 0) -> Scenario:
    """Many short chains plus one long chain sharing the cluster."""
    workload = generate_workload(
        seed=seed,
        params=WorkloadParams(
            num_users=1,
            chains_per_user=5,
            p_stop=0.8,
            max_rounds=3,
            op_mix=OpMix(gpu=0.0, db=0.0, cpu=1.0),
            service_mean=1.0,
            service_scale=0.0,
            arrival_times=[0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    )
    # Force the last chain to be long (many rounds).
    long_chain = workload.chains[-1]
    chains = list(workload.chains)
    chains[-1] = ChainSpec(
        chain_id=long_chain.chain_id,
        user_id=long_chain.user_id,
        cpu_units=long_chain.cpu_units,
        gpu_units=long_chain.gpu_units,
        round_services=(1.0,) * 8,
        round_kinds=(OpKind.CPU,) * 8,
    )
    workload = _workload_from_chains(chains, workload.arrival_times, workload.params)
    config = SimulationConfig(
        batch_size=1,
        cpu_groups=2,
        gpu_groups=0,
    )
    return Scenario(name="many_short_chains_one_long", workload=workload, config=config)


def two_users_uneven_graph_size(*, seed: int = 0) -> Scenario:
    """Fairness scenario: same item count per user, very different per-item area.

    Both users submit the same number of chains, but one user's chains declare
    far more GPU demand, so resource-area is skewed even though item counts are
    equal.
    """
    workload = generate_workload(
        seed=seed,
        params=WorkloadParams(
            num_users=2,
            chains_per_user=2,
            p_stop=0.5,
            max_rounds=4,
            op_mix=OpMix(gpu=1.0, db=0.0, cpu=0.0),
            service_mean=1.0,
            service_scale=0.0,
            arrival_times=[0.0, 0.0, 0.0, 0.0],
        ),
    )
    chains = [
        ChainSpec(
            chain_id=chain.chain_id,
            user_id=chain.user_id,
            cpu_units=chain.cpu_units,
            gpu_units=8 if chain.user_id == "user-1" else chain.gpu_units,
            round_services=chain.round_services,
            round_kinds=chain.round_kinds,
        )
        for chain in workload.chains
    ]
    workload = _workload_from_chains(chains, workload.arrival_times, workload.params)
    config = SimulationConfig(
        batch_size=1,
        cpu_groups=2,
        gpu_groups=2,
        gpu_units_per_worker=1,
    )
    return Scenario(
        name="two_users_uneven_graph_size", workload=workload, config=config
    )
