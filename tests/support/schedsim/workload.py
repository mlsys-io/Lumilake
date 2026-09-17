"""Synthetic workload generation for the scheduling simulation.

There is no trace corpus yet, so workloads are generated from explicit, named
hyperparameters with reasoned defaults. Every generator accepts a seed and is
fully reproducible: all per-round service times and op kinds are pre-drawn at
generation time. Chains emit real :class:`Job` objects that the real
:class:`PriorityJobManager` can enqueue.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any, cast

from lumilake_server.runtime.job_manager.base import Job
from lumilake_server.runtime.protocol import (
    HardwareRequirements,
    LumilakeRequestConfig,
    Priority,
)
from lumilake_server.runtime.request import WorkflowSliceMeta
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp


class OpKind(Enum):
    GPU = "gpu"
    DB = "db"
    CPU = "cpu"


@dataclass(frozen=True)
class OpMix:
    """Per-round op mix: fraction of GPU, DB, and pure-CPU ops."""

    gpu: float
    db: float
    cpu: float

    def __post_init__(self) -> None:
        total = self.gpu + self.db + self.cpu
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"OpMix fractions must sum to 1, got {total}")

    def sample(self, rng: random.Random) -> OpKind:
        roll = rng.random()
        if roll < self.gpu:
            return OpKind.GPU
        if roll < self.gpu + self.db:
            return OpKind.DB
        return OpKind.CPU


@dataclass(frozen=True)
class ChainSpec:
    """A single chain: user, declared hardware demand, and per-round shape."""

    chain_id: str
    user_id: str
    cpu_units: int
    gpu_units: int
    round_services: tuple[float, ...]
    round_kinds: tuple[OpKind, ...]
    memory_units: int = 0
    gpu_memory_units: int = 0

    @property
    def rounds(self) -> int:
        return len(self.round_services)

    @property
    def total_service_time(self) -> float:
        return sum(self.round_services)

    def dominant_share(
        self,
        cpu_capacity: int,
        gpu_capacity: int,
        memory_capacity: int = 0,
        gpu_memory_capacity: int = 0,
    ) -> float:
        """Dominant-resource share of this chain against cluster capacity.

        The share is the largest of the per-resource fractions across all four
        dimensions (cpu, memory, gpu, gpu_memory), matching the DRF accounting
        in the cost model.
        """
        cpu_share = self.cpu_units / cpu_capacity if cpu_capacity else 0.0
        gpu_share = self.gpu_units / gpu_capacity if gpu_capacity else 0.0
        memory_share = self.memory_units / memory_capacity if memory_capacity else 0.0
        gpu_memory_share = (
            self.gpu_memory_units / gpu_memory_capacity if gpu_memory_capacity else 0.0
        )
        return max(cpu_share, memory_share, gpu_share, gpu_memory_share)


@dataclass(frozen=True)
class WorkloadParams:
    """Named hyperparameters for workload generation, all overridable."""

    num_users: int = 4
    chains_per_user: int = 3
    # Geometric chain length: per-round STOP probability.
    p_stop: float = 0.3
    max_rounds: int = 10
    # Per-round op mix defaults: mostly GPU with some DB and CPU.
    op_mix: OpMix = OpMix(gpu=0.6, db=0.25, cpu=0.15)
    # Service time distribution (seconds).
    service_mean: float = 1.0
    service_scale: float = 0.3
    # Declared hardware demand per chain (cpu cores, gpu count).
    cpu_units: int = 4
    gpu_units: int = 1
    # Arrival process: Poisson rate, or a fixed list of arrival times.
    arrival_rate: float = 1.0
    arrival_times: list[float] | None = None
    priority: Priority = Priority.MEDIUM


@dataclass
class Workload:
    """A generated workload: the chains and their arrival times."""

    chains: list[ChainSpec]
    arrival_times: list[float]
    params: WorkloadParams

    @property
    def num_chains(self) -> int:
        return len(self.chains)


def _geometric_length(rng: random.Random, p_stop: float, max_rounds: int) -> int:
    """Draw a chain length from a geometric distribution with per-round stop
    probability ``p_stop``, capped at ``max_rounds``."""
    rounds = 1
    while rounds < max_rounds and rng.random() >= p_stop:
        rounds += 1
    return rounds


def _service_time(rng: random.Random, mean: float, scale: float) -> float:
    """Draw a per-round service time from a truncated normal distribution."""
    return max(0.1, rng.gauss(mean, scale))


def generate_workload(
    *,
    seed: int,
    params: WorkloadParams | None = None,
    chain_length: Callable[[random.Random], int] | None = None,
    service_time: Callable[[random.Random], float] | None = None,
) -> Workload:
    """Generate a reproducible synthetic workload.

    ``chain_length`` and ``service_time`` override the default geometric length
    and Gaussian service-time draws respectively.
    """
    params = params or WorkloadParams()
    rng = random.Random(seed)

    length_draw = chain_length or (
        lambda r: _geometric_length(r, params.p_stop, params.max_rounds)
    )
    service_draw = service_time or (
        lambda r: _service_time(r, params.service_mean, params.service_scale)
    )

    chains: list[ChainSpec] = []
    for user_idx in range(params.num_users):
        for chain_idx in range(params.chains_per_user):
            chain_id = f"u{user_idx}-c{chain_idx}"
            rounds = length_draw(rng)
            round_services = tuple(service_draw(rng) for _ in range(rounds))
            round_kinds = tuple(params.op_mix.sample(rng) for _ in range(rounds))
            chains.append(
                ChainSpec(
                    chain_id=chain_id,
                    user_id=f"user-{user_idx}",
                    cpu_units=params.cpu_units,
                    gpu_units=params.gpu_units,
                    round_services=round_services,
                    round_kinds=round_kinds,
                )
            )

    if params.arrival_times is not None:
        arrival_times = list(params.arrival_times)
    else:
        arrival_times = _poisson_arrivals(rng, params.arrival_rate, len(chains))

    return Workload(chains=chains, arrival_times=arrival_times, params=params)


def _poisson_arrivals(rng: random.Random, rate: float, count: int) -> list[float]:
    """Draw ``count`` arrival times from a Poisson process of ``rate`` per sec."""
    times: list[float] = []
    current = 0.0
    for _ in range(count):
        current += rng.expovariate(rate)
        times.append(current)
    return times


def chain_to_job(
    chain: ChainSpec,
    *,
    round_index: int,
    request_id: str,
    priority: Priority = Priority.MEDIUM,
) -> Job:
    """Build a real :class:`Job` for one round of a chain.

    Each round carries the chain's declared hardware demand and chain lineage
    (``chain_id`` / ``chain_round``) so the real scheduler treats it as part of
    the chain. The emitted graph's op reflects the round's sampled kind, so the
    real selection path (which classifies ops by backend/task_type) sees the
    same GPU/DB/CPU split the simulator bookkeeping does.
    """
    graph_name = f"{chain.chain_id}-r{round_index}"
    kind = chain.round_kinds[round_index]
    runtime_graph = _graph_for_kind(graph_name, kind)
    dsl_graph = SimpleNamespace(graph=object(), inputs={})
    return Job(
        request_id=request_id,
        runtime_graphs={graph_name: runtime_graph},
        data_profile_graphs={graph_name: runtime_graph},
        dsl_graphs={graph_name: cast(Any, dsl_graph)},
        workflow_slices={
            graph_name: WorkflowSliceMeta(
                public_graph_name=graph_name,
                slice_index=0,
                slice_start=0,
                slice_length=1,
                total_length=1,
                template_hash=f"hash-{graph_name}",
                varying_input_keys=(),
            )
        },
        config=LumilakeRequestConfig(
            priority=priority,
            user_id=chain.user_id,
            principal_id=chain.user_id,
            hardware_requirements=HardwareRequirements(
                cpu=chain.cpu_units, gpu=chain.gpu_units
            ),
            chain_id=chain.chain_id,
            chain_round=round_index,
        ),
        requires_gpu={graph_name: kind is OpKind.GPU},
    )


def _graph_for_kind(name: str, kind: OpKind) -> RuntimeGraph:
    """Build a single-op runtime graph whose op matches ``kind``.

    The backend/task_type mirror the real classification rules
    (``_runtime_op_requires_gpu`` and the cost model's ``_classify``) so the
    scheduler's GPU peek and the cost model see the intended op kind.
    """
    if kind is OpKind.GPU:
        backend, task_type = "vllm", "inference"
    elif kind is OpKind.DB:
        backend, task_type = "data_retrieval", "data_retrieval"
    else:
        backend, task_type = "http", "http"
    node_id = f"{name}_node"
    op = RuntimeOp(
        node_id=node_id,
        task_type=task_type,
        backend=backend,
        model="dummy-model",
        data_spec={},
        model_spec={},
        inference_spec={},
    )
    return RuntimeGraph(
        nodes={node_id: op},
        node_order=[node_id],
        output_node_map={node_id: "output"},
    )
