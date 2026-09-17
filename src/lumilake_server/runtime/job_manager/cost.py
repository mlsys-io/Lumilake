"""Analytic resource-area cost model for fair-weighted scheduling.

Every estimate is a function of graph shape and declared demand, parameterised
by named hyperparameters with reasoned defaults. There is no trace corpus and
nothing here fits, learns, or requires historical data.
"""

from dataclasses import dataclass, field

from lumilake import envs

from lumilake_server.runtime.job_manager.base import WorkflowItem
from lumilake_server.runtime.optimizer.multimodal_cost import (
    MultimodalCostCoefficients,
    compute_gpu_exec_cost,
)
from lumilake_server.runtime.optimizer.schedule.models import Node
from lumilake_server.runtime.runtime_ops import RuntimeOp
from lumilake_server.utils.utils import parse_memory_to_bytes

# GPU backends, matching HALO's ``_is_gpu_backend``.
_GPU_BACKENDS = frozenset({"vllm", "transformers", "diffusers", "omni"})
# DB backend, matching HALO's ``_map_engine``.
_DB_BACKEND = "data_retrieval"


@dataclass(frozen=True)
class CostParams:
    """Hyperparameters for the analytic cost model.

    Defaults mirror HALO's hand-tuned constants where the same quantity exists
    (GPU coefficients, DB cost), and add a CPU coefficient and a model-size
    fallback for graphs whose model name carries no inferable size.
    """

    gpu_coeffs: MultimodalCostCoefficients = field(
        default_factory=MultimodalCostCoefficients
    )
    db_sec_per_query: float = envs.LUMILAKE_COST_DB_SEC_PER_QUERY
    cpu_sec_per_node: float = envs.LUMILAKE_COST_CPU_SEC_PER_NODE
    default_model_size_b: float = envs.LUMILAKE_COST_DEFAULT_MODEL_SIZE_B
    input_query_count: int = envs.LUMILAKE_COST_INPUT_QUERY_COUNT


def _model_size_b(model: str | None, default: float) -> float:
    """Resolve a model name to a size in billions of parameters.

    Mirrors HALO's suffix inference (``7B``, ``1.5B``, ``560M``); falls back to
    the configured default when the name carries no inferable size.
    """
    if not model:
        return default
    normalized = str(model).strip().lower()
    if not normalized:
        return default
    # Trailing size suffix, e.g. "llama-7b", "qwen-1.5b", "bge-560m".
    import re

    match = re.search(r"(\d+(?:\.\d+)?)\s*(billion|bn|b|m)\b", normalized)
    if not match:
        return default
    try:
        raw = float(match.group(1))
    except ValueError:
        return default
    unit = match.group(2)
    if unit == "m":
        return raw / 1000.0
    return raw


def _classify(op: RuntimeOp) -> str:
    """Classify a runtime op as ``gpu``, ``db``, or ``cpu``.

    Mirrors HALO's engine mapping: GPU backends map to vLLM, the
    ``data_retrieval`` backend maps to DB, everything else is CPU.
    """
    backend = str(op.backend).strip().lower()
    if backend in _GPU_BACKENDS:
        return "gpu"
    if backend == _DB_BACKEND:
        return "db"
    return "cpu"


def _to_halo_node(op: RuntimeOp) -> Node:
    """Build the minimal ``Node`` view ``compute_gpu_exec_cost`` needs."""
    raw = dict(op.data_spec) if isinstance(op.data_spec, dict) else {}
    if isinstance(op.inference_spec, dict) and op.inference_spec:
        raw["_inference_spec"] = dict(op.inference_spec)
    return Node(
        id=op.node_id,
        type=op.task_type,
        engine="vllm",
        model=op.model,
        raw=raw,
    )


def _node_duration(op: RuntimeOp, params: CostParams) -> float:
    """Estimated wall-clock seconds for a single op."""
    kind = _classify(op)
    if kind == "gpu":
        return compute_gpu_exec_cost(
            node=_to_halo_node(op),
            model_size_b=_model_size_b(op.model, params.default_model_size_b),
            input_query_count=params.input_query_count,
            coeffs=params.gpu_coeffs,
        )
    if kind == "db":
        return params.db_sec_per_query * params.input_query_count
    return params.cpu_sec_per_node


def estimate_area(item: WorkflowItem, params: CostParams | None = None) -> float | None:
    """Estimated resource-area (dominant-resource share x seconds).

    Returns ``None`` when the graph cannot be estimated (e.g. agent-mode
    retrieval, whose duration is not predictable from graph shape) — callers
    fall back to least-attained-service rather than guessing a number.
    """
    params = params or CostParams()
    graph = item.runtime_graph
    if not graph.nodes:
        return None

    durations: dict[str, float] = {}
    for node_id, op in graph.nodes.items():
        if _classify(op) == "db":
            data_spec = op.data_spec if isinstance(op.data_spec, dict) else {}
            if data_spec.get("mode") == "agent":
                return None
        durations[node_id] = _node_duration(op, params)

    # Critical path through the dependency graph (the graph executes with
    # parallelism), not the sum of node durations.
    longest: dict[str, float] = {}
    for node_id in graph.topological_order():
        best = durations[node_id]
        for dep in graph.nodes[node_id].dependencies:
            if dep in longest:
                best = max(best, longest[dep] + durations[node_id])
        longest[node_id] = best
    critical_path = max(longest.values()) if longest else 0.0

    share = _dominant_share(item)
    area = critical_path * share
    if not area > 0:
        return None
    return area


def _dominant_share(item: WorkflowItem) -> float:
    """Dominant-resource share of the request against per-worker defaults.

    A hyperparameter-driven approximation of cluster share, not a measured one:
    the denominator is the per-worker default hardware, so a request that asks
    for the default worker claims share 1.0. The share is the largest of the
    request's per-resource fractions across all four dimensions (cpu, memory,
    gpu, gpu_memory), keeping the accounting coherent with DRF.
    """
    hw = item.config.hardware_requirements
    cpu = (
        hw.cpu
        if hw is not None and hw.cpu is not None
        else envs.HARDWARE_CPU_REQUIREMENT
    )
    gpu = (
        hw.gpu
        if hw is not None and hw.gpu is not None
        else envs.HARDWARE_GPU_REQUIREMENT
    )
    memory = (
        hw.memory
        if hw is not None and hw.memory is not None
        else envs.HARDWARE_MEMORY_REQUIREMENT
    )
    gpu_memory = (
        hw.gpu_memory
        if hw is not None and hw.gpu_memory is not None
        else envs.HARDWARE_GPU_MEMORY_REQUIREMENT
    )
    cpu_share = (
        cpu / envs.HARDWARE_CPU_REQUIREMENT if envs.HARDWARE_CPU_REQUIREMENT else 0.0
    )
    gpu_share = (
        gpu / envs.HARDWARE_GPU_REQUIREMENT if envs.HARDWARE_GPU_REQUIREMENT else 0.0
    )
    memory_share = _memory_share(memory, envs.HARDWARE_MEMORY_REQUIREMENT)
    gpu_memory_share = _memory_share(gpu_memory, envs.HARDWARE_GPU_MEMORY_REQUIREMENT)
    return max(cpu_share, memory_share, gpu_share, gpu_memory_share)


def _memory_share(requested: str, default: str) -> float:
    """Fraction of the per-worker default memory that ``requested`` claims."""
    requested_bytes = parse_memory_to_bytes(requested)
    default_bytes = parse_memory_to_bytes(default)
    if requested_bytes is None or not default_bytes:
        return 0.0
    return requested_bytes / default_bytes


__all__ = ["CostParams", "estimate_area"]
