"""Topological-sort optimizer.

Assigns every GPU-backend node to the first GPU worker and every CPU node to the
first CPU worker, in topological order. A node of a capability-gated task type
goes to the first worker of its class that advertises it and carries its GPU
request: a CPU worker, or a GPU worker for a python step that requests GPUs. It
uses no cost model.
"""

from collections.abc import Mapping, Sequence

from lumilake_hook import WorkerProfile
from lumilake_server.data_profile_models import DataProfileResultRow
from lumilake_server.runtime.optimizer.base import BaseOptimizer, Schedule
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.worker_capability import (
    CAPABILITY_GATED_TASK_TYPES,
    profile_fits_gpu_request,
)

_GPU_BACKENDS = {"vllm", "transformers", "diffusers", "omni"}


class TopologicalSortOptimizer(BaseOptimizer):
    def generate_schedule(
        self,
        graph: RuntimeGraph,
        worker_names: list[str],
        worker_profiles: Mapping[str, WorkerProfile],
        data_profile_results: (
            Mapping[str, Sequence[DataProfileResultRow]] | None
        ) = None,
    ) -> Schedule:
        gpu_workers = [
            w
            for w in worker_names
            if w in worker_profiles and worker_profiles[w].has_gpu
        ]
        cpu_workers = [
            w
            for w in worker_names
            if w not in worker_profiles or not worker_profiles[w].has_gpu
        ]
        if not gpu_workers:
            gpu_workers = worker_names
        if not cpu_workers:
            cpu_workers = worker_names
        assignment: dict[str, list[str]] = {w: [] for w in worker_names}
        for node_id in graph.topological_order():
            op = graph.nodes[node_id]
            if op.task_type in CAPABILITY_GATED_TASK_TYPES:
                gpu_request = op.data_spec.gpu_request
                candidates = [
                    w
                    for w in worker_names
                    if w in worker_profiles
                    and worker_profiles[w].has_gpu == (gpu_request is not None)
                    and op.task_type in worker_profiles[w].supported_task_types
                    and profile_fits_gpu_request(worker_profiles[w], gpu_request)
                ]
                if not candidates:
                    kind = "GPU" if gpu_request is not None else "CPU"
                    if gpu_request is not None:
                        detail = gpu_request.model_dump(exclude_none=True)
                        raise ValueError(
                            f"No {kind} worker carries the GPUs node '{node_id}'"
                            f" requests (gpu={detail})."
                        )
                    raise ValueError(
                        f"Node '{node_id}' has task type {op.task_type!r}, but none of"
                        f" the selected {kind} workers in {worker_names} advertises it."
                    )
                assignment[candidates[0]].append(node_id)
            elif op.backend in _GPU_BACKENDS:
                assignment[gpu_workers[0]].append(node_id)
            else:
                assignment[cpu_workers[0]].append(node_id)
        return Schedule(worker_assignment=assignment)
