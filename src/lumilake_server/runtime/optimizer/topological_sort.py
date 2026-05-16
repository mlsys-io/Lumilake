"""Topological-sort baseline optimizer.

Assigns every GPU-backend node to the first GPU worker and every CPU
node to the first CPU worker, in topological order. Serves as the
naive baseline against which cost-aware optimizers (HALO, HALO+Helium)
are measured.
"""

from typing import Any

from lumilake_server.runtime.optimizer.base import BaseOptimizer, Schedule
from lumilake_server.runtime.runtime_graph import RuntimeGraph

_GPU_BACKENDS = {"vllm", "transformers", "diffusers", "omni"}


class TopologicalSortOptimizer(BaseOptimizer):
    def generate_schedule(
        self,
        graph: RuntimeGraph,
        worker_names: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        data_profile_results: dict[str, list[dict[str, Any]]] | None = None,
    ) -> Schedule:
        gpu_workers = [
            w
            for w in worker_names
            if w in worker_profiles and worker_profiles[w]["has_gpu"]
        ]
        cpu_workers = [
            w
            for w in worker_names
            if w not in worker_profiles or not worker_profiles[w]["has_gpu"]
        ]
        if not gpu_workers:
            gpu_workers = worker_names
        if not cpu_workers:
            cpu_workers = worker_names
        assignment: dict[str, list[str]] = {w: [] for w in worker_names}
        for node_id in graph.topological_order():
            op = graph.nodes[node_id]
            if op.backend in _GPU_BACKENDS:
                assignment[gpu_workers[0]].append(node_id)
            else:
                assignment[cpu_workers[0]].append(node_id)
        return Schedule(worker_assignment=assignment)
