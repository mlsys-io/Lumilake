from typing import Any

from lumilake.runtime.optimizer import OPTIMIZER_TYPES
from lumilake.runtime.optimizer.base import BaseOptimizer, Schedule
from lumilake.runtime.runtime_graph import RuntimeGraph


class SimpleRoundRobinOptimizer(BaseOptimizer):
    def generate_schedule(
        self,
        graph: RuntimeGraph,
        worker_names: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        data_profile_results: dict[str, list[dict[str, Any]]] | None = None,
    ) -> Schedule:
        if not worker_names:
            raise ValueError("SimpleRoundRobinOptimizer requires at least one worker")
        assignment: dict[str, list[str]] = {worker: [] for worker in worker_names}
        for index, node_id in enumerate(graph.topological_order()):
            worker = worker_names[index % len(worker_names)]
            assignment[worker].append(node_id)
        return Schedule(worker_assignment=assignment)


def install_optimizer() -> None:
    OPTIMIZER_TYPES["simple"] = SimpleRoundRobinOptimizer
