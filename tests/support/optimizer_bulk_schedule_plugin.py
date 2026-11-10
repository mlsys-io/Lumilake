"""Plugin advertising an optimizer whose schedule pickles to a bulk payload.

Spawned optimizer subprocesses import plugins by module name, so this module
lives on disk instead of being injected through ``sys.modules``.
"""

from typing import Any

from lumilake_hook import BaseBindings, OptimizerHandle, Schedule
from lumilake_server.runtime.optimizer.base import BaseOptimizer

OPTIMIZER_TYPE = "bulk-schedule"
PAYLOAD_BYTES = 1_000_000


class _BulkScheduleOptimizer(BaseOptimizer):
    def generate_schedule(
        self,
        graph: Any,
        worker_names: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        data_profile_results: dict[str, list[dict[str, Any]]] | None = None,
    ) -> Schedule:
        worker = worker_names[0] if worker_names else "worker-1"
        return Schedule(worker_assignment={worker: ["n" + "x" * PAYLOAD_BYTES]})


class _BulkProvider:
    def list_optimizers(self) -> list[str]:
        return [OPTIMIZER_TYPE]

    def create_optimizer(self, optimizer_type: str, **kwargs: Any) -> OptimizerHandle:
        return _BulkScheduleOptimizer(**kwargs)


def install() -> BaseBindings:
    return BaseBindings(optimizer_providers=(_BulkProvider(),))
