"""RemoteOptimizer: delegates schedule generation to an external service.

Not registered in ``OPTIMIZER_TYPES``; contributed via an ``OptimizerProvider`` plugin.
Misconfiguration (missing URL or optimizer_type) raises at construction so it surfaces
at plugin-load time. URL must use ``https://``; ``http://`` only for loopback
(``localhost``, ``127.0.0.1``, ``::1``).
"""

from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from lumilake import envs
from lumilake_hook import DataProfileResultRow, WorkerProfile, validate_remote_url
from lumilake_server.hooks.security import runtime_token_var
from lumilake_server.runtime.optimizer.base import BaseOptimizer, Schedule
from lumilake_server.runtime.optimizer.schemas import ScheduleRequest, ScheduleResponse
from lumilake_server.runtime.runtime_graph import RuntimeGraph, RuntimeGraphSchema


class RemoteOptimizer(BaseOptimizer):
    """Optimizer that POSTs to a remote schedule-protocol endpoint."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        optimizer_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        url = (base_url or envs.LUMILAKE_REMOTE_OPTIMIZER_URL).strip()
        if not url:
            raise ValueError(
                "RemoteOptimizer requires LUMILAKE_REMOTE_OPTIMIZER_URL to be set."
            )
        self._base_url = validate_remote_url(url)

        if optimizer_type is None:
            raise ValueError(
                "RemoteOptimizer requires optimizer_type "
                "(e.g. RemoteOptimizer(optimizer_type='halo-greedy'))."
            )
        self._optimizer_type = optimizer_type

    def generate_schedule(
        self,
        graph: RuntimeGraph,
        worker_names: list[str],
        worker_profiles: Mapping[str, WorkerProfile],
        data_profile_results: (
            Mapping[str, Sequence[DataProfileResultRow]] | None
        ) = None,
    ) -> Schedule:
        gpu_python_nodes = sorted(
            node_id
            for node_id, op in graph.nodes.items()
            if op.data_spec.gpu_request is not None
        )
        if gpu_python_nodes:
            raise ValueError(
                f"Graph has GPU python step node(s) {gpu_python_nodes}, but a remote"
                " optimizer cannot place them; run the job under HALO or"
                " topological-sort."
            )
        graph_schema = RuntimeGraphSchema.model_validate(graph.serialize())
        request = ScheduleRequest(
            graph=graph_schema,
            worker_names=worker_names,
            worker_profiles=worker_profiles,
            data_profile_results=(
                None
                if data_profile_results is None
                else {key: list(rows) for key, rows in data_profile_results.items()}
            ),
            optimizer_type=self._optimizer_type,
        )

        headers: dict[str, str] = {"Content-Type": "application/json"}
        bearer = runtime_token_var.get()
        if bearer is not None:
            headers["Authorization"] = f"Bearer {bearer}"

        timeout = envs.LUMILAKE_HTTP_TIMEOUT_SECONDS
        url = f"{self._base_url}/api/v1/optimizer/schedule"

        response = httpx.post(
            url,
            content=request.model_dump_json(),
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        resp = ScheduleResponse.model_validate(response.json())
        return Schedule(worker_assignment=resp.worker_assignment)
