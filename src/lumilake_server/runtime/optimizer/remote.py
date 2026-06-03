"""RemoteOptimizer: delegates schedule generation to an external service.

Not registered in ``OPTIMIZER_TYPES``; contributed via an ``OptimizerProvider`` plugin.
Misconfiguration (missing URL or optimizer_type) raises at construction so it surfaces
at plugin-load time. URL must use ``https://``; ``http://`` only for loopback
(``localhost``, ``127.0.0.1``, ``::1``).
"""

from typing import Any
from urllib.parse import urlparse

import httpx
from lumilake import envs

from lumilake_server.hooks.security import runtime_token_var
from lumilake_server.runtime.optimizer.base import BaseOptimizer, Schedule
from lumilake_server.runtime.optimizer.schemas import ScheduleRequest, ScheduleResponse
from lumilake_server.runtime.runtime_graph import RuntimeGraph, RuntimeGraphSchema

_LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})


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
        parsed = urlparse(url)
        if parsed.scheme == "https":
            pass  # always allowed
        elif parsed.scheme == "http" and parsed.hostname in _LOOPBACK_HOSTS:
            pass  # loopback http allowed for local dev
        else:
            raise ValueError(
                "LUMILAKE_REMOTE_OPTIMIZER_URL must use https:// "
                "(or http:// for loopback only). "
                f"Got: {parsed.scheme}://{parsed.hostname}"
            )
        self._base_url = url.rstrip("/")

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
        worker_profiles: dict[str, dict[str, Any]],
        data_profile_results: dict[str, list[dict[str, Any]]] | None = None,
    ) -> Schedule:
        graph_schema = RuntimeGraphSchema.model_validate(graph.serialize())
        request = ScheduleRequest(
            graph=graph_schema,
            worker_names=worker_names,
            worker_profiles=worker_profiles,
            data_profile_results=data_profile_results,
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
