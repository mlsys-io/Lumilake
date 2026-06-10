"""Optimizer plugin contract: OptimizerProvider, OptimizerHandle, Schedule,
and a generic RemoteOptimizer for HTTP-routed providers."""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

import httpx

_LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})

# Set by the server's auth middleware; read by RemoteOptimizer to
# forward the caller's bearer to the upstream optimizer service.
runtime_token_var: ContextVar[str | None] = ContextVar(
    "lumilake_runtime_token", default=None
)


@dataclass(slots=True)
class Schedule:
    worker_assignment: dict[str, list[str]]


@runtime_checkable
class OptimizerHandle(Protocol):
    def generate_schedule(
        self,
        graph: Any,
        worker_names: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        data_profile_results: dict[str, list[dict[str, Any]]] | None = None,
    ) -> Schedule: ...


@runtime_checkable
class OptimizerProvider(Protocol):
    """Plugin hook contributing optimizer types beyond the built-in registry.

    Names from ``list_optimizers()`` are compared case-insensitively. See
    ``docs/PLUGINS.md`` for an example."""

    def list_optimizers(self) -> list[str]: ...

    def create_optimizer(
        self, optimizer_type: str, **kwargs: Any
    ) -> OptimizerHandle: ...


class RemoteOptimizer:
    """OptimizerHandle that POSTs to a remote schedule-protocol endpoint.

    URL must be ``https://`` (or ``http://`` for loopback)."""

    def __init__(
        self,
        *,
        base_url: str,
        optimizer_type: str,
        http_timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        url = (base_url or "").strip()
        if not url:
            raise ValueError("RemoteOptimizer requires a non-empty base_url.")
        parsed = urlparse(url)
        if parsed.scheme == "https":
            pass
        elif parsed.scheme == "http" and parsed.hostname in _LOOPBACK_HOSTS:
            pass
        else:
            raise ValueError(
                "RemoteOptimizer base_url must use https:// "
                "(or http:// for loopback only). "
                f"Got: {parsed.scheme}://{parsed.hostname}"
            )
        if not optimizer_type:
            raise ValueError("RemoteOptimizer requires optimizer_type")
        self._base_url = url.rstrip("/")
        self._optimizer_type = optimizer_type
        self._http_timeout_seconds = http_timeout_seconds

    def generate_schedule(
        self,
        graph: Any,
        worker_names: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        data_profile_results: dict[str, list[dict[str, Any]]] | None = None,
    ) -> Schedule:
        serialize = getattr(graph, "serialize", None)
        if not callable(serialize):
            raise TypeError(
                "RemoteOptimizer.generate_schedule expects graph.serialize()"
            )
        request_body = {
            "graph": serialize(),
            "worker_names": worker_names,
            "worker_profiles": worker_profiles,
            "data_profile_results": data_profile_results,
            "optimizer_type": self._optimizer_type,
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        bearer = runtime_token_var.get()
        if bearer is not None:
            headers["Authorization"] = f"Bearer {bearer}"
        response = httpx.post(
            f"{self._base_url}/api/v1/optimizer/schedule",
            json=request_body,
            headers=headers,
            timeout=self._http_timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        worker_assignment = body.get("worker_assignment")
        if not isinstance(worker_assignment, dict):
            raise RuntimeError(
                "Remote optimizer response missing worker_assignment dict"
            )
        return Schedule(worker_assignment=worker_assignment)
