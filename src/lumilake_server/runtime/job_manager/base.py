"""Base classes for workflow job management."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

from lumilake_server.graphs import CompiledGraph
from lumilake_server.runtime.protocol import LumilakeRequestConfig
from lumilake_server.runtime.request import WorkflowSliceMeta
from lumilake_server.runtime.runtime_graph import RuntimeGraph


@dataclass(slots=True)
class WorkflowItem:
    workflow_id: str
    request_id: str
    graph_name: str
    public_graph_name: str
    slice_index: int
    slice_start: int
    slice_length: int
    total_length: int
    template_hash: str
    varying_input_keys: tuple[str, ...]
    runtime_graph: RuntimeGraph
    data_profile_graph: RuntimeGraph
    dsl_graph: CompiledGraph
    config: LumilakeRequestConfig
    enqueued_at: float
    miss_count: int = 0


@dataclass(slots=True)
class BatchSelection:
    workflows: list[WorkflowItem]
    runtime_graphs: dict[str, RuntimeGraph]
    data_profile_graphs: dict[str, RuntimeGraph]
    config: LumilakeRequestConfig
    clustering_seconds: float = 0.0


@dataclass(slots=True)
class Job:
    request_id: str
    runtime_graphs: dict[str, RuntimeGraph]
    data_profile_graphs: dict[str, RuntimeGraph]
    dsl_graphs: dict[str, CompiledGraph]
    workflow_slices: dict[str, WorkflowSliceMeta]
    config: LumilakeRequestConfig


class BaseJobManager(ABC):
    """Interface for workflow queueing and batch selection."""

    @abstractmethod
    async def enqueue(
        self,
        job: Job,
    ) -> list[WorkflowItem]:
        """Enqueue multiple workflows and return their metadata."""

    @abstractmethod
    async def has_work(self) -> bool:
        """Return True if there are queued workflows."""

    @abstractmethod
    async def wait_for_work(self) -> None:
        """Block until there is queued work."""

    @abstractmethod
    async def get_pending_stats(self) -> tuple[int, float | None]:
        """Return (pending workflow count, oldest enqueue timestamp)."""

    @abstractmethod
    async def select_batch(self, batch_size: int) -> BatchSelection | None:
        """Select and dequeue a batch of workflows."""

    @abstractmethod
    def finalize_workflows(self, workflow_ids: Iterable[str]) -> None:
        """Drop metadata for completed workflows."""

    @abstractmethod
    def get_workflow(self, workflow_id: str) -> WorkflowItem:
        """Lookup workflow metadata by ID."""
