"""Base classes for workflow job management."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

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
    dispatch_token: str | None = None
    miss_count: int = 0


@dataclass(slots=True)
class BatchSelection:
    workflows: list[WorkflowItem]
    runtime_graphs: dict[str, RuntimeGraph]
    data_profile_graphs: dict[str, RuntimeGraph]
    config: LumilakeRequestConfig
    clustering_seconds: float = 0.0


@dataclass(slots=True)
class BatchReservation:
    """Two-phase batch selection handle.

    ``reserve_batch`` returns a reservation whose ``selection`` is the batch
    that *would* be consumed by a subsequent ``commit_reservation``. The
    queue is not mutated until commit; callers that fail to acquire
    workers can call ``abort_reservation`` to release the items back. This
    keeps "selected batches are never dropped" intact when scheduling
    decisions need to inspect batch contents before reserving workers.

    ``_payload`` is opaque per-implementation state (e.g. the set of
    workflow ids to remove on commit). External callers must not touch it.
    """

    selection: BatchSelection
    _payload: Any = field(default=None, repr=False)


@dataclass(slots=True)
class Job:
    request_id: str
    runtime_graphs: dict[str, RuntimeGraph]
    data_profile_graphs: dict[str, RuntimeGraph]
    dsl_graphs: dict[str, CompiledGraph]
    workflow_slices: dict[str, WorkflowSliceMeta]
    config: LumilakeRequestConfig
    dispatch_token: str | None = None


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
    async def reserve_batch(self, batch_size: int) -> BatchReservation | None:
        """Compute (without consuming) the next batch the manager would select.

        Returns ``None`` if no batch is available. The reservation must be
        finalized with ``commit_reservation`` (queue is updated) or
        released with ``abort_reservation`` (queue is unchanged).
        Overlapping reservations are not supported.
        """

    @abstractmethod
    async def commit_reservation(self, reservation: BatchReservation) -> None:
        """Apply the queue mutations associated with ``reservation``."""

    @abstractmethod
    async def abort_reservation(self, reservation: BatchReservation) -> None:
        """Discard the reservation; the queue is left exactly as it was."""

    async def select_batch(self, batch_size: int) -> BatchSelection | None:
        """Reserve + commit in one call. Convenience for callers that don't
        need to inspect batch contents before consuming."""
        reservation = await self.reserve_batch(batch_size)
        if reservation is None:
            return None
        await self.commit_reservation(reservation)
        return reservation.selection

    @abstractmethod
    def finalize_workflows(self, workflow_ids: Iterable[str]) -> None:
        """Drop metadata for completed workflows."""

    @abstractmethod
    def get_workflow(self, workflow_id: str) -> WorkflowItem:
        """Lookup workflow metadata by ID."""
