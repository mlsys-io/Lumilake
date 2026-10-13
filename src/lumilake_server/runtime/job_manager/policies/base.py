"""Base class for Lumilake scheduling policies."""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import BaseModel

from lumilake_server.runtime.job_manager.base import WorkflowItem

__all__ = ["BaseSchedulingPolicy"]


class BaseSchedulingPolicy(ABC):
    """Interface for the ordering/selection decisions a job manager makes.

    A policy decides which queued items form the next batch. The core method
    is :meth:`select_batch`, which receives the candidate *set* for the anchor
    partition and returns the ranked subset that forms the batch.

    Batch cost is not separable over its members (a model swap is charged once
    per distinct model in the batch, not per item), so the interface takes the
    candidate set and returns a subset rather than reducing to a per-item key
    — a future policy can score sets, not just items.
    """

    args_model: ClassVar[type[BaseModel] | None] = None
    """Pydantic model of the policy's custom args; ``None`` means it takes none."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        **kwargs: Any,
    ) -> None:
        self._clock = clock

    @abstractmethod
    def select_batch(
        self,
        candidates: list[WorkflowItem],
        affinity_rank: dict[str, int],
        batch_size: int,
    ) -> list[str]:
        """Return the ranked subset of ``candidates`` that forms the batch.

        ``candidates`` is the candidate set for the anchor partition (already
        narrowed by starvation pinning). ``affinity_rank`` maps workflow id to
        the affinity-selection order, so a policy can break ties the way
        clustering shaped the batch. The returned ids must be a subset of the
        candidate ids, ordered best-first, and at most ``batch_size`` long.

        Policies get read-only access to the items and the manager's state
        they were constructed with; they must not mutate queue state.
        """
        raise NotImplementedError

    def on_commit(self, workflows: list[WorkflowItem]) -> None:
        """Hook invoked after a batch is committed.

        The default is a no-op. A policy that tracks attained service (e.g.
        plas) charges it here, on commit rather than on selection so an
        aborted reservation does not corrupt the accounting.
        """

    def observe_free_lanes(self, free_lanes: int | None) -> None:
        """Hook invoked by ``reserve_batch`` before each ``select_batch``.

        ``free_lanes`` is the number of whole dispatches of the anchor
        partition the idle workers can serve at once; this batch's own lane is
        included. Per worker class the dispatch claims, it is the idle workers
        (those meeting the partition's hardware, when the capacity snapshot
        carries worker profiles) divided by the class's worker group size, and
        the smallest class count is taken. A CPU-only partition claims a CPU
        group, a GPU partition a GPU and a CPU group, and a configured CPU group
        of zero still claims one CPU worker when a candidate has a CPU-only
        node. When the candidates need a gated task type, the CPU side is also
        bounded by the eligible CPU workers advertising every such type (a
        conservative bound: the union over all candidates), and the count is
        ``0`` without worker profiles. It is ``0`` when no class is claimed.

        ``None`` means the count is unknown: ``reserve_batch`` was called
        without free capacity (the server passes it only while capacity-aware
        selection is on). The default is a no-op; a policy that spreads work
        over idle lanes stores the value here.
        """
