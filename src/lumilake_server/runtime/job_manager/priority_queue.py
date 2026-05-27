"""Priority queue-based job manager with starvation avoidance."""

import asyncio
import logging
import time
from collections import deque
from collections.abc import Iterable

from lumilake import envs
from lumilake.log import Logger, LogLevel, init_child_logger

from lumilake_server.runtime.optimizer.base import BaseOptimizer
from lumilake_server.runtime.protocol import Priority
from lumilake_server.utils.utils import unique_id

from .base import BaseJobManager, BatchSelection, Job, WorkflowItem
from .cluster_algo.clustering import select_affinity_batch_ids

DEFAULT_QUANTUMS: dict[Priority, int] = {
    Priority.HIGH: envs.LUMILAKE_QUEUE_QUANTUM_HIGH,
    Priority.MEDIUM: envs.LUMILAKE_QUEUE_QUANTUM_MEDIUM,
    Priority.LOW: envs.LUMILAKE_QUEUE_QUANTUM_LOW,
}


class PriorityJobManager(BaseJobManager):
    """Job manager with per-priority queues and starvation avoidance."""

    def __init__(
        self,
        optimizer: BaseOptimizer,
        quantums: dict[Priority, int] | None = None,
        starvation_limit: int = envs.LUMILAKE_STARVATION_LIMIT,
        logger: Logger | None = None,
        log_level: LogLevel | None = None,
    ) -> None:
        self._optimizer = optimizer
        self._quantums = DEFAULT_QUANTUMS.copy()
        if quantums is not None:
            self._quantums.update(quantums)
        for priority in Priority:
            self._quantums.setdefault(priority, 1)
        self._starvation_limit = starvation_limit
        # Per-priority, per-user queues.
        self._queues: dict[Priority, dict[str, deque[WorkflowItem]]] = {
            priority: {} for priority in Priority
        }
        # Round-robin order of user IDs per priority.
        self._rr_user_order: dict[Priority, deque[str]] = {
            priority: deque() for priority in Priority
        }
        # Round-robin order of principal IDs across all priorities. select_batch
        # picks one principal per round so a single FlowMesh dispatch never
        # spans principals.
        self._rr_principal_order: deque[str] = deque()
        self._items: dict[str, WorkflowItem] = {}
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Event()
        self.logger = init_child_logger("JobManager", logger, log_level)

    @staticmethod
    def _format_item(item: WorkflowItem) -> str:
        return (
            f"{item.workflow_id}"
            f"(req={item.request_id},user={item.config.user_id},pri={item.config.priority.value},"
            f"graph={item.graph_name},public={item.public_graph_name},"
            f"slice={item.slice_index},miss={item.miss_count})"
        )

    @staticmethod
    def _queue_owner_id(item: WorkflowItem) -> str:
        return item.config.user_id

    def _queue_sizes_by_priority(self) -> dict[str, int]:
        return {
            priority.value: self._priority_queue_size(priority) for priority in Priority
        }

    def _quantums_by_priority(self) -> dict[str, int]:
        return {priority.value: self._quantums[priority] for priority in Priority}

    async def enqueue(
        self,
        job: Job,
    ) -> list[WorkflowItem]:
        now = time.time()
        workflow_items: list[WorkflowItem] = []
        for graph_name, runtime_graph in job.runtime_graphs.items():
            data_profile_graph = job.data_profile_graphs[graph_name]
            dsl_graph = job.dsl_graphs[graph_name]
            slice_meta = job.workflow_slices[graph_name]
            workflow_id = unique_id()
            workflow_items.append(
                WorkflowItem(
                    workflow_id=workflow_id,
                    request_id=job.request_id,
                    graph_name=graph_name,
                    public_graph_name=slice_meta.public_graph_name,
                    slice_index=slice_meta.slice_index,
                    slice_start=slice_meta.slice_start,
                    slice_length=slice_meta.slice_length,
                    total_length=slice_meta.total_length,
                    template_hash=slice_meta.template_hash,
                    varying_input_keys=slice_meta.varying_input_keys,
                    runtime_graph=runtime_graph,
                    data_profile_graph=data_profile_graph,
                    dsl_graph=dsl_graph,
                    config=job.config,
                    enqueued_at=now,
                )
            )
        async with self._lock:
            for item in workflow_items:
                owner_id = self._queue_owner_id(item)
                user_queues = self._queues[job.config.priority]
                if owner_id not in user_queues:
                    user_queues[owner_id] = deque()
                    self._rr_user_order[job.config.priority].append(owner_id)
                user_queues[owner_id].append(item)
                self._items[item.workflow_id] = item
                principal_id = item.config.principal_id
                if principal_id not in self._rr_principal_order:
                    self._rr_principal_order.append(principal_id)
            self._not_empty.set()
            self.logger.debug(
                "Enqueued request=%s priority=%s workflows=%d queue_sizes=%s",
                job.request_id,
                job.config.priority.value,
                len(workflow_items),
                self._queue_sizes_by_priority(),
            )
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Enqueued items=%s",
                    [self._format_item(item) for item in workflow_items],
                )
        return workflow_items

    async def has_work(self) -> bool:
        async with self._lock:
            return any(
                self._priority_queue_size(priority) > 0 for priority in self._queues
            )

    async def wait_for_work(self) -> None:
        if await self.has_work():
            return
        await self._not_empty.wait()

    async def get_pending_stats(self) -> tuple[int, float | None]:
        async with self._lock:
            pending = 0
            oldest_enqueued_at: float | None = None
            for priority in Priority:
                for queue in self._queues[priority].values():
                    pending += len(queue)
                    for item in queue:
                        if (
                            oldest_enqueued_at is None
                            or item.enqueued_at < oldest_enqueued_at
                        ):
                            oldest_enqueued_at = item.enqueued_at
            return pending, oldest_enqueued_at

    def get_workflow(self, workflow_id: str) -> WorkflowItem:
        return self._items[workflow_id]

    def finalize_workflows(self, workflow_ids: Iterable[str]) -> None:
        for workflow_id in workflow_ids:
            self._items.pop(workflow_id, None)

    async def select_batch(self, batch_size: int) -> BatchSelection | None:
        if batch_size <= 0:
            batch_size = 1

        async with self._lock:
            present_principals: set[str] = set()
            items_by_principal: dict[str, int] = {}
            starved_global: list[WorkflowItem] = []
            for priority in Priority:
                for queue in self._queues[priority].values():
                    for item in queue:
                        pid = item.config.principal_id
                        present_principals.add(pid)
                        items_by_principal[pid] = items_by_principal.get(pid, 0) + 1
                        if item.miss_count >= self._starvation_limit:
                            starved_global.append(item)

            if not present_principals:
                self._not_empty.clear()
                return None

            anchor_principal: str
            if starved_global:
                anchor_principal = starved_global[0].config.principal_id
            else:
                picked = self._pick_principal_round_robin_locked(present_principals)
                if picked is None:
                    picked = next(iter(present_principals))
                    self._rr_principal_order.append(picked)
                anchor_principal = picked

            candidates = self._build_candidate_pool_for_principal_locked(
                anchor_principal
            )
            if not candidates:
                self._not_empty.clear()
                return None

            queue_sizes = self._queue_sizes_by_priority()
            candidate_by_priority = {priority.value: 0 for priority in Priority}
            candidate_by_user: dict[str, int] = {}
            for item in candidates:
                candidate_by_priority[item.config.priority.value] += 1
                owner_id = self._queue_owner_id(item)
                candidate_by_user[owner_id] = candidate_by_user.get(owner_id, 0) + 1
            self.logger.debug(
                "Candidate pool anchor=%s size=%d batch_size=%d quantums=%s "
                "queue_sizes=%s items_by_principal=%s "
                "candidate_by_priority=%s candidate_by_user=%s",
                anchor_principal,
                len(candidates),
                batch_size,
                self._quantums_by_priority(),
                queue_sizes,
                items_by_principal,
                candidate_by_priority,
                candidate_by_user,
            )
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Candidate pool items=%s",
                    [self._format_item(item) for item in candidates],
                )

            starved = [
                item for item in candidates if item.miss_count >= self._starvation_limit
            ]
            item_map = {item.workflow_id: item for item in candidates}
            clustering_start = time.perf_counter()
            base_batch_ids = select_affinity_batch_ids(
                {item.workflow_id: item.runtime_graph for item in candidates},
                {item.workflow_id: item.enqueued_at for item in candidates},
                batch_size,
                pinned_ids=[
                    item.workflow_id
                    for item in sorted(starved, key=lambda node: node.enqueued_at)
                ],
            )
            clustering_seconds = time.perf_counter() - clustering_start
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Affinity selected ids=%s",
                    [
                        self._format_item(item_map[workflow_id])
                        for workflow_id in base_batch_ids
                        if workflow_id in item_map
                    ],
                )
            selected_ids = self._apply_starvation_policy(
                candidates, base_batch_ids, batch_size
            )
            if not starved:
                selected_ids = self._apply_user_fairness(
                    candidates,
                    selected_ids,
                    batch_size,
                )
            if starved:
                self.logger.info(
                    "Starvation override: forced=%d selected=%d forced_items=%s",
                    len(starved),
                    len(selected_ids),
                    [self._format_item(item) for item in starved],
                )
            else:
                self.logger.debug("Selected batch size=%d", len(selected_ids))

            if self.logger.isEnabledFor(logging.DEBUG):
                selected_names = [
                    self._format_item(item_map[workflow_id])
                    for workflow_id in selected_ids
                    if workflow_id in item_map
                ]
                self.logger.debug("Selected batch items=%s", selected_names)

            selected_items = [item_map[workflow_id] for workflow_id in selected_ids]
            selected_set = set(selected_ids)

            # Update miss counters for candidates not selected.
            for item in candidates:
                if item.workflow_id not in selected_set:
                    item.miss_count += 1

            # Remove selected items from queues.
            for priority, user_queues in self._queues.items():
                if not user_queues:
                    continue
                for user_id in list(user_queues.keys()):
                    queue = user_queues[user_id]
                    filtered = deque(
                        item for item in queue if item.workflow_id not in selected_set
                    )
                    if filtered:
                        user_queues[user_id] = filtered
                    else:
                        del user_queues[user_id]
                self._prune_empty_user_queues_locked(priority)
            self.logger.debug(
                "Post-select queue_sizes=%s",
                self._queue_sizes_by_priority(),
            )

            if not any(
                self._priority_queue_size(priority) > 0 for priority in self._queues
            ):
                self._not_empty.clear()

        # All items in a single user queue share priority + config, so the
        # first item's config is representative for the whole batch.
        config = selected_items[0].config.model_copy(deep=True)
        runtime_graphs = {
            item.workflow_id: item.runtime_graph for item in selected_items
        }
        data_profile_graphs = {
            item.workflow_id: item.data_profile_graph for item in selected_items
        }
        return BatchSelection(
            workflows=selected_items,
            runtime_graphs=runtime_graphs,
            data_profile_graphs=data_profile_graphs,
            config=config,
            clustering_seconds=clustering_seconds,
        )

    def _build_candidate_pool_for_principal_locked(
        self, principal_id: str
    ) -> list[WorkflowItem]:
        candidates: list[WorkflowItem] = []
        for priority in Priority:
            quantum = max(0, self._quantums.get(priority, 0))
            if quantum == 0:
                continue
            candidates.extend(
                self._peek_round_robin_for_principal_locked(
                    priority, quantum, principal_id
                )
            )
        return candidates

    def _pick_principal_round_robin_locked(
        self, present_principals: set[str]
    ) -> str | None:
        while self._rr_principal_order:
            head = self._rr_principal_order[0]
            if head in present_principals:
                self._rr_principal_order.rotate(-1)
                return head
            self._rr_principal_order.popleft()
        return None

    def _priority_queue_size(self, priority: Priority) -> int:
        return sum(len(queue) for queue in self._queues[priority].values())

    def _prune_empty_user_queues_locked(self, priority: Priority) -> None:
        user_queues = self._queues[priority]
        for user_id in list(user_queues.keys()):
            if user_queues[user_id]:
                continue
            del user_queues[user_id]
        rr_order = self._rr_user_order[priority]
        if not rr_order:
            return
        self._rr_user_order[priority] = deque(
            user_id for user_id in rr_order if user_id in user_queues
        )

    def _peek_round_robin_for_principal_locked(
        self, priority: Priority, limit: int, principal_id: str
    ) -> list[WorkflowItem]:
        if limit <= 0:
            return []
        self._prune_empty_user_queues_locked(priority)
        user_queues = self._queues[priority]
        rr_order = self._rr_user_order[priority]
        if not user_queues or not rr_order:
            return []

        filtered: dict[str, list[WorkflowItem]] = {}
        for user_id in rr_order:
            queue = user_queues.get(user_id)
            if not queue:
                continue
            user_items = [
                item for item in queue if item.config.principal_id == principal_id
            ]
            if user_items:
                filtered[user_id] = user_items

        ordered_users = [user_id for user_id in rr_order if user_id in filtered]
        if not ordered_users:
            return []
        local_offsets = {user_id: 0 for user_id in ordered_users}
        candidates: list[WorkflowItem] = []
        while len(candidates) < limit:
            progressed = False
            for user_id in ordered_users:
                items = filtered[user_id]
                idx = local_offsets[user_id]
                if idx >= len(items):
                    continue
                candidates.append(items[idx])
                local_offsets[user_id] = idx + 1
                progressed = True
                if len(candidates) >= limit:
                    break
            if not progressed:
                break

        if len(rr_order) > 1:
            rr_order.rotate(-1)
        return candidates

    def _apply_starvation_policy(
        self,
        candidates: list[WorkflowItem],
        base_batch_ids: list[str],
        batch_size: int,
    ) -> list[str]:
        starved = [
            item for item in candidates if item.miss_count >= self._starvation_limit
        ]
        if not starved:
            return base_batch_ids[:batch_size]

        # Ensure starved workflows are included first.
        selected_ids: list[str] = [
            item.workflow_id for item in sorted(starved, key=lambda i: i.enqueued_at)
        ]

        if len(selected_ids) >= batch_size:
            return selected_ids[:batch_size]

        # Fill remaining slots using base batch order, then candidate order.
        for workflow_id in base_batch_ids:
            if workflow_id not in selected_ids and len(selected_ids) < batch_size:
                selected_ids.append(workflow_id)
        if len(selected_ids) < batch_size:
            for item in candidates:
                if (
                    item.workflow_id not in selected_ids
                    and len(selected_ids) < batch_size
                ):
                    selected_ids.append(item.workflow_id)

        return selected_ids

    def _apply_user_fairness(
        self,
        candidates: list[WorkflowItem],
        selected_ids: list[str],
        batch_size: int,
    ) -> list[str]:
        if batch_size <= 1:
            return selected_ids[:batch_size]
        candidate_map = {item.workflow_id: item for item in candidates}
        selected_ids = [wid for wid in selected_ids if wid in candidate_map]
        if not selected_ids:
            return []

        user_order: list[str] = []
        user_to_ids: dict[str, list[str]] = {}
        for item in candidates:
            owner_id = self._queue_owner_id(item)
            if owner_id not in user_to_ids:
                user_to_ids[owner_id] = []
                user_order.append(owner_id)
            user_to_ids[owner_id].append(item.workflow_id)
        if len(user_order) <= 1:
            return selected_ids[:batch_size]

        selected_set = set(selected_ids)
        fair_seed: list[str] = []
        for user_id in user_order:
            preferred = next(
                (wid for wid in user_to_ids[user_id] if wid in selected_set),
                None,
            )
            picked = preferred or user_to_ids[user_id][0]
            if picked in fair_seed:
                continue
            fair_seed.append(picked)
            if len(fair_seed) >= batch_size:
                return fair_seed[:batch_size]

        final_ids: list[str] = list(fair_seed)
        for workflow_id in selected_ids:
            if workflow_id in final_ids:
                continue
            final_ids.append(workflow_id)
            if len(final_ids) >= batch_size:
                return final_ids[:batch_size]
        for item in candidates:
            if item.workflow_id in final_ids:
                continue
            final_ids.append(item.workflow_id)
            if len(final_ids) >= batch_size:
                break
        return final_ids[:batch_size]
