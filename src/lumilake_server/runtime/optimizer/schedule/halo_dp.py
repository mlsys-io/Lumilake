import logging
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache

from .models import GraphSpec, Node, QueryPlanChoice, Worker
from .topo_utils import topological_order

logger = logging.getLogger(__name__)


# Signature of a single DB query, used as the key for cache multiplier lookups.
@dataclass(frozen=True, slots=True)
class QuerySignature:
    node_id: str
    query_name: str
    plan_id: str
    footprints: tuple[tuple[str, int], ...] = tuple()


# Per-worker local state:
#   - GPU worker: tracks last_model / last_node
#   - CPU worker: used purely for state deduplication
@dataclass(frozen=True, slots=True)
class WorkerState:
    worker_idx: int
    last_model_id: int
    last_node_id: int


class DPSolver:
    """DP scheduler: optimize GPU-side batches and worker mappings only;
    CPU nodes are auto-filled based on the GPU plan.

    Per epoch:
      1. GPU side: pick a batch (a feasible subgraph) from the ready LLM
         nodes, enumerate GPU worker mappings, and compute LLM cost
         (exec_cost * llm_cache_bonus + model_init).
      2. CPU side: based on the GPU plan, auto-fill the necessary
         executable CPU nodes for this epoch (no cost, no capacity cap).
      3. cpu_load_cost = sum of DB-query costs for the chosen GPU nodes
         (prefer EXPLAIN raw_cost times cache_multiplier), plus HTTP sleep
         costs from CPU nodes.
      4. Total cost per epoch = global_epoch_penalty + gpu_cost
         + cpu_load_weight(epoch) * cpu_load_cost
         (cpu_load_cost already accounts for ordering effects).
      5. The deeper the topological depth of the chosen GPU subgraph, the
         larger the additional depth penalty added to epoch cost.
    """

    def __init__(
        self,
        *,
        graph: GraphSpec,
        dependencies: Mapping[str, Sequence[str]],
        workers: dict[str, Worker],
        worker_ids: tuple[str, ...],
        plan_choices: Mapping[tuple[str, str], Sequence[QueryPlanChoice]],
        window_size: int,
        exec_cost_fn,
        cache_multiplier_fn: Callable[
            [tuple[QuerySignature, ...], QueryPlanChoice], float
        ],
        model_init_cost_fn: Callable[[Node, str | None], float] | None = None,
        llm_cache_bonus_fn: (
            Callable[[Node, str | None, Sequence[str]], float] | None
        ) = None,
        node_ids: Sequence[str] | None = None,
        epoch_penalty_fn: Callable[[int], float] | None = None,
        cpu_load_cost_weight: float | None = None,
        cpu_load_early_weight: float | None = None,
        cpu_cost_mode: str | None = None,
        switch_penalty_weight: float | None = None,
        gpu_cost_max_weight: float | None = None,
        gpu_cost_sum_weight: float | None = None,
        disable_epoch_batch_cost: bool | None = None,
        disable_cpu_load_cost: bool | None = None,
        node_worker_options: Mapping[str, Sequence[str]] | None = None,
        gpu_worker_ids: tuple[str, ...] | None = None,
        cpu_worker_ids: tuple[str, ...] | None = None,
        http_latency_s: Mapping[str, float] | None = None,
        # Optimization options
        enable_batch_shape_pruning: bool = True,
        gpu_batch_slack: int = 1,
        enable_lower_bound_pruning: bool = True,
        enable_worker_symmetry: bool = True,
        lower_bound_cost_factor: float = 0.2,
        gpu_depth_cost_weight: float | None = None,
        debug_log: bool = True,
        debug_every: int = 100000,
        progress_enabled: bool = True,
        progress_interval_s: float = 5.0,
        progress_logger: Callable[[str], None] | None = None,
    ) -> None:
        self.graph = graph
        self.dependencies = dependencies
        self.workers = workers
        self.worker_ids = worker_ids
        self.exec_cost_fn = exec_cost_fn
        self.plan_choices = plan_choices or {}
        self.window_size = window_size
        self.cache_multiplier_fn = cache_multiplier_fn
        self.model_init_cost_fn = model_init_cost_fn or (lambda node, last_model: 0.0)
        self.llm_cache_bonus_fn = llm_cache_bonus_fn or (
            lambda node, last_node, parents: 1.0
        )
        self._http_latency_s = dict(http_latency_s or {})
        # Per-epoch penalty may grow with depth.
        self._epoch_penalty_fn = epoch_penalty_fn or (lambda epoch: 1.0)

        if cpu_load_cost_weight is None:
            cpu_load_cost_weight = 1.0
        self.cpu_load_cost_weight = max(0.0, float(cpu_load_cost_weight))
        if cpu_load_early_weight is None:
            cpu_load_early_weight = 2.0
        self.cpu_load_early_weight = max(0.0, float(cpu_load_early_weight))
        self.disable_epoch_batch_cost = bool(disable_epoch_batch_cost)
        self.disable_cpu_load_cost = bool(disable_cpu_load_cost)
        self.cpu_cost_mode = self._resolve_cpu_cost_mode(cpu_cost_mode)
        max_weight = gpu_cost_max_weight
        sum_weight = gpu_cost_sum_weight
        if max_weight is None and sum_weight is None:
            max_weight = 0.9
            sum_weight = 0.1
        if max_weight is None:
            max_weight = 0.0
        if sum_weight is None:
            sum_weight = 0.0
        total = float(max_weight) + float(sum_weight)
        if total <= 0:
            max_weight = 0.9
            sum_weight = 0.1
        else:
            max_weight = float(max_weight) / total
            sum_weight = float(sum_weight) / total
        self.gpu_cost_max_weight = max(0.0, float(max_weight))
        self.gpu_cost_sum_weight = max(0.0, float(sum_weight))
        if switch_penalty_weight is None:
            switch_penalty_weight = 0.1
        self.switch_penalty_weight = max(0.0, float(switch_penalty_weight))
        self.enable_batch_shape_pruning = enable_batch_shape_pruning
        self.gpu_batch_slack = max(0, gpu_batch_slack)
        self.enable_lower_bound_pruning = enable_lower_bound_pruning
        self.enable_worker_symmetry = enable_worker_symmetry
        self.lower_bound_cost_factor = max(0.0, float(lower_bound_cost_factor))
        self.debug_log = bool(debug_log)
        self.debug_every = max(1, int(debug_every))
        self._progress_logger = progress_logger
        self._progress_enabled = bool(progress_enabled)
        progress_interval = float(progress_interval_s)
        if not math.isfinite(progress_interval) or progress_interval <= 0:
            progress_interval = 5.0
        self._progress_interval_s = progress_interval
        self._progress_start = time.perf_counter()
        self._progress_last = self._progress_start
        self._progress_last_states = 0

        node_worker_options = node_worker_options or {}
        if node_ids is not None:
            self.node_ids = tuple(node_ids)
        else:
            self.node_ids = tuple(sorted(graph.nodes))
        self.node_worker_options = {
            node_id: tuple(node_worker_options.get(node_id, worker_ids))
            for node_id in self.node_ids
        }
        self._gpu_node_ids = tuple(
            nid for nid in self.node_ids if self._is_gpu_node(nid)
        )
        self._db_node_ids = tuple(
            nid for nid in self.node_ids if not self._is_gpu_node(nid)
        )

        if gpu_worker_ids is not None and cpu_worker_ids is not None:
            self.gpu_worker_ids = gpu_worker_ids
            self.cpu_worker_ids = cpu_worker_ids
        else:
            self.gpu_worker_ids = tuple(
                wid for wid in worker_ids if workers[wid].kind == "gpu"
            )
            self.cpu_worker_ids = tuple(
                wid for wid in worker_ids if workers[wid].kind != "gpu"
            )
        if not self.cpu_worker_ids:
            # Fall back to using all workers when no CPU worker is explicit.
            self.cpu_worker_ids = worker_ids

        # Record the topological order of CPU nodes for auto-fill batching.
        try:
            topo = topological_order(self.dependencies, self.node_ids)
        except ValueError:
            topo = list(self.node_ids)
        self._db_topo_order = tuple(nid for nid in topo if not self._is_gpu_node(nid))

        self.node_index = {node_id: idx for idx, node_id in enumerate(self.node_ids)}
        self.all_mask = (1 << len(self.node_ids)) - 1
        self.worker_index = {wid: idx for idx, wid in enumerate(worker_ids)}
        self._worker_id_by_idx = tuple(worker_ids)

        # Fallback plan: used when a query has no plan_choices.
        self._fallback_choice = QueryPlanChoice(
            plan_id="default",
            description="fallback",
            cost=1.0,
            raw_cost=None,
            explain_json=None,
            footprints={},
        )
        depth_weight = 1.0 if gpu_depth_cost_weight is None else gpu_depth_cost_weight
        self._gpu_depth_cost_weight = max(0.0, float(depth_weight))

        # ID compaction: map node/model/query/plan/footprint keys to ints to
        # reduce hashing overhead.
        self._none_id = -1
        self._node_id_to_int = dict(self.node_index)
        self._node_int_to_id = tuple(self.node_ids)
        self._parents_mask = self._build_parents_mask(self.dependencies)
        self._gpu_parents_mask = self._build_parents_mask(
            self.dependencies, gpu_only=True
        )

        all_models_set: set[str] = set()
        for node in graph.nodes.values():
            model_name = getattr(node, "model", None)
            if isinstance(model_name, str) and model_name:
                all_models_set.add(model_name)
        all_models = sorted(all_models_set)
        self._model_to_int = {name: i for i, name in enumerate(all_models)}
        self._model_int_to_name = tuple(all_models)

        all_query_names: set[str] = set()
        for node in graph.nodes.values():
            for q in getattr(node, "db_queries", []) or []:
                all_query_names.add(q.name)
        self._query_to_int = {name: i for i, name in enumerate(sorted(all_query_names))}
        self._query_int_to_name = tuple(sorted(all_query_names))

        all_plan_ids: set[str] = {"default"}
        for choice_list in self.plan_choices.values():
            for choice in choice_list:
                all_plan_ids.add(choice.plan_id)
        self._plan_to_int = {pid: i for i, pid in enumerate(sorted(all_plan_ids))}
        self._plan_int_to_id = tuple(sorted(all_plan_ids))

        all_fp_keys: set[str] = set()
        for choice_list in self.plan_choices.values():
            for choice in choice_list:
                all_fp_keys.update((choice.footprints or {}).keys())
        self._fp_to_int = {k: i for i, k in enumerate(sorted(all_fp_keys))}
        self._fp_int_to_key = tuple(sorted(all_fp_keys))

        self._signature_to_id: dict[
            tuple[int, int, int, tuple[tuple[int, int], ...]], int
        ] = {}
        self._id_to_signature: list[QuerySignature] = []
        self._node_min_cost: dict[str, float] = {}

        # cache: (node_id_int, enter_window) -> all query-plan combinations
        self._query_plan_cache: dict[
            tuple[int, tuple[int, ...]],
            tuple[
                tuple[
                    float,
                    tuple[int, ...],
                    tuple[tuple[str, QueryPlanChoice], ...],
                ],
                ...,
            ],
        ] = {}
        self._lower_bound_cache: dict[tuple[int, int], float] = {}
        self._window_sig_cache: dict[tuple[int, ...], tuple[QuerySignature, ...]] = {}
        self._node_min_cost = self._precompute_node_min_cost()
        self._solve_calls = 0
        self._global_best_cost = float("inf")
        self._memo_hits = 0

        # memo: (done_mask, worker_states, epoch_idx) -> (cost, schedule_rel, plans)
        self._memo: dict[
            tuple[int, tuple[WorkerState, ...], int],
            tuple[
                float,
                list[tuple[int, str, str]],
                dict[tuple[str, str], QueryPlanChoice],
            ],
        ] = {}

        self._cpu_dep_counts = self._precompute_cpu_dep_counts()

    # === Public interface ===

    def solve(
        self,
        initial_worker_states: tuple[WorkerState, ...] | None = None,
    ) -> tuple[
        float,
        list[tuple[int, str, str]],
        dict[tuple[str, str], QueryPlanChoice],
    ]:
        """Returns:
        - best_cost: float
        - schedule: list[(epoch, worker_id, node_id)]
        - best_plans: {(node_id, query_name) -> QueryPlanChoice}
        """
        if initial_worker_states is None:
            initial_worker_states = tuple(
                WorkerState(
                    worker_idx=self.worker_index[wid],
                    last_model_id=-1,
                    last_node_id=-1,
                )
                for wid in self.worker_ids
            )

        if self._progress_enabled:
            self._emit_progress_log(
                f"[DP][start] nodes={len(self.node_ids)} "
                f"gpu_nodes={len(self._gpu_node_ids)} "
                f"cpu_nodes={len(self._db_node_ids)} "
                f"workers={len(self.worker_ids)} "
                f"gpu_workers={len(self.gpu_worker_ids)} "
                f"cpu_workers={len(self.cpu_worker_ids)} "
                f"db_queries={len(self.plan_choices)} "
                f"progress_interval_s={self._progress_interval_s:.2f}"
            )

        best_cost, schedule_rel, plans = self._solve(
            done_mask=0,
            worker_states=initial_worker_states,
            epoch_idx=0,
        )
        return best_cost, schedule_rel, plans

    # === DP main loop: state excludes epoch; epoch depth is implicit via recursion. ===

    def _solve(
        self,
        *,
        done_mask: int,
        worker_states: tuple[WorkerState, ...],
        epoch_idx: int,
        allow_relax: bool = True,
    ) -> tuple[
        float,
        list[tuple[int, str, str]],
        dict[tuple[str, str], QueryPlanChoice],
    ]:
        self._solve_calls += 1
        if self.debug_log and self._solve_calls % self.debug_every == 0:
            done_cnt = done_mask.bit_count()
            self._emit_progress_log(
                f"[DP][state {self._solve_calls}] done={done_cnt}/{len(self.node_ids)} "
                f"memo={len(self._memo)}"
            )
        self._maybe_log_progress(done_mask, epoch_idx)
        if done_mask == self.all_mask:
            return 0.0, [], {}

        key = (done_mask, worker_states, epoch_idx)
        if key in self._memo:
            self._memo_hits += 1
            return self._memo[key]

        best_cost = float("inf")
        best_schedule_rel: list[tuple[int, str, str]] = []
        best_plans: dict[tuple[str, str], QueryPlanChoice] = {}

        gpu_batches = self._enumerate_gpu_batches(done_mask)

        for gpu_nodes in gpu_batches:
            gpu_depths = self._batch_gpu_depths(gpu_nodes)
            if gpu_depths:
                gpu_nodes = sorted(gpu_nodes, key=lambda nid: gpu_depths.get(nid, 0))
            cpu_nodes = self._auto_cpu_batch(done_mask, gpu_nodes)
            cpu_nodes = self._order_cpu_nodes_by_parent_depth(cpu_nodes, gpu_depths)
            if not gpu_nodes and not cpu_nodes:
                continue  # At least one node must be selected.

            combined = set(gpu_nodes) | set(cpu_nodes)
            if not self._batch_feasible(combined, done_mask):
                continue
            new_done_mask = done_mask
            for node_id in combined:
                new_done_mask |= 1 << self.node_index[node_id]
            epoch_penalty = (
                0.0
                if self.disable_epoch_batch_cost
                else float(self._epoch_penalty_fn(epoch_idx))
            )
            depth_penalty = (
                0.0
                if self.disable_epoch_batch_cost
                else self._batch_gpu_depth_penalty(gpu_nodes, gpu_depths)
            )
            cpu_load_weight = self.cpu_load_cost_weight
            if self.disable_cpu_load_cost:
                cpu_load_weight = 0.0
            elif self.cpu_load_early_weight > 0:
                cpu_load_weight *= 1.0 + self.cpu_load_early_weight / (
                    1.0 + float(epoch_idx)
                )
            naive_cpu_cost = None
            if self.cpu_cost_mode == "naive":
                naive_cpu_cost = self._naive_cpu_cost(gpu_nodes)

            gpu_assignments = list(self._gpu_assignments(gpu_nodes))
            if self.enable_worker_symmetry:
                gpu_assignments = self._dedup_worker_assignments(
                    gpu_assignments, worker_states
                )

            cpu_assign = self._cpu_assignments(cpu_nodes)

            for g_assign in gpu_assignments:
                if not g_assign and not cpu_assign:
                    continue

                for (
                    gpu_cost_max,
                    gpu_cost_sum,
                    cpu_load_cost,
                    next_states,
                    batch_plans,
                ) in self._assignment_outcomes(g_assign, cpu_nodes, worker_states):
                    if naive_cpu_cost is not None:
                        cpu_load_cost = naive_cpu_cost
                    gpu_cost = self.gpu_cost_max_weight * float(
                        gpu_cost_max
                    ) + self.gpu_cost_sum_weight * float(gpu_cost_sum)
                    epoch_cost = (
                        epoch_penalty
                        + gpu_cost
                        + cpu_load_weight * cpu_load_cost
                        + depth_penalty
                    )

                    if self.enable_lower_bound_pruning and best_cost < float("inf"):
                        remaining_lb = self._lower_bound_remaining(
                            new_done_mask, epoch_idx + 1
                        )
                        if epoch_cost + remaining_lb >= best_cost:
                            continue

                    sub_cost, sub_sched_rel, sub_plans = self._solve(
                        done_mask=new_done_mask,
                        worker_states=next_states,
                        epoch_idx=epoch_idx + 1,
                        allow_relax=allow_relax,
                    )

                    total_cost = epoch_cost + sub_cost
                    if total_cost < best_cost:
                        if self.debug_log and total_cost < self._global_best_cost:
                            self._emit_progress_log(
                                f"[DP][best] cost={total_cost:.4f} "
                                f"epochs={len(sub_sched_rel) + 1} "
                                f"done={new_done_mask.bit_count()}/{len(self.node_ids)}"
                            )
                        self._global_best_cost = min(self._global_best_cost, total_cost)
                        best_cost = total_cost
                        # Mark the current epoch as 0; sub-schedule epochs shift by +1.
                        this_epoch_sched = [(0, wid, nid) for (wid, nid) in g_assign]
                        this_epoch_sched += [(0, wid, nid) for (wid, nid) in cpu_assign]
                        shifted_sub_sched = [
                            (e + 1, wid, nid) for (e, wid, nid) in sub_sched_rel
                        ]
                        best_schedule_rel = this_epoch_sched + shifted_sub_sched

                        # Merge plan/order: each DB node runs once, direct overwrite OK.
                        best_plans = dict(sub_plans)
                        best_plans.update(batch_plans)

        if best_cost == float("inf"):
            if allow_relax and self.enable_batch_shape_pruning:
                # If pruning was too aggressive and yielded no solution, retry without
                # batch-shape pruning (keep symmetry elimination).
                self._emit_progress_log(
                    "DP relaxation: retrying without batch shape pruning (symmetry"
                    " preserved)..."
                )
                orig_batch_prune = self.enable_batch_shape_pruning
                self.enable_batch_shape_pruning = False
                try:
                    return self._solve(
                        done_mask=done_mask,
                        worker_states=worker_states,
                        epoch_idx=epoch_idx,
                        allow_relax=False,
                    )
                finally:
                    self.enable_batch_shape_pruning = orig_batch_prune
            elif self.debug_log:
                self._emit_progress_log(
                    "[DP][fail] No feasible schedule from state"
                    f" done={done_mask.bit_count()}/{len(self.node_ids)};"
                    f" memo={len(self._memo)}"
                )
            raise RuntimeError(
                "DP failed to find a valid schedule; check DAG dependencies."
            )

        self._memo[key] = (best_cost, best_schedule_rel, best_plans)
        return self._memo[key]

    def _emit_progress_log(self, message: str) -> None:
        if self._progress_logger is not None:
            self._progress_logger(message)
            return
        logger.info(message)

    def _maybe_log_progress(self, done_mask: int, epoch_idx: int) -> None:
        if not self._progress_enabled:
            return
        now = time.perf_counter()
        elapsed_since_last = now - self._progress_last
        if elapsed_since_last < self._progress_interval_s:
            return
        elapsed = now - self._progress_start
        state_delta = self._solve_calls - self._progress_last_states
        rate = state_delta / elapsed_since_last if elapsed_since_last > 0 else 0.0
        best = self._global_best_cost if self._global_best_cost < float("inf") else None
        done_cnt = done_mask.bit_count()
        self._emit_progress_log(
            "[DP][progress] states={} memo={} memo_hits={} done={}/{} epoch={} "
            "best={} elapsed={:.1f}s rate={:.1f}/s".format(
                self._solve_calls,
                len(self._memo),
                self._memo_hits,
                done_cnt,
                len(self.node_ids),
                epoch_idx,
                f"{best:.4f}" if best is not None else "n/a",
                elapsed,
                rate,
            )
        )
        self._progress_last = now
        self._progress_last_states = self._solve_calls

    # === Simple lower bound (used for branch-and-bound pruning) ===

    def _lower_bound_remaining(self, done_mask: int, epoch_idx: int) -> float:
        """Optimistic lower bound on remaining cost, derived from the minimum
        number of epochs needed for the remaining GPU / DB nodes."""
        if not self.enable_lower_bound_pruning:
            return 0.0

        cached = self._lower_bound_cache.get((done_mask, epoch_idx))
        if cached is not None:
            return cached

        remaining_gpu = sum(
            1 for node_id in self._gpu_node_ids if not self._is_done(done_mask, node_id)
        )
        remaining_cpu = sum(
            1 for node_id in self._db_node_ids if not self._is_done(done_mask, node_id)
        )
        if remaining_gpu == 0 and remaining_cpu == 0:
            self._lower_bound_cache[(done_mask, epoch_idx)] = 0.0
            return 0.0

        max_gpu_per_epoch = max(1, len(self.gpu_worker_ids))
        min_epoch_gpu = (
            math.ceil(remaining_gpu / max_gpu_per_epoch) if remaining_gpu else 0
        )
        min_epoch_cpu = 1 if remaining_cpu else 0
        min_epochs = max(min_epoch_gpu, min_epoch_cpu)
        # Rough cost lower bound: minimum load amortized over resource
        # capacity, then multiplied by a discount factor to avoid
        # overestimating parallelism.
        remaining_gpu_nodes = [
            nid for nid in self._gpu_node_ids if not self._is_done(done_mask, nid)
        ]
        gpu_min_sum = sum(
            self._node_min_cost.get(nid, 0.0) for nid in remaining_gpu_nodes
        )
        gpu_capacity = max(1, len(self.gpu_worker_ids))
        load_lb = gpu_min_sum / gpu_capacity if gpu_min_sum > 0 else 0.0
        if self.disable_epoch_batch_cost:
            penalty_lb = 0.0
        else:
            penalty_lb = sum(
                float(self._epoch_penalty_fn(epoch_idx + i)) for i in range(min_epochs)
            )
        lb = penalty_lb + self.lower_bound_cost_factor * load_lb
        self._lower_bound_cache[(done_mask, epoch_idx)] = lb
        return lb

    def _resolve_cpu_cost_mode(self, mode: str | None) -> str:
        raw = (mode or "default").strip().lower()
        if raw in ("default", "naive"):
            return raw
        raise ValueError(
            f"Unsupported cpu_cost_mode '{mode}'. Expected 'default' or 'naive'."
        )

    def _precompute_cpu_dep_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node_id in self._gpu_node_ids:
            counts[node_id] = len(self._cpu_needed_for_gpu([node_id]))
        return counts

    def _precompute_node_min_cost(self) -> dict[str, float]:
        """Estimate each node's minimum single-machine cost (GPU nodes include
        a CPU-load cost lower bound)."""
        res: dict[str, float] = {}
        for node_id in self.node_ids:
            node = self.graph.nodes[node_id]
            if not self._is_gpu_node(node_id):
                res[node_id] = 0.0
                continue
            # Base exec cost (assuming the smallest worker capacity).
            min_worker_cost = float("inf")
            allowed_workers = self.node_worker_options.get(node_id, self.worker_ids)
            for wid in allowed_workers:
                worker = self.workers[wid]
                try:
                    cost = float(self.exec_cost_fn(node, worker))
                except Exception:
                    cost = float("inf")
                min_worker_cost = min(min_worker_cost, cost)
            if min_worker_cost == float("inf"):
                min_worker_cost = 0.0
            # Cheapest plan cost (ignoring the cache multiplier).
            min_plan_cost = 0.0
            for q in getattr(node, "db_queries", []) or []:
                choices = self.plan_choices.get((node.id, q.name), ())
                if not choices:
                    choices = (self._fallback_choice,)
                best = float("inf")
                for choice in choices:
                    if choice.raw_cost is not None:
                        base = max(0.0, float(choice.raw_cost))
                    elif choice.cost is not None:
                        base = float(choice.cost)
                    else:
                        base = 1.0
                    best = min(best, base)
                if best == float("inf"):
                    best = 0.0
                min_plan_cost += best
            if self.disable_cpu_load_cost or self.cpu_cost_mode == "naive":
                min_plan_cost = 0.0
            else:
                min_plan_cost *= self.cpu_load_cost_weight
            res[node_id] = min_worker_cost + min_plan_cost
        return res

    # === assignment → cost + state update (GPU + CPU-load costs separate) ===

    def _assignment_outcomes(
        self,
        assign: Sequence[tuple[str, str]],
        cpu_nodes: Sequence[str],
        worker_states: tuple[WorkerState, ...],
    ) -> Iterable[
        tuple[
            float,
            float,
            float,
            tuple[WorkerState, ...],
            dict[tuple[str, str], QueryPlanChoice],
        ]
    ]:
        """Given a set of (worker_id, node_id) pairs, enumerate:
        - GPU cost (LLM makespan)
        - GPU cost sum (total LLM duration)
        - CPU load cost (GPU node DB queries accumulated + CPU node HTTP sleep)
        - updated worker_states
        - plan choices for this batch
        """

        @cache
        def compute(
            assign_key: tuple[tuple[str, str], ...],
            worker_states_key: tuple[WorkerState, ...],
            cpu_key: tuple[str, ...],
        ) -> tuple[
            tuple[
                float,
                float,
                float,
                tuple[WorkerState, ...],
                dict[tuple[str, str], QueryPlanChoice],
            ],
            ...,
        ]:
            results: list[
                tuple[
                    float,
                    float,
                    float,
                    tuple[WorkerState, ...],
                    dict[tuple[str, str], QueryPlanChoice],
                ]
            ] = []

            def dfs(
                idx: int,
                current_states: list[WorkerState],
                worker_costs: list[float],
                acc_plans: dict[tuple[str, str], QueryPlanChoice],
                cpu_load_cost: float,
                window: tuple[int, ...],
            ) -> None:
                if idx >= len(assign_key):
                    cpu_load_cost_extra, cpu_plans = self._apply_cpu_load(
                        cpu_key, window
                    )
                    gpu_costs = [
                        worker_costs[self.worker_index[wid]]
                        for wid in self.gpu_worker_ids
                    ]
                    gpu_cost_max = max(gpu_costs) if gpu_costs else 0.0
                    gpu_cost_sum = sum(gpu_costs) if gpu_costs else 0.0
                    merged_plans = dict(acc_plans)
                    merged_plans.update(cpu_plans)
                    results.append(
                        (
                            gpu_cost_max,
                            gpu_cost_sum,
                            float(cpu_load_cost + cpu_load_cost_extra),
                            tuple(current_states),
                            merged_plans,
                        )
                    )
                    return

                worker_id, node_id = assign_key[idx]
                worker_idx = self.worker_index[worker_id]
                state = current_states[worker_idx]

                for (
                    node_cost,
                    node_cpu_cost,
                    updated_state,
                    plan_map,
                    exit_window,
                ) in self._execute_node(node_id, state, window):
                    current_states[worker_idx] = updated_state

                    new_worker_costs = list(worker_costs)
                    new_worker_costs[worker_idx] += node_cost

                    merged_plans = dict(acc_plans)
                    merged_plans.update(plan_map)

                    dfs(
                        idx + 1,
                        current_states,
                        new_worker_costs,
                        merged_plans,
                        cpu_load_cost + node_cpu_cost,
                        exit_window,
                    )

                    # Backtrack.
                    current_states[worker_idx] = state

            dfs(
                0,
                list(worker_states_key),
                [0.0 for _ in self.worker_ids],
                {},
                0.0,
                tuple(),
            )
            return tuple(results)

        yield from compute(tuple(assign), worker_states, tuple(cpu_nodes))

    # === All execution variants for a single node on a given worker (LLM or DB) ===

    def _execute_node(
        self,
        node_id: str,
        state: WorkerState,
        enter_window: tuple[int, ...],
    ) -> Iterable[
        tuple[
            float,
            float,
            WorkerState,
            dict[tuple[str, str], QueryPlanChoice],
            tuple[int, ...],
        ]
    ]:
        node = self.graph.nodes[node_id]
        parents = tuple(self.dependencies.get(node_id, []))

        worker_id = self._worker_id_by_idx[state.worker_idx]

        # Both LLM and DB use exec_cost_fn; DB estimate cost comes from the query plan.
        exec_cost = self._exec_cost_cached(node_id, worker_id)
        model_cost = self._model_init_cost_cached(node_id, state.last_model_id)
        bonus_multiplier = self._llm_bonus_cached(node_id, state.last_node_id, parents)
        if bonus_multiplier <= 0:
            bonus_multiplier = 1.0

        if self._is_gpu_node(node_id):
            base_cost = exec_cost * bonus_multiplier + model_cost
            queries = getattr(node, "db_queries", []) or []
            if not queries:
                new_state = WorkerState(
                    worker_idx=state.worker_idx,
                    last_model_id=self._model_to_id(
                        node.model, fallback=state.last_model_id
                    ),
                    last_node_id=self._node_id_to_int[node.id],
                )
                yield base_cost, 0.0, new_state, {}, enter_window
                return

            options = self._query_plan_options(node, enter_window)
            for query_cost, exit_window, plan_seq in options:
                new_state = WorkerState(
                    worker_idx=state.worker_idx,
                    last_model_id=self._model_to_id(
                        node.model, fallback=state.last_model_id
                    ),
                    last_node_id=self._node_id_to_int[node.id],
                )
                plan_map = {(node.id, q_name): choice for q_name, choice in plan_seq}
                yield base_cost, query_cost, new_state, plan_map, exit_window
            return

        # Non-GPU node: no cost, no window update (filled in by CPU auto-fill).
        new_state = WorkerState(
            worker_idx=state.worker_idx,
            last_model_id=state.last_model_id,
            last_node_id=self._node_id_to_int[node.id],
        )
        yield 0.0, 0.0, new_state, {}, enter_window

    # === DB queries: enumerate plan combinations (currently assumes a single
    # query per node); window only takes effect within the epoch. ===

    def _query_plan_options(
        self,
        node: Node,
        enter_window: tuple[int, ...],
    ) -> tuple[
        tuple[float, tuple[int, ...], tuple[tuple[str, QueryPlanChoice], ...]],
        ...,
    ]:
        cache_key = (self._node_id_to_int[node.id], enter_window)
        if cache_key in self._query_plan_cache:
            return self._query_plan_cache[cache_key]

        queries = list(getattr(node, "db_queries", []) or [])
        if not queries:
            # No DB query: query cost is 0 and the window is unchanged.
            self._query_plan_cache[cache_key] = ((0.0, enter_window, tuple()),)
            return self._query_plan_cache[cache_key]
        # Each DB node currently has a single query; if there were multiple,
        # we would multiply plan combinations in declaration order without
        # enumerating permutations.
        outcomes: list[
            tuple[float, tuple[int, ...], tuple[tuple[str, QueryPlanChoice], ...]]
        ] = [(0.0, enter_window, tuple())]
        for query in queries:
            choices = self.plan_choices.get((node.id, query.name), ())
            if not choices:
                choices = (self._fallback_choice,)
            next_outcomes: list[
                tuple[float, tuple[int, ...], tuple[tuple[str, QueryPlanChoice], ...]]
            ] = []
            for base_cost, window, plan_seq in outcomes:
                for choice in choices:
                    q_cost = self._query_cost(window, choice)
                    next_window = self._append_window(
                        window, node.id, query.name, choice
                    )
                    next_outcomes.append(
                        (
                            base_cost + q_cost,
                            next_window,
                            plan_seq + ((query.name, choice),),
                        )
                    )
            outcomes = next_outcomes

        results = tuple(outcomes)
        self._query_plan_cache[cache_key] = results
        return results

    def _query_cost(
        self,
        window: tuple[int, ...],
        choice: QueryPlanChoice,
    ) -> float:
        if choice.raw_cost is not None:
            base = max(0.0, float(choice.raw_cost))
        elif choice.cost is not None:
            base = float(choice.cost)
        else:
            base = 1.0
        return base * self.cache_multiplier_fn(self._window_signatures(window), choice)

    def _append_window(
        self,
        window: tuple[int, ...],
        node_id: str,
        query_name: str,
        choice: QueryPlanChoice,
    ) -> tuple[int, ...]:
        if self.window_size <= 0:
            return window
        sig_id = self._signature_id(node_id, query_name, choice)
        if not window:
            if self.window_size == 1:
                return (sig_id,)
            return (self._none_id,) * (self.window_size - 1) + (sig_id,)
        new_window = window + (sig_id,)
        if len(new_window) > self.window_size:
            new_window = new_window[-self.window_size :]
        return new_window

    def _batch_gpu_depths(self, gpu_nodes: Sequence[str]) -> dict[str, int]:
        """Topological depth within the chosen GPU subgraph for this epoch (root=0)."""
        if not gpu_nodes:
            return {}
        selected_list = list(gpu_nodes)
        selected = set(selected_list)
        indeg: dict[str, int] = {nid: 0 for nid in selected_list}
        children: dict[str, list[str]] = {nid: [] for nid in selected_list}
        for nid in selected_list:
            for parent in self.dependencies.get(nid, ()):
                if parent not in selected:
                    continue
                indeg[nid] += 1
                children[parent].append(nid)

        queue = [nid for nid in selected_list if indeg[nid] == 0]
        depths: dict[str, int] = {nid: 0 for nid in queue}
        while queue:
            nid = queue.pop(0)
            base = depths.get(nid, 0)
            for child in children.get(nid, []):
                depths[child] = max(depths.get(child, 0), base + 1)
                indeg[child] -= 1
                if indeg[child] == 0:
                    queue.append(child)
        for nid in selected_list:
            depths.setdefault(nid, 0)
        return depths

    def _batch_gpu_depth_penalty(
        self, gpu_nodes: Sequence[str], gpu_depths: Mapping[str, int] | None = None
    ) -> float:
        """Penalty derived from the depth of the chosen GPU subgraph this epoch."""
        if self._gpu_depth_cost_weight <= 0 or not gpu_nodes:
            return 0.0
        depths = (
            gpu_depths if gpu_depths is not None else self._batch_gpu_depths(gpu_nodes)
        )
        total_depth = sum(depths.get(nid, 0) for nid in gpu_nodes)
        return self._gpu_depth_cost_weight * float(total_depth)

    def _apply_cpu_load(
        self,
        cpu_nodes: Sequence[str],
        enter_window: tuple[int, ...],
    ) -> tuple[float, dict[tuple[str, str], QueryPlanChoice]]:
        """Accumulate CPU load cost in the given order (DB plan + HTTP sleep);
        the window is only valid within the epoch."""
        if not cpu_nodes:
            return 0.0, {}
        window = enter_window
        total_cost = 0.0
        plan_map: dict[tuple[str, str], QueryPlanChoice] = {}
        for node_id in cpu_nodes:
            node = self.graph.nodes[node_id]
            if node.engine == "http":
                total_cost += self._http_sleep_cost(node)
                continue
            if not getattr(node, "db_queries", None):
                continue
            options = self._query_plan_options(node, window)
            best_cost = float("inf")
            best_window = window
            best_seq: tuple[tuple[str, QueryPlanChoice], ...] = tuple()
            for cost, exit_window, plan_seq in options:
                if cost < best_cost:
                    best_cost = cost
                    best_window = exit_window
                    best_seq = plan_seq
            if best_cost == float("inf"):
                continue
            total_cost += float(best_cost)
            for q_name, choice in best_seq:
                plan_map[(node.id, q_name)] = choice
            window = best_window
        return total_cost, plan_map

    def _http_sleep_cost(self, node: Node) -> float:
        if self._http_latency_s:
            profiled = self._http_latency_s.get(node.id)
            if profiled is not None:
                try:
                    return max(0.0, float(profiled))
                except (TypeError, ValueError):
                    pass
        raw = node.raw if isinstance(node.raw, dict) else {}
        for key, scale in (
            ("sleep_s", 1.0),
            ("sleep_ms", 0.001),
            ("latency_s", 1.0),
            ("latency_ms", 0.001),
            ("timeout_s", 1.0),
            ("timeout_ms", 0.001),
        ):
            if key not in raw:
                continue
            value = raw.get(key)
            if isinstance(value, str):
                value = value.strip()
            if not isinstance(value, (str, int, float)):
                continue
            try:
                seconds = float(value) * scale
            except (TypeError, ValueError):
                continue
            return max(0.0, seconds)
        return 0.0

    def _order_cpu_nodes_by_parent_depth(
        self, cpu_nodes: Sequence[str], gpu_depths: Mapping[str, int]
    ) -> list[str]:
        if not cpu_nodes or not gpu_depths:
            return list(cpu_nodes)

        def key(node_id: str) -> tuple[bool, int]:
            node = self.graph.nodes.get(node_id)
            parent = None
            if node is not None and isinstance(node.raw, dict):
                parent = node.raw.get("parent")
            depth = gpu_depths.get(parent) if parent is not None else None
            return (depth is None, depth or 0)

        return sorted(cpu_nodes, key=key)

    # === ID mapping & window/signature helpers ===

    def _model_to_id(self, model: str | None, fallback: int | None = None) -> int:
        if model is None:
            return self._none_id if fallback is None else fallback
        return self._model_to_int.get(
            model, self._none_id if fallback is None else fallback
        )

    def _model_from_id(self, model_id: int) -> str | None:
        if model_id == self._none_id:
            return None
        if 0 <= model_id < len(self._model_int_to_name):
            return self._model_int_to_name[model_id]
        return None

    def _node_from_id(self, node_id: int) -> str | None:
        if node_id == self._none_id:
            return None
        if 0 <= node_id < len(self._node_int_to_id):
            return self._node_int_to_id[node_id]
        return None

    def _signature_id(
        self, node_id: str, query_name: str, choice: QueryPlanChoice
    ) -> int:
        fp_tuple: tuple[tuple[int, int], ...] = tuple(
            sorted(
                (self._fp_to_int.get(k, self._none_id), int(v))
                for k, v in (choice.footprints or {}).items()
            )
        )
        key = (
            self._node_id_to_int[node_id],
            self._query_to_int.get(query_name, self._none_id),
            self._plan_to_int.get(choice.plan_id, self._none_id),
            fp_tuple,
        )
        cached = self._signature_to_id.get(key)
        if cached is not None:
            return cached
        sig = QuerySignature(
            node_id=node_id,
            query_name=query_name,
            plan_id=choice.plan_id,
            footprints=tuple(sorted((choice.footprints or {}).items())),
        )
        sig_id = len(self._id_to_signature)
        self._signature_to_id[key] = sig_id
        self._id_to_signature.append(sig)
        return sig_id

    def _window_signatures(self, window: tuple[int, ...]) -> tuple[QuerySignature, ...]:
        cached = self._window_sig_cache.get(window)
        if cached is not None:
            return cached
        sigs = tuple(
            self._id_to_signature[sig_id]
            for sig_id in window
            if sig_id != self._none_id
        )
        self._window_sig_cache[window] = sigs
        return sigs

    def _build_parents_mask(
        self, deps: Mapping[str, Sequence[str]], gpu_only: bool = False
    ) -> tuple[int, ...]:
        masks = [0] * len(self.node_ids)
        for nid, idx in self.node_index.items():
            mask = 0
            for parent in deps.get(nid, []):
                if parent not in self.node_index:
                    continue
                if gpu_only and not self._is_gpu_node(parent):
                    continue
                pidx = self.node_index[parent]
                mask |= 1 << pidx
            masks[idx] = mask
        return tuple(masks)

    @cache
    def _exec_cost_cached(self, node_id: str, worker_id: str) -> float:
        node = self.graph.nodes[node_id]
        worker = self.workers[worker_id]
        return float(self.exec_cost_fn(node, worker))

    @cache
    def _model_init_cost_cached(self, node_id: str, last_model_id: int) -> float:
        node = self.graph.nodes[node_id]
        last_model = self._model_from_id(last_model_id)
        return float(self.model_init_cost_fn(node, last_model))

    @cache
    def _llm_bonus_cached(
        self,
        node_id: str,
        last_node_id: int,
        parents: tuple[str, ...],
    ) -> float:
        node = self.graph.nodes[node_id]
        last_node = self._node_from_id(last_node_id)
        return float(self.llm_cache_bonus_fn(node, last_node, parents))

    # === Worker symmetry elimination ===

    def _worker_signature(self, worker_idx: int, state: WorkerState) -> tuple:
        worker = self.workers[self._worker_id_by_idx[worker_idx]]
        return (
            worker.kind,
            float(worker.capacity),
            state.last_model_id,
            state.last_node_id,
        )

    def _dedup_worker_assignments(
        self,
        assignments: list[list[tuple[str, str]]],
        worker_states: tuple[WorkerState, ...],
    ) -> list[list[tuple[str, str]]]:
        """Eliminate symmetric mappings across isomorphic workers (drop duplicates)."""
        seen: set[tuple] = set()
        uniq: list[list[tuple[str, str]]] = []
        for assign in assignments:
            key = self._assignment_signature(assign, worker_states)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(assign)
        return uniq

    def _assignment_signature(
        self, assign: list[tuple[str, str]], worker_states: tuple[WorkerState, ...]
    ) -> tuple:
        buckets: dict[tuple, list[str]] = {}
        for wid, node_id in assign:
            idx = self.worker_index[wid]
            sig = self._worker_signature(idx, worker_states[idx])
            buckets.setdefault(sig, []).append(node_id)
        items = []
        for sig, nodes in buckets.items():
            items.append((sig, tuple(sorted(nodes))))
        items.sort(key=lambda x: x[0])
        return tuple(items)

    # === GPU batch enumeration ===

    def _enumerate_gpu_batches(self, done_mask: int) -> list[list[str]]:
        """Enumerate the feasible GPU-side subgraphs for the current epoch
        (capped by GPU count; may be empty)."""
        max_gpu = max(1, len(self.gpu_worker_ids))
        pending = [
            node_id
            for node_id in self.node_ids
            if not self._is_done(done_mask, node_id) and self._is_gpu_node(node_id)
        ]
        if not pending:
            return [[]]

        # Topo-sort the GPU subgraph first so parents come before children;
        # enumeration then only emits dependency-closed subsets.
        pending_list = list(pending)
        pending_set = set(pending_list)
        in_deg = {nid: 0 for nid in pending_list}
        for nid in pending_list:
            for parent in self.dependencies.get(nid, []):
                if parent in pending_set:
                    in_deg[nid] += 1
        queue = [nid for nid in pending_list if in_deg[nid] == 0]
        topo: list[str] = []
        while queue:
            nid = queue.pop(0)
            topo.append(nid)
            for child in pending_list:
                if nid in self.dependencies.get(child, []):
                    in_deg[child] -= 1
                    if in_deg[child] == 0:
                        queue.append(child)
        if len(topo) != len(pending_set):
            # If there's a cycle or malformed dependency, fall back to the
            # original order to avoid crashing.
            topo = pending_list

        target = min(max_gpu, len(pending))
        min_r = 1
        max_r = min(max_gpu, len(pending))
        if self.enable_batch_shape_pruning:
            min_r = max(1, target - self.gpu_batch_slack)
            max_r = target

        results: list[list[str]] = [[]]

        def dfs(idx: int, selected: list[str], selected_mask: int) -> None:
            if idx >= len(topo):
                size = len(selected)
                if size >= min_r and size <= max_r:
                    results.append(list(selected))
                return

            # Skip the current node.
            dfs(idx + 1, selected, selected_mask)

            # Try selecting the current node (parents must already be done
            # or part of the current selection).
            if len(selected) >= max_r:
                return
            node_id = topo[idx]
            node_idx = self.node_index[node_id]
            gpu_parents_mask = self._gpu_parents_mask[node_idx]
            if gpu_parents_mask & ~(done_mask | selected_mask):
                dfs(idx + 1, selected, selected_mask)
                return

            selected.append(node_id)
            dfs(idx + 1, selected, selected_mask | (1 << node_idx))
            selected.pop()

        dfs(0, [], 0)
        if pending and len(results) == 1 and results[0] == []:
            raise RuntimeError(
                "DP GPU batch enumeration produced no executable GPU batch "
                f"despite pending nodes: pending={pending}"
            )
        return results

    def _auto_cpu_batch(self, done_mask: int, gpu_nodes: Sequence[str]) -> list[str]:
        """Auto-fill the CPU nodes needed for this epoch, ignoring capacity / cost."""
        if not self._db_node_ids:
            return []
        batch_mask = done_mask
        for node_id in gpu_nodes:
            batch_mask |= 1 << self.node_index[node_id]

        needed = self._cpu_needed_for_gpu(gpu_nodes)
        needed = self._expand_cpu_parents(needed)
        pending = {
            nid
            for nid in self._db_node_ids
            if not self._is_done(done_mask, nid) and nid in needed
        }
        if not pending:
            return []

        selected: list[str] = []
        progressed = True
        while progressed:
            progressed = False
            for nid in self._db_topo_order:
                if nid not in pending:
                    continue
                idx = self.node_index[nid]
                pmask = self._parents_mask[idx]
                if pmask & ~batch_mask:
                    continue
                selected.append(nid)
                pending.remove(nid)
                batch_mask |= 1 << idx
                progressed = True
        return selected

    def _cpu_needed_for_gpu(self, gpu_nodes: Sequence[str]) -> set[str]:
        if not gpu_nodes:
            return set()
        needed: set[str] = set()
        stack = list(gpu_nodes)
        visited: set[str] = set()
        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            for parent in self.dependencies.get(node_id, ()):
                if parent not in visited:
                    stack.append(parent)
                if parent in self._db_node_ids:
                    needed.add(parent)
        return needed

    def _expand_cpu_parents(self, cpu_nodes: set[str]) -> set[str]:
        if not cpu_nodes:
            return set()
        expanded = set(cpu_nodes)
        stack = list(cpu_nodes)
        while stack:
            node_id = stack.pop()
            for parent in self.dependencies.get(node_id, ()):
                if parent in self._db_node_ids and parent not in expanded:
                    expanded.add(parent)
                    stack.append(parent)
        return expanded

    def _cpu_assignments(self, cpu_nodes: Sequence[str]) -> list[tuple[str, str]]:
        if not cpu_nodes:
            return []
        workers = self.cpu_worker_ids or self.worker_ids
        if not workers:
            return []
        seq: list[tuple[str, str]] = []
        for idx, node_id in enumerate(cpu_nodes):
            wid = workers[idx % len(workers)]
            seq.append((wid, node_id))
        return seq

    def _naive_cpu_cost(self, gpu_nodes: Sequence[str]) -> float:
        if not gpu_nodes:
            return 0.0
        return float(sum(self._cpu_dep_counts.get(node_id, 0) for node_id in gpu_nodes))

    # === Worker assignment (GPU) ===
    def _gpu_assignments(
        self,
        gpu_nodes: Sequence[str],
    ) -> Iterable[list[tuple[str, str]]]:
        """Map GPU nodes to GPU workers (each worker holds at most one LLM node)."""
        if not gpu_nodes:
            return [[]]

        results: list[list[tuple[str, str]]] = []

        def dfs(idx: int, used: set[str], current: list[tuple[str, str]]) -> None:
            if idx >= len(gpu_nodes):
                results.append(list(current))
                return
            node_id = gpu_nodes[idx]
            allowed = self.node_worker_options.get(node_id, self.gpu_worker_ids)
            for wid in self.gpu_worker_ids:
                if wid not in allowed:
                    continue
                if wid in used:
                    continue
                used.add(wid)
                current.append((wid, node_id))
                dfs(idx + 1, used, current)
                current.pop()
                used.remove(wid)

        dfs(0, set(), [])
        return results

    # === Helpers ===

    def _batch_feasible(self, nodes: set[str], done_mask: int) -> bool:
        """Check whether a node set within one epoch satisfies dependencies
        (every parent is done or also in the batch)."""
        batch_mask = done_mask
        for node_id in nodes:
            batch_mask |= 1 << self.node_index[node_id]
        for node_id in nodes:
            idx = self.node_index[node_id]
            pmask = self._parents_mask[idx]
            if pmask & ~batch_mask:
                return False
        return True

    def _is_done(self, done_mask: int, node_id: str) -> bool:
        idx = self.node_index[node_id]
        return bool(done_mask & (1 << idx))

    def _is_gpu_node(self, node_id: str) -> bool:
        node = self.graph.nodes[node_id]
        return node.engine == "vllm"
