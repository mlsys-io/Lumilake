import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lumilake import envs

from lumilake_server.data_profile_models import (
    DataProfileCostEstimate,
    DataProfileResultRow,
)
from lumilake_server.runtime.data_profile_utils import (
    coerce_data_profile_footprints,
    data_profile_key_for_node_query,
    normalize_table_name,
)
from lumilake_server.runtime.optimizer.base import BaseOptimizer, Schedule
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp

from .multimodal_cost import MultimodalCostCoefficients, compute_gpu_exec_cost
from .schedule.halo_dp import DPSolver, QuerySignature, WorkerState
from .schedule.model_size import MODEL_SIZE
from .schedule.models import DBQuery, Edge, GraphSpec, Node, QueryPlanChoice, Worker


@dataclass(frozen=True, slots=True)
class HaloTuningContext:
    input_query_count: int
    total_nodes: int
    gpu_nodes: int
    db_nodes: int
    gpu_workers: int
    cpu_workers: int
    unique_models: int
    switch_pressure: float
    dominant_model_size_b: float
    db_query_count: int
    plan_variants_min: int
    plan_variants_avg: float
    plan_variants_max: int
    reuse_signal: float


@dataclass(frozen=True, slots=True)
class HaloCostModelParams:
    model_init_sec_per_b: float
    llm_base_sec_per_b: float
    llm_input_sec: float
    db_input_sec: float


@dataclass(frozen=True, slots=True)
class HaloDPTuningParams:
    window_size: int
    cpu_load_cost_weight: float
    cpu_load_early_weight: float
    cpu_cost_mode: str
    switch_penalty_weight: float
    gpu_cost_max_weight: float
    gpu_cost_sum_weight: float
    disable_epoch_batch_cost: bool
    disable_cpu_load_cost: bool
    enable_batch_shape_pruning: bool
    gpu_batch_slack: int
    enable_lower_bound_pruning: bool
    enable_worker_symmetry: bool
    lower_bound_cost_factor: float
    gpu_depth_cost_weight: float


@dataclass(frozen=True, slots=True)
class HaloResolvedTuning:
    context: HaloTuningContext
    cost_model: HaloCostModelParams
    dp: HaloDPTuningParams


class HaloOptimizer(BaseOptimizer):
    """Halo DP optimizer."""

    def __init__(
        self,
        *,
        logger=None,
        log_level=None,
    ) -> None:
        super().__init__(logger, log_level)
        self._epoch_penalty_weight = 1.0

        self._model_suffix_pattern = re.compile(
            r".+-(\d+(?:\.\d+)?)(?:[\s_-]?)(billion|bn|b|m).*", re.IGNORECASE
        )
        self._model_size_lookup = {
            key.lower(): float(value) for key, value in MODEL_SIZE.items()
        }
        self._default_model_size_b = 20.0
        self._model_init_sec_per_b = 0.75
        self._llm_base_sec_per_b = 0.25
        self._llm_input_sec = 0.15
        self._db_input_sec = 0.05
        self._embed_base_sec_per_b = 0.10
        self._embed_input_sec = 0.08
        self._vlm_base_sec_per_b = 0.30
        self._vlm_input_sec = 0.25
        self._diff_base_sec_per_b = 0.20
        self._diff_prompt_input_sec = 0.05
        self._input_query_count_default = 1
        self._input_query_count = self._input_query_count_default
        self._dp_progress_enabled = True
        self._dp_progress_interval_s = self._safe_positive_float(
            envs.LUMILAKE_POLL_INTERVAL_SECONDS, 5.0
        )
        self._sql_table_pattern = re.compile(
            r"(?:from|join)\s+([^\s,;]+)", re.IGNORECASE
        )

    def generate_schedule(
        self,
        graph: RuntimeGraph,
        worker_names: Sequence[str],
        worker_profiles: Mapping[str, Mapping[str, Any]],
        data_profile_results: (
            Mapping[str, Sequence[Mapping[str, Any] | DataProfileResultRow]] | None
        ) = None,
    ) -> Schedule:
        workers_input = [name.strip() for name in worker_names if name and name.strip()]
        if not workers_input:
            raise ValueError("HaloOptimizer requires at least one worker")
        if not graph.nodes:
            return Schedule(worker_assignment={worker: [] for worker in workers_input})
        self._assert_no_http_nodes(graph)
        self._validate_supported_runtime_nodes(graph)

        resolved_query_count = self._resolve_input_query_count(graph)
        self._input_query_count = resolved_query_count
        self.logger.info(
            "Halo DP scheduling started: nodes=%d workers=%d input_query_count=%d",
            len(graph.nodes),
            len(workers_input),
            self._input_query_count,
        )
        if envs.LUMILAKE_DISABLE_DATA_PROFILE:
            self.logger.info(
                "Halo data profile disabled by LUMILAKE_DISABLE_DATA_PROFILE;"
                " ignoring %d result keys",
                len(data_profile_results) if data_profile_results else 0,
            )
            parsed_data_profile_results: dict[str, tuple[DataProfileResultRow, ...]] = (
                {}
            )
        else:
            parsed_data_profile_results = self._parse_data_profile_results(
                data_profile_results
            )

        try:
            dependencies: dict[str, tuple[str, ...]] = {}
            halo_nodes: dict[str, Node] = {}
            halo_edges: list[Edge] = []
            for node_id in graph.node_order:
                op = graph.nodes[node_id]
                engine = self._map_engine(op.backend, op.task_type)
                deps = tuple(op.dependencies)
                dependencies[node_id] = deps

                db_queries: tuple[DBQuery, ...] = ()
                if engine == "db":
                    sql_template = ""
                    if isinstance(op.data_spec, dict):
                        template = op.data_spec.get("template")
                        if isinstance(template, str):
                            sql_template = template
                    db_queries = (
                        DBQuery(
                            name=f"{node_id}_query",
                            sql=sql_template,
                            parameters={},
                        ),
                    )

                raw = self._build_node_raw(op, deps)

                halo_nodes[node_id] = Node(
                    id=node_id,
                    type=op.task_type,
                    engine=engine,
                    model=op.model,
                    inputs=deps,
                    outputs=tuple(),
                    db_queries=db_queries,
                    raw=raw,
                )
                for dep in deps:
                    halo_edges.append(Edge(source=dep, target=node_id, mapping={}))

            halo_graph = GraphSpec(
                name="lumilake_runtime",
                description="Runtime graph for Halo DP optimizer",
                nodes=halo_nodes,
                edges=tuple(halo_edges),
            )

            workers = self._build_workers(workers_input, worker_profiles, graph)
            worker_ids = tuple(workers.keys())
            self.logger.info(
                "Halo worker pool: total=%d gpu=%d cpu=%d",
                len(workers),
                sum(1 for worker in workers.values() if worker.kind == "gpu"),
                sum(1 for worker in workers.values() if worker.kind != "gpu"),
            )
            node_worker_options = self._build_node_worker_options(halo_graph, workers)
            gpu_worker_ids = tuple(
                wid for wid, worker in workers.items() if worker.kind == "gpu"
            )
            cpu_worker_ids = tuple(
                wid for wid, worker in workers.items() if worker.kind != "gpu"
            )
            if not cpu_worker_ids:
                cpu_worker_ids = worker_ids

            initial_worker_states = tuple(
                WorkerState(
                    worker_idx=idx,
                    last_model_id=-1,
                    last_node_id=-1,
                )
                for idx, _ in enumerate(worker_ids)
            )

            data_profile_plan_choices = self._build_data_profile_plan_choices(
                graph, parsed_data_profile_results
            )
            plan_choice_sizes = [
                len(choices) for choices in data_profile_plan_choices.values()
            ]
            self.logger.info(
                "Halo DP search space summary: gpu_nodes=%d cpu_nodes=%d db_queries=%d "
                "plan_variants[min/avg/max]=%s/%s/%s",
                sum(1 for node in halo_nodes.values() if node.engine == "vllm"),
                sum(1 for node in halo_nodes.values() if node.engine != "vllm"),
                len(data_profile_plan_choices),
                min(plan_choice_sizes) if plan_choice_sizes else 0,
                (
                    f"{(sum(plan_choice_sizes) / len(plan_choice_sizes)):.2f}"
                    if plan_choice_sizes
                    else "0.00"
                ),
                max(plan_choice_sizes) if plan_choice_sizes else 0,
            )
            self.logger.info(
                "Halo data profile plans prepared for %d db queries",
                len(data_profile_plan_choices),
            )
            resolved_tuning = self._resolve_and_apply_tuning(
                halo_graph=halo_graph,
                workers=workers,
                plan_choices=data_profile_plan_choices,
            )
            # Align with Halo: http_latency_s is only for externally profiled latencies.
            # We do not synthesize it from static node specs here.
            http_latency_s: dict[str, float] | None = None
            self.logger.info(
                "Halo DP options: window=%d cpu_cost_mode=%s switch_penalty=%.3f"
                " gpu_max_weight=%.3f gpu_sum_weight=%.3f cpu_load=%.3f cpu_early=%.3f"
                " depth_weight=%.3f batch_pruning=%s batch_slack=%d lower_bound=%s"
                " worker_symmetry=%s disable_epoch_batch_cost=%s"
                " disable_cpu_load_cost=%s http_profiled_nodes=%d",
                resolved_tuning.dp.window_size,
                resolved_tuning.dp.cpu_cost_mode,
                resolved_tuning.dp.switch_penalty_weight,
                resolved_tuning.dp.gpu_cost_max_weight,
                resolved_tuning.dp.gpu_cost_sum_weight,
                resolved_tuning.dp.cpu_load_cost_weight,
                resolved_tuning.dp.cpu_load_early_weight,
                resolved_tuning.dp.gpu_depth_cost_weight,
                resolved_tuning.dp.enable_batch_shape_pruning,
                resolved_tuning.dp.gpu_batch_slack,
                resolved_tuning.dp.enable_lower_bound_pruning,
                resolved_tuning.dp.enable_worker_symmetry,
                resolved_tuning.dp.disable_epoch_batch_cost,
                resolved_tuning.dp.disable_cpu_load_cost,
                0,
            )

            solver = DPSolver(
                graph=halo_graph,
                dependencies=dependencies,
                workers=workers,
                worker_ids=worker_ids,
                plan_choices=data_profile_plan_choices,
                window_size=resolved_tuning.dp.window_size,
                exec_cost_fn=self._exec_cost,
                cache_multiplier_fn=self._cache_multiplier,
                model_init_cost_fn=self._model_init_cost,
                llm_cache_bonus_fn=self._llm_cache_bonus,
                node_ids=tuple(graph.node_order),
                epoch_penalty_fn=self._epoch_penalty,
                cpu_load_cost_weight=resolved_tuning.dp.cpu_load_cost_weight,
                cpu_load_early_weight=resolved_tuning.dp.cpu_load_early_weight,
                cpu_cost_mode=resolved_tuning.dp.cpu_cost_mode,
                switch_penalty_weight=resolved_tuning.dp.switch_penalty_weight,
                gpu_cost_max_weight=resolved_tuning.dp.gpu_cost_max_weight,
                gpu_cost_sum_weight=resolved_tuning.dp.gpu_cost_sum_weight,
                disable_epoch_batch_cost=resolved_tuning.dp.disable_epoch_batch_cost,
                disable_cpu_load_cost=resolved_tuning.dp.disable_cpu_load_cost,
                node_worker_options=node_worker_options,
                gpu_worker_ids=gpu_worker_ids,
                cpu_worker_ids=cpu_worker_ids,
                http_latency_s=http_latency_s,
                enable_batch_shape_pruning=resolved_tuning.dp.enable_batch_shape_pruning,
                gpu_batch_slack=resolved_tuning.dp.gpu_batch_slack,
                enable_lower_bound_pruning=resolved_tuning.dp.enable_lower_bound_pruning,
                enable_worker_symmetry=resolved_tuning.dp.enable_worker_symmetry,
                lower_bound_cost_factor=resolved_tuning.dp.lower_bound_cost_factor,
                gpu_depth_cost_weight=resolved_tuning.dp.gpu_depth_cost_weight,
                debug_log=False,
                progress_enabled=self._dp_progress_enabled,
                progress_interval_s=self._dp_progress_interval_s,
                progress_logger=lambda msg: self.logger.info("%s", msg),
            )

            best_cost, raw_schedule, selected_plans = solver.solve(
                initial_worker_states
            )
            states_explored, memo_entries, memo_hits = self._solver_dp_stats(solver)
            self.logger.info(
                "Halo DP solved: best_cost=%.4f states=%d memo=%d memo_hits=%d",
                float(best_cost),
                states_explored,
                memo_entries,
                memo_hits,
            )
            worker_assignment = self._group_schedule_by_worker(
                raw_schedule, worker_ids, graph.node_order
            )
            schedule = Schedule(worker_assignment=worker_assignment)
            self._validate_schedule(schedule.worker_assignment, graph, worker_ids)
            self.logger.info(
                "Halo DP schedule generated: assignments=%d",
                len(raw_schedule),
            )
            return schedule
        except Exception as exc:
            raise RuntimeError("Halo optimizer failed to generate schedule") from exc

    def _build_workers(
        self,
        worker_names: Sequence[str],
        worker_profiles: Mapping[str, Mapping[str, Any]],
        graph: RuntimeGraph,
    ) -> dict[str, Worker]:
        requires_gpu = any(
            self._is_gpu_backend(op.backend) for op in graph.nodes.values()
        )

        workers: dict[str, Worker] = {}
        for worker_name in worker_names:
            worker_profile = worker_profiles.get(worker_name, {})
            has_gpu = bool(worker_profile.get("has_gpu"))
            worker_kind = "gpu" if has_gpu else "cpu"
            workers[worker_name] = Worker(
                id=worker_name,
                kind=worker_kind,
                device="cuda:0" if worker_kind == "gpu" else "cpu",
                capacity=1.0,
            )

        if requires_gpu and not any(w.kind == "gpu" for w in workers.values()):
            raise ValueError(
                "Halo DP requires at least one GPU worker because the graph contains"
                " GPU nodes."
            )

        return workers

    def _build_node_worker_options(
        self,
        graph: GraphSpec,
        workers: Mapping[str, Worker],
    ) -> dict[str, tuple[str, ...]]:
        gpu_workers = tuple(
            worker_id for worker_id, worker in workers.items() if worker.kind == "gpu"
        )
        cpu_workers = tuple(
            worker_id for worker_id, worker in workers.items() if worker.kind != "gpu"
        )

        options: dict[str, tuple[str, ...]] = {}
        for node_id, node in graph.nodes.items():
            if node.engine == "vllm":
                eligible = gpu_workers
            elif node.engine == "db" and node.type == "data_retrieval":
                eligible = cpu_workers
            else:
                raise ValueError(
                    "Unsupported node for Halo worker assignment: "
                    f"node='{node_id}' engine={node.engine!r} type={node.type!r}"
                )
            if not eligible:
                raise ValueError(
                    f"No compatible workers for node '{node_id}' (engine={node.engine},"
                    f" type={node.type})"
                )
            options[node_id] = tuple(sorted(eligible))
        return options

    def _build_data_profile_plan_choices(
        self,
        graph: RuntimeGraph,
        data_profile_results: Mapping[str, Sequence[DataProfileResultRow]],
    ) -> dict[tuple[str, str], tuple[QueryPlanChoice, ...]]:
        plan_choices: dict[tuple[str, str], tuple[QueryPlanChoice, ...]] = {}
        db_node_count = 0

        for node_id in graph.node_order:
            op = graph.nodes[node_id]
            if not self._is_data_profile_candidate(op):
                continue
            db_node_count += 1

            choices = self._extract_data_profile_choices_for_op(
                node_id, op, data_profile_results
            )
            if not choices:
                continue

            query_name = f"{node_id}_query"
            plan_choices[(node_id, query_name)] = choices
            self.logger.debug(
                "Halo data profile choices for node '%s': %d candidate plans",
                node_id,
                len(choices),
            )

        if db_node_count and not plan_choices:
            self.logger.warning(
                "Halo data profile plan lookup missed all DB nodes: db_nodes=%d"
                " result_keys=%d",
                db_node_count,
                len(data_profile_results),
            )

        return plan_choices

    def _extract_data_profile_choices_for_op(
        self,
        node_id: str,
        op: RuntimeOp,
        data_profile_results: Mapping[str, Sequence[DataProfileResultRow]],
    ) -> tuple[QueryPlanChoice, ...]:
        data_spec = op.data_spec
        spec_type = data_spec.get("type")
        if spec_type not in {"sql", "s3"}:
            return tuple()
        query_name = f"{node_id}_query"
        cache_key = data_profile_key_for_node_query(node_id, query_name)
        rows = data_profile_results.get(cache_key, ())
        if not rows:
            self.logger.debug(
                "Halo data profile lookup miss for node '%s' query '%s' (cache_key=%s,"
                " available_keys=%d)",
                node_id,
                query_name,
                cache_key,
                len(data_profile_results),
            )
        cost_estimates: list[DataProfileCostEstimate] = []
        for row in rows:
            cost_estimates.extend(row.cost_estimates)

        if not cost_estimates:
            return tuple()

        choices = self._build_data_profile_choices(node_id, cost_estimates)
        return choices

    def _build_data_profile_choices(
        self,
        node_id: str,
        estimates: Sequence[DataProfileCostEstimate],
    ) -> tuple[QueryPlanChoice, ...]:
        by_plan_id: dict[str, QueryPlanChoice] = {}
        for idx, estimate in enumerate(estimates):
            raw_cost = estimate.raw_cost
            plan_id = estimate.plan_id or f"{node_id}-data-profile-{idx}"
            description = estimate.description or "data profile"
            choice = QueryPlanChoice(
                plan_id=plan_id,
                description=description,
                cost=None,
                raw_cost=raw_cost,
                explain_json=estimate.explain_json,
                footprints=coerce_data_profile_footprints(estimate.footprints),
            )
            existing = by_plan_id.get(plan_id)
            if existing is None:
                by_plan_id[plan_id] = choice
                continue
            existing_score = self._plan_choice_score(existing)
            new_score = self._plan_choice_score(choice)
            if new_score < existing_score:
                by_plan_id[plan_id] = choice

        ordered = sorted(by_plan_id.values(), key=self._plan_choice_score)
        return tuple(ordered)

    @staticmethod
    def _plan_choice_score(choice: QueryPlanChoice) -> tuple[float, str]:
        return (
            choice.raw_cost if choice.raw_cost is not None else float("inf"),
            choice.plan_id,
        )

    def _extract_sql_table(self, template: str) -> str | None:
        match = self._sql_table_pattern.search(template)
        if not match:
            return None
        return normalize_table_name(match.group(1)) or None

    def _parse_data_profile_results(
        self,
        data_profile_results: (
            Mapping[str, Sequence[Mapping[str, Any] | DataProfileResultRow]] | None
        ),
    ) -> dict[str, tuple[DataProfileResultRow, ...]]:
        if not data_profile_results:
            return {}
        parsed: dict[str, tuple[DataProfileResultRow, ...]] = {}
        for key, rows in data_profile_results.items():
            parsed_rows: list[DataProfileResultRow] = []
            for idx, row in enumerate(rows):
                if isinstance(row, DataProfileResultRow):
                    parsed_rows.append(row)
                    continue
                try:
                    parsed_rows.append(DataProfileResultRow.model_validate(row))
                except Exception as exc:
                    raise RuntimeError(
                        f"Invalid data profile result row for key '{key}' at index"
                        f" {idx}"
                    ) from exc
            if parsed_rows:
                parsed[key] = tuple(parsed_rows)
        return parsed

    def _is_data_profile_candidate(self, op: RuntimeOp) -> bool:
        if self._map_engine(op.backend, op.task_type) != "db":
            return False
        return op.data_spec.get("type") in {"sql", "s3"}

    @staticmethod
    def _build_node_raw(op: RuntimeOp, deps: Sequence[str]) -> dict[str, Any]:
        raw = dict(op.data_spec) if isinstance(op.data_spec, dict) else {}
        if isinstance(op.inference_spec, dict) and op.inference_spec:
            raw["_inference_spec"] = dict(op.inference_spec)
        if deps and "parent" not in raw:
            raw["parent"] = deps[0]
        return raw

    @staticmethod
    def _safe_positive_float(value: object, default: float) -> float:
        if not isinstance(value, (int, float, str)):
            return default
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(parsed) or parsed <= 0:
            return default
        return parsed

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _resolve_and_apply_tuning(
        self,
        *,
        halo_graph: GraphSpec,
        workers: Mapping[str, Worker],
        plan_choices: Mapping[tuple[str, str], Sequence[QueryPlanChoice]],
    ) -> HaloResolvedTuning:
        resolved = self._resolve_halo_tuning(
            halo_graph=halo_graph,
            workers=workers,
            plan_choices=plan_choices,
        )
        self._apply_cost_model_params(resolved.cost_model)
        self.logger.info(
            "Halo tuning resolved: source=auto objective=latency cost=(init=%.4f"
            " llm_base=%.4f llm_input=%.4f db_input=%.4f) context=(nodes=%d"
            " gpu_nodes=%d db_nodes=%d gpu_workers=%d cpu_workers=%d"
            " input_query_count=%d model_size_b=%.2f switch_pressure=%.3f"
            " plan_variants=%.2f reuse=%.3f)",
            resolved.cost_model.model_init_sec_per_b,
            resolved.cost_model.llm_base_sec_per_b,
            resolved.cost_model.llm_input_sec,
            resolved.cost_model.db_input_sec,
            resolved.context.total_nodes,
            resolved.context.gpu_nodes,
            resolved.context.db_nodes,
            resolved.context.gpu_workers,
            resolved.context.cpu_workers,
            resolved.context.input_query_count,
            resolved.context.dominant_model_size_b,
            resolved.context.switch_pressure,
            resolved.context.plan_variants_avg,
            resolved.context.reuse_signal,
        )
        return resolved

    def _apply_cost_model_params(self, params: HaloCostModelParams) -> None:
        self._model_init_sec_per_b = params.model_init_sec_per_b
        self._llm_base_sec_per_b = params.llm_base_sec_per_b
        self._llm_input_sec = params.llm_input_sec
        self._db_input_sec = params.db_input_sec

    def _resolve_halo_tuning(
        self,
        *,
        halo_graph: GraphSpec,
        workers: Mapping[str, Worker],
        plan_choices: Mapping[tuple[str, str], Sequence[QueryPlanChoice]],
    ) -> HaloResolvedTuning:
        context = self._build_tuning_context(
            halo_graph=halo_graph,
            workers=workers,
            plan_choices=plan_choices,
        )
        cost_model = self._resolve_cost_model_params(context)
        dp = self._resolve_dp_tuning(context)
        return HaloResolvedTuning(context=context, cost_model=cost_model, dp=dp)

    def _build_tuning_context(
        self,
        *,
        halo_graph: GraphSpec,
        workers: Mapping[str, Worker],
        plan_choices: Mapping[tuple[str, str], Sequence[QueryPlanChoice]],
    ) -> HaloTuningContext:
        nodes = tuple(halo_graph.nodes.values())
        gpu_nodes = tuple(node for node in nodes if node.engine == "vllm")
        db_nodes = tuple(node for node in nodes if node.engine != "vllm")
        gpu_workers = sum(1 for worker in workers.values() if worker.kind == "gpu")
        cpu_workers = max(
            1, sum(1 for worker in workers.values() if worker.kind != "gpu")
        )
        model_names = {str(node.model).strip() for node in gpu_nodes if node.model}
        unique_models = len(model_names)
        switch_pressure_den = max(1, min(len(gpu_nodes), gpu_workers))
        switch_pressure = unique_models / switch_pressure_den
        model_sizes = [self._model_size_b(node) for node in gpu_nodes]
        dominant_model_size_b = (
            sum(model_sizes) / len(model_sizes)
            if model_sizes
            else self._default_model_size_b
        )
        plan_sizes = [len(choices) for choices in plan_choices.values()]
        plan_variants_min = min(plan_sizes) if plan_sizes else 0
        plan_variants_avg = sum(plan_sizes) / len(plan_sizes) if plan_sizes else 0.0
        plan_variants_max = max(plan_sizes) if plan_sizes else 0
        return HaloTuningContext(
            input_query_count=max(1, int(self._input_query_count)),
            total_nodes=len(nodes),
            gpu_nodes=len(gpu_nodes),
            db_nodes=len(db_nodes),
            gpu_workers=max(1, gpu_workers),
            cpu_workers=max(1, cpu_workers),
            unique_models=unique_models,
            switch_pressure=float(switch_pressure),
            dominant_model_size_b=float(dominant_model_size_b),
            db_query_count=len(plan_choices),
            plan_variants_min=plan_variants_min,
            plan_variants_avg=float(plan_variants_avg),
            plan_variants_max=plan_variants_max,
            reuse_signal=self._data_profile_reuse_signal(plan_choices),
        )

    def _resolve_cost_model_params(
        self, context: HaloTuningContext
    ) -> HaloCostModelParams:
        gpu_scale = math.sqrt(3.0 / max(1.0, float(context.gpu_workers)))
        model_scale = self._clamp(context.dominant_model_size_b / 20.0, 0.6, 2.0)
        query_scale = self._clamp(
            math.log2(float(context.input_query_count) + 1.0), 0.0, 6.0
        )
        query_extra = max(0.0, query_scale - 1.0)
        db_ratio = context.db_nodes / max(1.0, float(context.total_nodes))
        switch_extra = max(0.0, context.switch_pressure - 1.0)
        model_init_sec_per_b = self._clamp(
            2.5 * model_scale * (1.0 + 0.35 * switch_extra) * gpu_scale,
            1.5,
            5.0,
        )
        llm_base_sec_per_b = self._clamp(0.25 * model_scale * gpu_scale, 0.10, 0.80)
        llm_input_sec = self._clamp(0.15 * (1.0 + 0.08 * query_extra), 0.10, 0.35)
        db_input_sec = self._clamp(
            0.05 * (1.0 + 1.2 * db_ratio + 0.05 * query_extra),
            0.03,
            0.30,
        )
        return HaloCostModelParams(
            model_init_sec_per_b=model_init_sec_per_b,
            llm_base_sec_per_b=llm_base_sec_per_b,
            llm_input_sec=llm_input_sec,
            db_input_sec=db_input_sec,
        )

    def _resolve_dp_tuning(self, context: HaloTuningContext) -> HaloDPTuningParams:
        db_ratio = context.db_nodes / max(1.0, float(context.total_nodes))
        reuse_strong = context.reuse_signal >= 0.25 and context.plan_variants_avg > 1.3
        window_size = 2 if reuse_strong else 1
        if context.gpu_workers <= 1:
            gpu_cost_max_weight = 1.0
            gpu_cost_sum_weight = 0.0
        else:
            gpu_cost_max_weight = 0.93
            gpu_cost_sum_weight = 0.07
        switch_penalty_weight = self._clamp(
            0.08 + 0.20 * max(0.0, context.switch_pressure - 1.0), 0.05, 0.30
        )
        cpu_load_cost_weight = self._clamp(0.35 + 0.90 * db_ratio, 0.25, 1.25)
        cpu_load_early_weight = self._clamp(0.8 + 1.6 * db_ratio, 0.5, 2.5)
        gpu_depth_cost_weight = self._clamp(
            0.5 + 1.0 * max(0.0, db_ratio - 0.3), 0.5, 1.6
        )
        return HaloDPTuningParams(
            window_size=window_size,
            cpu_load_cost_weight=cpu_load_cost_weight,
            cpu_load_early_weight=cpu_load_early_weight,
            cpu_cost_mode="default",
            switch_penalty_weight=switch_penalty_weight,
            gpu_cost_max_weight=gpu_cost_max_weight,
            gpu_cost_sum_weight=gpu_cost_sum_weight,
            disable_epoch_batch_cost=False,
            disable_cpu_load_cost=False,
            enable_batch_shape_pruning=True,
            gpu_batch_slack=0 if context.total_nodes > 32 else 1,
            enable_lower_bound_pruning=True,
            enable_worker_symmetry=True,
            lower_bound_cost_factor=0.25 if context.total_nodes > 32 else 0.20,
            gpu_depth_cost_weight=gpu_depth_cost_weight,
        )

    @staticmethod
    def _data_profile_reuse_signal(
        plan_choices: Mapping[tuple[str, str], Sequence[QueryPlanChoice]],
    ) -> float:
        if not plan_choices:
            return 0.0
        scores: list[float] = []
        for choices in plan_choices.values():
            max_overlap = 0.0
            for idx in range(len(choices)):
                for jdx in range(idx + 1, len(choices)):
                    overlap = HaloOptimizer._footprint_overlap(
                        choices[idx].footprints,
                        choices[jdx].footprints,
                    )
                    if overlap > max_overlap:
                        max_overlap = overlap
            if max_overlap > 0:
                scores.append(max_overlap)
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    @staticmethod
    def _footprint_overlap(
        left: Mapping[str, int],
        right: Mapping[str, int],
    ) -> float:
        if not left or not right:
            return 0.0
        union = set(left) | set(right)
        if not union:
            return 0.0
        overlap = 0.0
        total = 0.0
        for key in union:
            left_weight = int(left.get(key, 0))
            right_weight = int(right.get(key, 0))
            overlap += min(left_weight, right_weight)
            total += max(left_weight, right_weight)
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, overlap / total))

    def _resolve_input_query_count(self, graph: RuntimeGraph) -> int:
        inferred_lengths: list[int] = []
        for op in graph.nodes.values():
            inferred_lengths.extend(self._collect_input_lengths(op.data_spec))
        inferred = max(inferred_lengths) if inferred_lengths else 0
        resolved = inferred if inferred > 0 else self._input_query_count_default
        self.logger.debug(
            "Halo input-query-count resolution: inferred=%s default=%d resolved=%d",
            inferred if inferred > 0 else "none",
            self._input_query_count_default,
            resolved,
        )
        return max(1, int(resolved))

    def _collect_input_lengths(self, value: object) -> list[int]:
        lengths: list[int] = []
        if isinstance(value, dict):
            value_type = value.get("type")
            items = value.get("items")
            if value_type == "list" and isinstance(items, list):
                lengths.append(len(items))
            for item in value.values():
                lengths.extend(self._collect_input_lengths(item))
            return lengths
        if isinstance(value, list):
            for item in value:
                lengths.extend(self._collect_input_lengths(item))
        return lengths

    def _assert_no_http_nodes(self, graph: RuntimeGraph) -> None:
        http_node_ids: list[str] = []
        for node_id in graph.node_order:
            op = graph.nodes[node_id]
            backend = str(op.backend).strip().lower()
            if backend == "http":
                http_node_ids.append(node_id)
        if not http_node_ids:
            return
        self.logger.error(
            "Halo does not support HTTP nodes. Found %d HTTP node(s): %s",
            len(http_node_ids),
            http_node_ids,
        )
        raise ValueError(
            "Halo optimizer does not support HTTP nodes by design. "
            f"Unsupported node_ids={http_node_ids}"
        )

    def _validate_supported_runtime_nodes(self, graph: RuntimeGraph) -> None:
        for node_id in graph.node_order:
            op = graph.nodes[node_id]
            try:
                engine = self._map_engine(op.backend, op.task_type)
            except ValueError as exc:
                raise ValueError(
                    "Halo optimizer encountered unsupported runtime node:"
                    f" node_id={node_id} task_type={op.task_type!r}"
                    f" backend={op.backend!r}"
                ) from exc
            if engine != "db":
                continue
            backend = str(op.backend).strip().lower()
            task_type = str(op.task_type).strip().lower()
            if backend != "data_retrieval" or task_type != "data_retrieval":
                raise ValueError(
                    "Halo optimizer only supports data_retrieval nodes on CPU. Found"
                    f" node_id={node_id} task_type={op.task_type!r}"
                    f" backend={op.backend!r}"
                )

    @staticmethod
    def _solver_dp_stats(solver: DPSolver) -> tuple[int, int, int]:
        return int(solver._solve_calls), len(solver._memo), int(solver._memo_hits)

    def _group_schedule_by_worker(
        self,
        schedule: Sequence[tuple[int, str, str]],
        worker_ids: Sequence[str],
        node_order: Sequence[str],
    ) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {worker_id: [] for worker_id in worker_ids}
        order_index = {node_id: idx for idx, node_id in enumerate(node_order)}
        sorted_items = sorted(
            schedule,
            key=lambda item: (
                item[0],
                order_index.get(item[2], 10**9),
                item[1],
            ),
        )
        for _, worker_id, node_id in sorted_items:
            if worker_id not in grouped:
                grouped[worker_id] = []
            grouped[worker_id].append(node_id)
        return grouped

    def _validate_schedule(
        self,
        schedule: dict[str, list[str]],
        graph: RuntimeGraph,
        worker_ids: Sequence[str],
    ) -> None:
        worker_set = set(worker_ids)
        schedule_workers = set(schedule.keys())
        if schedule_workers != worker_set:
            missing = sorted(worker_set - schedule_workers)
            extra = sorted(schedule_workers - worker_set)
            raise ValueError(
                f"Halo schedule worker mismatch. Missing={missing}, extra={extra}"
            )

        flattened: list[str] = []
        for worker_id in worker_ids:
            flattened.extend(schedule.get(worker_id, []))
        nodes = set(graph.nodes)
        if len(flattened) != len(set(flattened)):
            raise ValueError("Halo schedule has duplicate nodes")
        if set(flattened) != nodes:
            missing = sorted(nodes - set(flattened))
            extra = sorted(set(flattened) - nodes)
            raise ValueError(
                f"Halo schedule mismatch. Missing={missing}, extra={extra}"
            )

    @staticmethod
    def _is_gpu_backend(backend: str) -> bool:
        return str(backend).strip().lower() in {
            "vllm",
            "transformers",
            "diffusers",
            "omni",
        }

    def _map_engine(self, backend: str, task_type: str) -> str:
        normalized_backend = str(backend).strip().lower()
        normalized_task_type = str(task_type).strip().lower()
        if self._is_gpu_backend(normalized_backend):
            return "vllm"
        if normalized_backend == "http":
            return "http"
        if normalized_backend == "data_retrieval":
            return "db"
        if normalized_backend == "data_profiling":
            raise ValueError(
                "Halo optimizer does not support data_profiling nodes in optimized"
                f" graphs. backend={backend!r} task_type={task_type!r}"
            )
        raise ValueError(
            "Halo optimizer only supports backends in "
            "{'vllm','transformers','diffusers','omni','data_retrieval','http'}. "
            f"Got backend={backend!r} task_type={normalized_task_type!r}"
        )

    def _exec_cost(self, node: Node, _worker: Worker) -> float:
        if node.engine == "vllm":
            size_b = self._model_size_b(node)
            coeffs = MultimodalCostCoefficients(
                alpha_text=self._llm_base_sec_per_b,
                beta_text=self._llm_input_sec,
                alpha_embed=self._embed_base_sec_per_b,
                beta_embed=self._embed_input_sec,
                alpha_vlm=self._vlm_base_sec_per_b,
                beta_vlm=self._vlm_input_sec,
                alpha_diff=self._diff_base_sec_per_b,
                beta_diff=self._diff_prompt_input_sec,
            )
            return compute_gpu_exec_cost(
                node=node,
                model_size_b=size_b,
                input_query_count=self._input_query_count,
                coeffs=coeffs,
            )
        if node.engine == "http":
            return self._http_exec_cost(node)
        return self._db_input_sec * self._input_query_count

    def _http_exec_cost(self, node: Node) -> float:
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

    def _cache_multiplier(
        self,
        window: Sequence[QuerySignature],
        choice: QueryPlanChoice,
    ) -> float:
        base_fp = choice.footprints or {}
        if not base_fp or not window:
            return 1.0
        overlap = 0.0
        for sig in window:
            sig_fp = dict(sig.footprints)
            for name, weight in sig_fp.items():
                if name in base_fp:
                    overlap += min(weight, base_fp[name])
        if overlap <= 0:
            return 1.0
        return max(0.5, 1.0 - 0.01 * overlap)

    def _model_init_cost(self, node: Node, last_model: str | None) -> float:
        if node.engine != "vllm":
            return 0.0
        node_model = node.model or ""
        if last_model == node_model:
            return 0.0
        return self._model_init_sec_per_b * self._model_size_b(node)

    @staticmethod
    def _llm_cache_bonus(
        _node: Node,
        last_node: str | None,
        parents: Sequence[str],
    ) -> float:
        if not parents or not last_node:
            return 1.0
        if last_node in parents:
            return 0.9
        return 1.0

    def _epoch_penalty(self, epoch: int) -> float:
        return self._epoch_penalty_weight * (1.0 + 0.1 * max(0, epoch))

    def _model_size_b(self, node: Node) -> float:
        model_name = node.model
        if not model_name:
            raise ValueError(
                "Halo model size resolution requires a non-empty model name. "
                f"node_id={node.id!r}"
            )
        normalized = str(model_name).strip()
        if not normalized:
            raise ValueError(
                "Halo model size resolution requires a non-empty model name. "
                f"node_id={node.id!r}"
            )

        lookup_size = self._model_size_lookup.get(normalized.lower())
        if lookup_size is not None:
            return lookup_size

        suffix_match = self._model_suffix_pattern.search(normalized)
        if suffix_match:
            try:
                raw_size = float(suffix_match.group(1))
            except ValueError:
                raw_size = None
            if raw_size is not None:
                unit = suffix_match.group(2).lower()
                if unit == "m":
                    return raw_size / 1000.0
                return raw_size

        raise ValueError(
            "Halo model size is unknown; add it to MODEL_SIZE or provide an inferable "
            "suffix (e.g. 7B, 1.5B, 560M). "
            f"node_id={node.id!r} model={normalized!r}"
        )


__all__ = ["HaloOptimizer"]
