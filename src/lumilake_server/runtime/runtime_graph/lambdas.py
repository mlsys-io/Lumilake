from collections.abc import Sequence

from lumilake_server.data_specs import (
    PYTHON_TASK_TYPE,
    ColumnPlanEntry,
    PythonData,
    PythonMode,
)
from lumilake_server.ops import (
    DataOp,
    DataRetrievalOp,
    InputOp,
    LambdaMode,
    LambdaOp,
    LLMOp,
    ParseOp,
)
from lumilake_server.ops.join_ops import JoinOp
from lumilake_server.runtime import python_step
from lumilake_server.runtime.runtime_graph.base import _RuntimeGraphBuilderBase
from lumilake_server.runtime.runtime_graph.lowering import (
    LoweringContext,
    LoweringResult,
    Phase,
    after_phase,
    has_lowering,
    register,
)
from lumilake_server.runtime.runtime_graph.outputs import item_relative_path
from lumilake_server.runtime.runtime_ops import RuntimeOp


class _LambdaBuilder(_RuntimeGraphBuilderBase):
    def _build_python_step(self, op: LambdaOp, ctx: LoweringContext) -> list[RuntimeOp]:
        """Compile a standalone LambdaOp, and every LambdaOp it reads, into
        python tasks: an upstream first, then the one that reads it."""
        if op.id in ctx.nodes:
            return []
        ctx.visited.add(op.id)
        nodes: list[RuntimeOp] = []
        dependencies: list[str] = []
        plan: list[ColumnPlanEntry] = []
        row_counts: list[tuple[str, int | None]] = []
        inputs_dict = ctx.inputs_dict
        for inp in op.inputs:
            if isinstance(inp, LambdaOp):
                nodes.extend(self._build_python_step(inp, ctx))
                dep = inp.id
            elif isinstance(inp, JoinOp):
                # A join's runtime node is its op id; the join phase builds it.
                dep = inp.id
            elif isinstance(inp, (LLMOp, DataRetrievalOp)) or has_lowering(inp):
                runtime_ids = ctx.dsl_to_runtime.get(inp.id, [])
                if len(runtime_ids) != 1:
                    raise ValueError(
                        f"LambdaOp '{op.id}' reads '{inp.id}', which compiled to"
                        f" {len(runtime_ids)} runtime nodes; a standalone Python"
                        " step can read one upstream node per input"
                    )
                dep = runtime_ids[0]
            elif isinstance(inp, InputOp):
                ctx.visited.add(inp.id)
                plan.append(
                    ColumnPlanEntry(kind="literal", values=inputs_dict[inp.name])
                )
                row_counts.append((inp.id, len(inputs_dict[inp.name])))
                continue
            elif isinstance(inp, DataOp):
                ctx.visited.add(inp.id)
                plan.append(ColumnPlanEntry(kind="literal", values=list(inp.data)))
                row_counts.append((inp.id, len(inp.data)))
                continue
            else:
                raise ValueError(
                    f"LambdaOp '{op.id}' reads a {type(inp).__name__}; a standalone"
                    " Python step reads ops the graph lowers, Lambda, input or"
                    " data ops"
                )
            output = self._op_output(inp)
            row_counts.append((inp.id, output.row_count))
            plan.append(
                ColumnPlanEntry(
                    kind="stage",
                    node=dep,
                    # A rowwise LLM groups its per-row outputs back into one
                    # item; flatten it to one value per row for the python step.
                    flatten=True if output.needs_flatten else None,
                )
            )
            if dep not in dependencies:
                dependencies.append(dep)

        row_count = self._python_step_row_count(op.id, row_counts, combines_rows=True)
        nodes.append(self._python_step(op, plan, row_count, dependencies, ctx))
        return nodes

    def _tag_seed_entry(
        self,
        op: LambdaOp,
        plan: python_step.ColumnPlan,
        ctx: LoweringContext,
        dependencies: list[str],
    ) -> python_step.ColumnPlan:
        """Tag the plan entry a seeded Lambda reads its key column from."""
        seed = op.seed
        if seed is None:
            return plan
        seed_node = seed.node
        if seed_node not in ctx.graph_dict:
            raise ValueError(
                f"LambdaOp '{op.id}' seed references unknown node '{seed_node}'"
            )
        path = seed.path
        for index, entry in enumerate(plan):
            if entry.node != seed_node:
                continue
            seeded = entry.model_copy(update={"role": "seed"})
            if path and path != "items.output":
                seeded.path = item_relative_path(path)
            break
        else:
            raise ValueError(
                f"LambdaOp '{op.id}' seed node '{seed_node}' is not one of its inputs;"
                " add it there so the step reads its column"
            )
        if seed_node not in dependencies:
            dependencies.append(seed_node)
        return [*plan[:index], seeded, *plan[index + 1 :]]

    def _python_step(
        self,
        op: LambdaOp,
        plan: list[ColumnPlanEntry],
        row_count: int | None,
        dependencies: list[str],
        ctx: LoweringContext,
    ) -> RuntimeOp:
        """Build the python task for one LambdaOp from its resolved plan."""
        data_spec = PythonData(
            code=op.code,
            fn_name=op.fn.__name__,
            plan=self._tag_seed_entry(op, plan, ctx, dependencies),
            timeout_s=(
                python_step.DEFAULT_TIMEOUT_SECONDS
                if op.timeout_s is None
                else op.timeout_s
            ),
            mode=PythonMode(op.mode) if op.mode is not LambdaMode.ROW else None,
            by=op.by,
            on_fail=op.on_fail if op.mode == "reduce" else None,
            on_empty=op.on_empty if op.mode == "expand" else None,
            memory_mb=op.memory_mb,
            image=op.image,
            gpu=op.gpu,
            expected_runtime_s=op.expected_runtime_s,
            row_count=row_count if op.expected_runtime_s is not None else None,
            cache=op.cache,
        )
        return self._create_runtime_op(
            name=op.id,
            task_type=PYTHON_TASK_TYPE,
            data_spec=data_spec,
            model_spec={},
            inference_spec={},
            backend=python_step.BACKEND,
            model="",
            dependencies=dependencies if dependencies else None,
        )

    @register(LambdaOp, Phase.LIST_LAMBDA)
    def _lower_list_lambda(
        self, op: LambdaOp, ctx: LoweringContext
    ) -> LoweringResult | None:
        """Lower a list-mode LambdaOp; a row-mode one is lowered by its consumer
        or on its own."""
        if not op.takes_whole_columns:
            return None
        op_id = op.id
        inputs_dict = ctx.inputs_dict
        plan: list[ColumnPlanEntry] = []
        dependencies: list[str] = []
        row_counts: list[tuple[str, int | None]] = []
        for inp in op.inputs:
            if isinstance(inp, LambdaOp) and inp.takes_whole_columns:
                plan.append(ColumnPlanEntry(kind="stage", node=inp.id))
                dependencies.append(inp.id)
            elif isinstance(inp, LLMOp):
                plan.append(
                    ColumnPlanEntry(
                        kind="stage",
                        node=inp.id,
                        # A rowwise LLM groups its per-row outputs back into one
                        # item; the python step must flatten it to one value per
                        # row before the list-mode Lambda sees it.
                        flatten=True if self._op_output(inp).needs_flatten else None,
                    )
                )
                dependencies.append(inp.id)
            elif isinstance(inp, DataRetrievalOp) or has_lowering(inp):
                plan.append(ColumnPlanEntry(kind="stage", node=inp.id))
                dependencies.append(inp.id)
            elif isinstance(inp, InputOp):
                values = inputs_dict.get(inp.name)
                if values is None:
                    raise ValueError(
                        f"LambdaOp '{op_id}' input '{inp.id}' has no values supplied."
                    )
                plan.append(ColumnPlanEntry(kind="literal", values=list(values)))
                row_counts.append((inp.id, len(values)))
            elif isinstance(inp, DataOp):
                plan.append(ColumnPlanEntry(kind="literal", values=list(inp.data)))
                row_counts.append((inp.id, len(inp.data)))
            else:
                raise ValueError(
                    f"LambdaOp '{op_id}' list-mode input must be an op the graph"
                    " lowers, a list-mode LambdaOp, InputOp, or DataOp "
                    f"(got {type(inp).__name__})"
                )
            if not isinstance(inp, (InputOp, DataOp)):
                row_counts.append(
                    (
                        inp.id,
                        self._op_output(inp).row_count,
                    )
                )
        row_count = self._python_step_row_count(
            op_id, row_counts, combines_rows=op.mode is LambdaMode.ALIGNED
        )
        data_spec = PythonData(
            code=op.code,
            fn_name=op.fn.__name__,
            plan=plan,
            mode=PythonMode(op.mode),
            timeout_s=(
                python_step.DEFAULT_TIMEOUT_SECONDS
                if op.timeout_s is None
                else op.timeout_s
            ),
            memory_mb=op.memory_mb,
            image=op.image,
            gpu=op.gpu,
            expected_runtime_s=op.expected_runtime_s,
            row_count=row_count if op.expected_runtime_s is not None else None,
            cache=op.cache,
        )
        node = self._create_runtime_op(
            name=op_id,
            task_type=PYTHON_TASK_TYPE,
            data_spec=data_spec,
            model_spec={},
            inference_spec={},
            backend=python_step.BACKEND,
            model="",
            dependencies=dependencies if dependencies else None,
        )
        ctx.visited.update(input_op.id for input_op in op.inputs)
        return LoweringResult(nodes=[node])

    @register(LambdaOp, Phase.ROW_LAMBDA)
    def _lower_row_lambda(
        self, op: LambdaOp, ctx: LoweringContext
    ) -> LoweringResult | None:
        """Lower a row LambdaOp to its own step when an LLM did not inline it:
        one that is an output, that nothing reached, that asks for an image or
        GPUs, or that a runtime node references. The last one lowers checks that
        no runtime node reads an inlined Lambda."""
        if op.takes_whole_columns:
            return None
        referenced = self._referenced_row_lambdas(ctx)
        if (
            op.mode is LambdaMode.ROW
            and op.id in ctx.visited
            and op.id not in ctx.outputs_by_source
            and op.image is None
            and op.gpu is None
            and op.id not in referenced
        ):
            return None
        steps = self._build_python_step(op, ctx)
        return LoweringResult(
            nodes=steps,
            built={
                step.node_id: [step.node_id] for step in steps if step.node_id != op.id
            },
        )

    def _referenced_row_lambdas(self, ctx: LoweringContext) -> set[str]:
        """Row-mode LambdaOps referenced by any built runtime node, or by a join."""
        refs: set[str] = set()
        for node in ctx.nodes.values():
            for ref in _RuntimeGraphBuilderBase._iter_runtime_refs(node):
                op = ctx.graph_dict.get(ref)
                if isinstance(op, LambdaOp) and not op.takes_whole_columns:
                    refs.add(ref)
            for dep in node.dependencies:
                op = ctx.graph_dict.get(dep)
                if isinstance(op, LambdaOp) and not op.takes_whole_columns:
                    refs.add(dep)
        for op in ctx.graph_dict.values():
            if isinstance(op, JoinOp):
                refs.update(inp.id for inp in op.inputs)
        return refs

    @register(ParseOp, Phase.ROW_LAMBDA)
    def _lower_parse(self, op: ParseOp, ctx: LoweringContext) -> LoweringResult | None:
        """Lower a ParseOp to its own python step."""
        if op.id in ctx.nodes:
            return None
        plan, dependencies = self._parse_plan(op, ctx)
        row_count = self._python_step_row_count(
            op.id,
            [(op.inputs[0].id, self._op_output(op.inputs[0]).row_count)],
            combines_rows=False,
        )
        node = self._create_runtime_op(
            name=op.id,
            task_type=PYTHON_TASK_TYPE,
            data_spec=PythonData(
                mode=PythonMode.PARSE,
                parse=op.parse_spec,
                plan=plan,
                timeout_s=python_step.DEFAULT_TIMEOUT_SECONDS,
                row_count=row_count,
            ),
            model_spec={},
            inference_spec={},
            backend=python_step.BACKEND,
            model="",
            dependencies=dependencies if dependencies else None,
        )
        ctx.visited.add(op.inputs[0].id)
        return LoweringResult(nodes=[node])

    def _parse_plan(
        self, op: ParseOp, ctx: LoweringContext
    ) -> tuple[python_step.ColumnPlan, list[str]]:
        """The single-column plan a ParseOp reads: one value per input row."""
        source = op.inputs[0]
        if isinstance(source, InputOp):
            values = list(ctx.inputs_dict[source.name])
            return [ColumnPlanEntry(kind="literal", values=values)], []
        if isinstance(source, DataOp):
            return [ColumnPlanEntry(kind="literal", values=list(source.data))], []
        if isinstance(source, LambdaOp) and not source.takes_whole_columns:
            nodes = self._build_python_step(source, ctx)
            for node in nodes:
                ctx.nodes.setdefault(node.node_id, node)
            ctx.dsl_to_runtime.setdefault(source.id, [node.node_id for node in nodes])
        runtime_ids = ctx.dsl_to_runtime.get(source.id, [])
        if len(runtime_ids) != 1:
            raise ValueError(
                f"ParseOp '{op.id}' reads '{source.id}', which compiled to"
                f" {len(runtime_ids)} runtime nodes; a ParseOp reads one column"
            )
        entry = ColumnPlanEntry(kind="stage", node=runtime_ids[0])
        output = self._op_output(source)
        if output.needs_flatten:
            entry.flatten = True
        elif output.path:
            entry.path = item_relative_path(output.path)
        return [entry], [runtime_ids[0]]

    @after_phase(Phase.ROW_LAMBDA)
    def _check_inlined_lambda_refs(self, ctx: LoweringContext) -> None:
        """Reject runtime references to a row-mode LambdaOp that was inlined
        into its consuming LLM and therefore has no runtime python task."""
        materialized = set(ctx.nodes)
        inlined = {
            op_id
            for op_id, op in ctx.graph_dict.items()
            if isinstance(op, LambdaOp)
            and not op.takes_whole_columns
            and op_id not in materialized
        }
        if not inlined:
            return
        for node in ctx.nodes.values():
            for ref in _RuntimeGraphBuilderBase._iter_runtime_refs(node):
                if ref in inlined:
                    raise ValueError(
                        f"Runtime node '{node.node_id}' references LambdaOp"
                        f" '{ref}', which is inlined into the LLM that consumes"
                        " it and has no runtime output."
                    )

    def _python_step_row_count(
        self,
        op_id: str,
        row_counts: Sequence[tuple[str, int | None]],
        combines_rows: bool,
    ) -> int | None:
        """The rows a python step processes, or ``None`` when an input's count
        is unknown at build time. A step that combines its inputs row by row
        needs every multi-row input to carry the same count."""
        if combines_rows:
            multi = {
                count for _, count in row_counts if count is not None and count > 1
            }
            if len(multi) > 1:
                counts = ", ".join(f"{name}={count}" for name, count in row_counts)
                raise ValueError(
                    f"LambdaOp '{op_id}' combines inputs of different row counts"
                    f" ({counts}); every multi-row input must carry the same number"
                    " of rows"
                )
        known = [count for _, count in row_counts if count is not None]
        if len(known) != len(row_counts):
            return None
        return max([1, *known])
