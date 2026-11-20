from collections.abc import Sequence

from lumilake_server.data_specs import PYTHON_TASK_TYPE, ColumnPlanEntry, PythonData
from lumilake_server.ops import (
    DataOp,
    DataRetrievalOp,
    InputOp,
    LambdaOp,
    LLMOp,
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
        nodes.append(self._python_step(op, plan, row_count, dependencies))
        return nodes

    def _python_step(
        self,
        op: LambdaOp,
        plan: list[ColumnPlanEntry],
        row_count: int | None,
        dependencies: list[str],
    ) -> RuntimeOp:
        """Build the python task for one LambdaOp from its resolved plan."""
        data_spec = PythonData(
            code=op.code,
            fn_name=op.fn.__name__,
            plan=plan,
            timeout_s=(
                python_step.DEFAULT_TIMEOUT_SECONDS
                if op.timeout_s is None
                else op.timeout_s
            ),
            mode=op.mode if op.mode != "row" else None,
            by=op.by,
            memory_mb=op.memory_mb,
            image=op.image,
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
            elif isinstance(inp, JoinOp):
                # A join's runtime node is its op id; the join phase builds it.
                plan.append({"kind": "stage", "node": inp.id})
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
            op_id, row_counts, combines_rows=op.mode == "aligned"
        )
        data_spec = PythonData(
            code=op.code,
            fn_name=op.fn.__name__,
            plan=plan,
            mode=op.mode,
            timeout_s=(
                python_step.DEFAULT_TIMEOUT_SECONDS
                if op.timeout_s is None
                else op.timeout_s
            ),
            memory_mb=op.memory_mb,
            image=op.image,
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
        """Lower a row-mode LambdaOp to its own python step when no LLM inlined
        it: one that is an output, that nothing reached, that has an image, or
        that a runtime node references. An expand or reduce mode one always
        gets its own step."""
        if op.takes_whole_columns:
            return None
        referenced = self._referenced_row_lambdas(ctx)
        if (
            op.mode == "row"
            and op.id in ctx.visited
            and op.id not in ctx.outputs_by_source
            and op.image is None
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
        """Row-mode LambdaOps referenced by any built runtime node, either in
        its data_spec/condition or as a runtime dependency."""
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
        return refs

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
