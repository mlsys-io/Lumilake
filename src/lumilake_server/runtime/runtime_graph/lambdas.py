from collections.abc import Sequence
from typing import Any

from lumilake_server.ops import (
    DataOp,
    DataRetrievalOp,
    InputOp,
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
        plan: python_step.ColumnPlan = []
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
                plan.append({"kind": "literal", "values": inputs_dict[inp.name]})
                row_counts.append((inp.id, len(inputs_dict[inp.name])))
                continue
            elif isinstance(inp, DataOp):
                ctx.visited.add(inp.id)
                plan.append({"kind": "literal", "values": list(inp.data)})
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
            entry: dict[str, Any] = {"kind": "stage", "node": dep}
            if output.needs_flatten:
                # A rowwise LLM groups its per-row outputs back into one item;
                # flatten it to one value per row for the python step.
                entry["flatten"] = True
            plan.append(entry)
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
        """Tag the plan entry a seeded Lambda reads its key column from, so its
        step passes each row's seed as a second argument. The seed node is one
        of the op's declared inputs, so its entry is already in the plan."""
        if op.seed is None:
            return plan
        seed_node = op.seed["node"]
        if seed_node not in ctx.graph_dict:
            raise ValueError(
                f"LambdaOp '{op.id}' seed references unknown node '{seed_node}'"
            )
        path = op.seed["path"]
        for index, entry in enumerate(plan):
            if entry.get("node") != seed_node:
                continue
            seeded = {**entry, "role": "seed"}
            if path and path != "items.output":
                seeded["path"] = item_relative_path(path)
            break
        else:
            raise ValueError(
                f"LambdaOp '{op.id}' seed node '{seed_node}' is not one of its"
                " inputs; add it there so the step reads its column"
            )
        if seed_node not in dependencies:
            dependencies.append(seed_node)
        return [*plan[:index], seeded, *plan[index + 1 :]]

    def _python_step(
        self,
        op: LambdaOp,
        plan: python_step.ColumnPlan,
        row_count: int | None,
        dependencies: list[str],
        ctx: LoweringContext,
    ) -> RuntimeOp:
        """Build the python task for one LambdaOp from its resolved plan."""
        data_spec: dict[str, Any] = {
            "code": op.code,
            "fn_name": op.fn.__name__,
            "plan": self._tag_seed_entry(op, plan, ctx, dependencies),
            "timeout_s": (
                python_step.DEFAULT_TIMEOUT_SECONDS
                if op.timeout_s is None
                else op.timeout_s
            ),
        }
        if op.mode != "row":
            data_spec["mode"] = op.mode
        if op.by is not None:
            data_spec["by"] = op.by
        if op.mode == "reduce":
            data_spec["on_fail"] = op.on_fail
        if op.memory_mb is not None:
            data_spec["memory_mb"] = op.memory_mb
        if op.image is not None:
            data_spec["image"] = op.image
        if op.expected_runtime_s is not None:
            data_spec["expected_runtime_s"] = op.expected_runtime_s
            if row_count is not None:
                data_spec["row_count"] = row_count
        if op.cache is not None:
            data_spec["cache"] = op.cache
        return self._create_runtime_op(
            name=op.id,
            task_type=python_step.TASK_TYPE,
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
        plan: python_step.ColumnPlan = []
        dependencies: list[str] = []
        row_counts: list[tuple[str, int | None]] = []
        for inp in op.inputs:
            if isinstance(inp, LambdaOp) and inp.takes_whole_columns:
                plan.append({"kind": "stage", "node": inp.id})
                dependencies.append(inp.id)
            elif isinstance(inp, LLMOp):
                entry: dict[str, Any] = {"kind": "stage", "node": inp.id}
                if self._op_output(inp).needs_flatten:
                    # A rowwise LLM groups its per-row outputs back into one
                    # item; the python step must flatten it to one value per
                    # row before the list-mode Lambda sees it.
                    entry["flatten"] = True
                plan.append(entry)
                dependencies.append(inp.id)
            elif isinstance(inp, DataRetrievalOp) or has_lowering(inp):
                plan.append({"kind": "stage", "node": inp.id})
                dependencies.append(inp.id)
            elif isinstance(inp, InputOp):
                values = inputs_dict.get(inp.name)
                if values is None:
                    raise ValueError(
                        f"LambdaOp '{op_id}' input '{inp.id}' has no values supplied."
                    )
                plan.append({"kind": "literal", "values": list(values)})
                row_counts.append((inp.id, len(values)))
            elif isinstance(inp, DataOp):
                plan.append({"kind": "literal", "values": list(inp.data)})
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
        data_spec: dict[str, Any] = {
            "code": op.code,
            "fn_name": op.fn.__name__,
            "plan": plan,
            "mode": op.mode,
            "timeout_s": (
                python_step.DEFAULT_TIMEOUT_SECONDS
                if op.timeout_s is None
                else op.timeout_s
            ),
        }
        if op.memory_mb is not None:
            data_spec["memory_mb"] = op.memory_mb
        if op.image is not None:
            data_spec["image"] = op.image
        if op.expected_runtime_s is not None:
            data_spec["expected_runtime_s"] = op.expected_runtime_s
            if row_count is not None:
                data_spec["row_count"] = row_count
        if op.cache is not None:
            data_spec["cache"] = op.cache
        node = self._create_runtime_op(
            name=op_id,
            task_type=python_step.TASK_TYPE,
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
        that a runtime node references. An expand or reduce mode one reads one
        decoded row per call and always gets its own step. The last row-mode
        case lowers checks that no runtime node reads an inlined Lambda."""
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
        its data_spec/condition or as a runtime dependency, or by a join, which
        reads its columns and attach targets in the join phase."""
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
        """Lower a ParseOp to its own python step: one parsed record per row of
        its single input column."""
        if op.id in ctx.nodes:
            return None
        plan, dependencies = self._parse_plan(op, ctx)
        row_count = self._python_step_row_count(
            op.id,
            [(op.inputs[0].id, self._op_output(op.inputs[0]).row_count)],
            combines_rows=False,
        )
        data_spec: dict[str, Any] = {
            "mode": "parse",
            "parse": op.parse_spec,
            "plan": plan,
            "timeout_s": python_step.DEFAULT_TIMEOUT_SECONDS,
        }
        if row_count is not None:
            data_spec["row_count"] = row_count
        node = self._create_runtime_op(
            name=op.id,
            task_type=python_step.TASK_TYPE,
            data_spec=data_spec,
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
            return (
                [{"kind": "literal", "values": list(ctx.inputs_dict[source.name])}],
                [],
            )
        if isinstance(source, DataOp):
            return [{"kind": "literal", "values": list(source.data)}], []
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
        entry: dict[str, Any] = {"kind": "stage", "node": runtime_ids[0]}
        output = self._op_output(source)
        if output.needs_flatten:
            entry["flatten"] = True
        elif output.path:
            entry["path"] = item_relative_path(output.path)
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
