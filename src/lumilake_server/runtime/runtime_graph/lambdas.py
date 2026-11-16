from collections.abc import Sequence
from typing import Any

from lumilake_server.ops import (
    DataOp,
    DataRetrievalOp,
    InputOp,
    LambdaOp,
    LLMOp,
    Op,
)
from lumilake_server.ops.llm_ops import LLMChatOp
from lumilake_server.runtime import python_step
from lumilake_server.runtime.runtime_graph.base import _RuntimeGraphBuilderBase
from lumilake_server.runtime.runtime_ops import RuntimeOp


class _LambdaBuilder(_RuntimeGraphBuilderBase):
    def _build_python_step(
        self,
        op_id: str,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        visited_node_ids: set[str],
        nodes: dict[str, RuntimeOp],
        node_order: list[str],
        dsl_to_runtime: dict[str, list[str]],
    ) -> str:
        """Compile one standalone LambdaOp (and any LambdaOp it reads) into a
        FlowMesh python task. Returns its runtime node id."""
        if op_id in nodes:
            return op_id
        op = graph_dict[op_id]
        assert isinstance(op, LambdaOp)
        visited_node_ids.add(op_id)

        plan: python_step.ColumnPlan = []
        dependencies: list[str] = []
        row_counts: list[tuple[str, int | None]] = []
        for inp in op.inputs:
            if isinstance(inp, LambdaOp):
                dep = self._build_python_step(
                    inp.id,
                    graph_dict,
                    inputs_dict,
                    visited_node_ids,
                    nodes,
                    node_order,
                    dsl_to_runtime,
                )
            elif isinstance(inp, (LLMOp, DataRetrievalOp)):
                runtime_ids = dsl_to_runtime.get(inp.id, [])
                if len(runtime_ids) != 1:
                    raise ValueError(
                        f"LambdaOp '{op_id}' reads '{inp.id}', which compiled to"
                        f" {len(runtime_ids)} runtime nodes; a standalone Python"
                        " step can read one upstream node per input"
                    )
                dep = runtime_ids[0]
            elif isinstance(inp, InputOp):
                visited_node_ids.add(inp.id)
                plan.append({"kind": "literal", "values": inputs_dict[inp.name]})
                row_counts.append((inp.id, len(inputs_dict[inp.name])))
                continue
            elif isinstance(inp, DataOp):
                visited_node_ids.add(inp.id)
                plan.append({"kind": "literal", "values": list(inp.data)})
                row_counts.append((inp.id, len(inp.data)))
                continue
            else:
                raise ValueError(
                    f"LambdaOp '{op_id}' reads a {type(inp).__name__}; a standalone"
                    " Python step reads LLM, data-retrieval, Lambda, input or data"
                    " ops"
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

        row_count = self._python_step_row_count(op_id, row_counts, combines_rows=True)

        data_spec: dict[str, Any] = {
            "code": op.code,
            "fn_name": op.fn.__name__,
            "plan": plan,
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
        runtime_op = self._create_runtime_op(
            name=op_id,
            task_type=python_step.TASK_TYPE,
            data_spec=data_spec,
            model_spec={},
            inference_spec={},
            backend=python_step.BACKEND,
            model="",
            dependencies=dependencies,
        )
        nodes[op_id] = runtime_op
        node_order.append(op_id)
        dsl_to_runtime[op_id] = [op_id]
        return op_id

    def _build_node_from_list_lambda_op(
        self,
        op_id: str,
        op: LambdaOp,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        dsl_to_runtime: dict[str, list[str]],
    ) -> RuntimeOp:
        """Compile a list-mode LambdaOp to one FlowMesh ``python`` task."""
        plan: python_step.ColumnPlan = []
        dependencies: list[str] = []
        row_counts: list[tuple[str, int | None]] = []
        for inp in op.inputs:
            if isinstance(inp, LambdaOp) and inp.takes_whole_columns:
                plan.append({"kind": "stage", "node": inp.id})
                dependencies.append(inp.id)
            elif isinstance(inp, LLMChatOp) and inp.row_gate is not None:
                # A gated op has no runtime node of its own: its merge stage is
                # its output, one item per input row holding the served answer
                # under ``output`` and ``None`` for a row the gate skipped.
                merge_id = f"{inp.id}__rows"
                if merge_id not in dsl_to_runtime.get(inp.id, []):
                    raise ValueError(
                        f"LambdaOp '{op_id}' reads the row-gated LLMChatOp"
                        f" '{inp.id}' before its stages are built"
                    )
                plan.append(self._python_plan_entry(merge_id, path="value.output"))
                dependencies.append(merge_id)
            elif isinstance(inp, LLMOp):
                entry: dict[str, Any] = {"kind": "stage", "node": inp.id}
                if self._op_output(inp).needs_flatten:
                    # A rowwise LLM groups its per-row outputs back into one
                    # item; the python step must flatten it to one value per
                    # row before the list-mode Lambda sees it.
                    entry["flatten"] = True
                plan.append(entry)
                dependencies.append(inp.id)
            elif isinstance(inp, DataRetrievalOp):
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
                    f"LambdaOp '{op_id}' list-mode input must be an LLMOp, "
                    f"DataRetrievalOp, list-mode LambdaOp, InputOp, or DataOp "
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
        return self._create_runtime_op(
            name=op_id,
            task_type=python_step.TASK_TYPE,
            data_spec=data_spec,
            model_spec={},
            inference_spec={},
            backend=python_step.BACKEND,
            model="",
            dependencies=dependencies if dependencies else None,
        )

    def _referenced_row_lambdas(
        self,
        nodes: dict[str, RuntimeOp],
        graph_dict: dict[str, Op],
    ) -> set[str]:
        """Row-mode LambdaOps referenced by any built runtime node, either in
        its data_spec/condition or as a runtime dependency."""
        refs: set[str] = set()
        for node in nodes.values():
            for ref in _RuntimeGraphBuilderBase._iter_runtime_refs(node):
                op = graph_dict.get(ref)
                if isinstance(op, LambdaOp) and not op.takes_whole_columns:
                    refs.add(ref)
            for dep in node.dependencies:
                op = graph_dict.get(dep)
                if isinstance(op, LambdaOp) and not op.takes_whole_columns:
                    refs.add(dep)
        return refs

    def _check_inlined_lambda_refs(
        self,
        nodes: dict[str, RuntimeOp],
        graph_dict: dict[str, Op],
    ) -> None:
        """Reject runtime references to a row-mode LambdaOp that was inlined
        into its consuming LLM and therefore has no runtime python task."""
        materialized = set(nodes.keys())
        inlined = {
            op_id
            for op_id, op in graph_dict.items()
            if isinstance(op, LambdaOp)
            and not op.takes_whole_columns
            and op_id not in materialized
        }
        if not inlined:
            return
        for node in nodes.values():
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
