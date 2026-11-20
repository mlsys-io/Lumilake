"""Join lowering: one standalone python step merging keyed rows.

The first join input fixes the output rows. Each further input contributes the
fields of the row with the same key, and each attach value lands on the row it
was computed from. An attach target's row source is resolved here, when the
runtime graph is built, so the step pairs values with keys by position within
that one pair and never by user code.
"""

from typing import Any

from lumilake_server.ops import DataOp, InputOp, LambdaOp, Op
from lumilake_server.ops.embedding_ops import EmbeddingOp
from lumilake_server.ops.join_ops import JoinOp
from lumilake_server.ops.llm_ops import LLMChatOp, LLMCompletionOp, LLMVisionOp
from lumilake_server.ops.util_ops import FormatOp
from lumilake_server.runtime import join_step, python_step
from lumilake_server.runtime.runtime_graph.derived import _DerivedBuilder
from lumilake_server.runtime.runtime_graph.lowering import (
    LoweringContext,
    LoweringResult,
    Phase,
    register,
)
from lumilake_server.runtime.runtime_graph.outputs import (
    item_relative_path,
    output_row_count,
)


def row_source_candidates(op: Op, ctx: LoweringContext, consumer: str) -> list[Op]:
    """The ops ``op`` emits one output row per, narrowed to a single candidate
    where the op's shape names one."""
    graph = ctx.graph_dict
    if isinstance(op, LambdaOp):
        inputs = _non_literal(op)
        if op.takes_whole_columns:
            # A list-mode Lambda over several inputs is not 1:1 over one row
            # column, so it has no row source at all.
            return inputs if len(inputs) == 1 else []
        return _drop_broadcasts(inputs, ctx)
    if isinstance(op, FormatOp):
        return _drop_broadcasts(_non_literal(op), ctx)
    if isinstance(op, (LLMCompletionOp, EmbeddingOp)):
        return _drop_broadcasts([op.inputs[0]], ctx)
    if isinstance(op, (LLMChatOp, LLMVisionOp)):
        if op.rowwise_messages is not None:
            return op.inputs[1:2]
        if op.rowwise_columns:
            # Every column reads one node, or the op has no single row source.
            nodes = {
                column["node"]
                for column in op.rowwise_columns
                if isinstance(column.get("node"), str)
            }
            if len(nodes) > 1:
                names = ", ".join(f"'{node}'" for node in sorted(nodes))
                raise ValueError(
                    f"{consumer} reads columns from several nodes ({names}); a"
                    " row source is the single node all its columns read"
                )
            if not nodes:
                return _drop_broadcasts(_non_literal(op), ctx)
            (node,) = nodes
            upstream = graph.get(node)
            if upstream is None:
                raise ValueError(f"{consumer} reads unknown node '{node}'")
            return [upstream]
        raise ValueError(f"{consumer} takes no rowwise_template or messages column")
    raise ValueError(f"{consumer} is not an op that emits one row per input row")


def _non_literal(op: Op) -> list[Op]:
    return [inp for inp in op.inputs if not isinstance(inp, (InputOp, DataOp))]


def _drop_broadcasts(inputs: list[Op], ctx: LoweringContext) -> list[Op]:
    """Ignore a length-1 input: a broadcast source does not fix the row count."""
    if len(inputs) > 1:
        kept = [
            inp
            for inp in inputs
            if output_row_count(inp, ctx.inputs_dict, ctx.graph_dict) != 1
        ]
        if kept:
            return kept
    return inputs


def resolve_row_source(
    target: Op, allowed: set[str], ctx: LoweringContext
) -> tuple[Op, list[Op]]:
    """Follow an attach target's row sources to the node whose rows its values
    belong to, returning that node and the chain of hops above it."""
    consumer = f"attach target '{target.id}' ({type(target).__name__})"
    chain: list[Op] = []
    current = target
    while current.id not in allowed:
        if any(current is op for op in chain):
            raise ValueError(f"{consumer}: its row source passes back through itself")
        chain.append(current)
        source = current.row_source
        if source is None:
            candidates = row_source_candidates(current, ctx, consumer)
            if len(candidates) != 1:
                names = ", ".join(f"'{inp.id}'" for inp in candidates) or "none"
                raise ValueError(
                    f"{consumer}: op '{current.id}' is not 1:1 over one row"
                    f" column, so its row source is ambiguous ({names})"
                )
            source = candidates[0]
        current = source
    return current, chain


class _JoinBuilder(_DerivedBuilder):
    @register(JoinOp, Phase.JOIN)
    def _lower_join(self, op: JoinOp, ctx: LoweringContext) -> LoweringResult | None:
        """Lower a join to one python step reading its row columns and attach
        value/source pairs."""
        plan: list[dict[str, Any]] = []
        dependencies: list[str] = []
        columns: list[int] = []
        for column in op.columns:
            columns.append(self._planned(op, column, ctx, plan, dependencies))
        attach: list[list[Any]] = []
        for field, target in op.attach_ops.items():
            try:
                source, chain = resolve_row_source(target, set(op.column_ids), ctx)
            except ValueError as exc:
                raise ValueError(
                    f"JoinOp attach field {field!r} reads '{target.id}': {exc}"
                ) from exc
            for hop in chain:
                # Every hop on the chain feeds no other node, so the join is
                # what keeps it in the graph.
                ctx.visited.add(hop.id)
            attach.append(
                [
                    field,
                    self._planned(op, target, ctx, plan, dependencies),
                    self._planned(op, source, ctx, plan, dependencies),
                ]
            )
        node = self._create_runtime_op(
            name=op.id,
            task_type=python_step.TASK_TYPE,
            data_spec={
                "mode": join_step.JOIN_MODE,
                "plan": plan,
                "join": {
                    "node": op.id,
                    "on": op.on,
                    "inputs": columns,
                    "attach": attach,
                    "missing": op.missing,
                },
                "timeout_s": python_step.DEFAULT_TIMEOUT_SECONDS,
            },
            model_spec={},
            inference_spec={},
            backend=python_step.BACKEND,
            model="",
            dependencies=dependencies or None,
        )
        return LoweringResult(nodes=[node])

    def _planned(
        self,
        join: JoinOp,
        column: Op,
        ctx: LoweringContext,
        plan: list[dict[str, Any]],
        dependencies: list[str],
    ) -> int:
        """The plan index reading one column's values, one per row."""
        ctx.visited.add(column.id)
        runtime_ids = ctx.dsl_to_runtime.get(column.id)
        if not runtime_ids:
            raise ValueError(
                f"JoinOp '{join.id}' reads '{column.id}', which compiled to no"
                " runtime node before the join"
            )
        node = runtime_ids[-1]
        entry: dict[str, Any] = {"kind": "stage", "node": node}
        output = self._op_output(column)
        if output.needs_flatten:
            entry["flatten"] = True
        else:
            entry["path"] = item_relative_path(output.path)
        plan.append(entry)
        if node not in dependencies:
            dependencies.append(node)
        return len(plan) - 1
