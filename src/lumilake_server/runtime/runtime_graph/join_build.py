"""Building one join's python step.

An attach target's row source is resolved here, so the step pairs values with keys by
position. Consumers in an earlier phase read the join by its op id, which is its runtime
node id, so only the join phase builds the step.
"""

from lumilake_server.data_specs import (
    PYTHON_TASK_TYPE,
    ColumnPlanEntry,
    JoinDescription,
    NodeColumn,
    PythonData,
)
from lumilake_server.ops import DataOp, InputOp, LambdaOp, Op
from lumilake_server.ops.embedding_ops import EmbeddingOp
from lumilake_server.ops.join_ops import JoinOp
from lumilake_server.ops.llm_ops import LLMChatOp, LLMCompletionOp, LLMVisionOp
from lumilake_server.ops.util_ops import FormatOp
from lumilake_server.runtime import python_step
from lumilake_server.runtime.runtime_graph.base import _RuntimeGraphBuilderBase
from lumilake_server.runtime.runtime_graph.lowering import LoweringContext
from lumilake_server.runtime.runtime_graph.outputs import (
    OutputKind,
    item_relative_path,
    output_row_count,
)
from lumilake_server.runtime.runtime_ops import RuntimeOp


def row_source_candidates(op: Op, ctx: LoweringContext, consumer: str) -> list[Op]:
    """The inputs ``op`` emits one output row per, one candidate where one is named."""
    graph = ctx.graph_dict
    if isinstance(op, LambdaOp):
        if op.mode not in ("row", "aligned"):
            # A whole-result mode may filter, reorder or expand rows.
            return []
        return _drop_broadcasts(_non_literal(op), ctx)
    if isinstance(op, FormatOp):
        return _drop_broadcasts(_non_literal(op), ctx)
    if isinstance(op, (LLMCompletionOp, EmbeddingOp)):
        return _drop_broadcasts([op.inputs[0]], ctx)
    if isinstance(op, (LLMChatOp, LLMVisionOp)):
        if op.rowwise_messages is not None:
            return op.inputs[1:2]
        if op.rowwise_columns:
            nodes = {
                column.node
                for column in op.rowwise_columns
                if isinstance(column, NodeColumn)
            }
            if len(nodes) > 1:
                names = ", ".join(f"'{node}'" for node in sorted(nodes))
                raise ValueError(
                    f"{consumer} reads columns from several nodes ({names}); a row"
                    " source is the single node all its columns read"
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
    """Ignore a length-1 input: a broadcast does not fix the row count."""
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
    target: Op, allowed: set[str], ctx: LoweringContext, consumer: str
) -> Op:
    """Follow an attach target's row sources to the node holding its rows."""
    current = target
    while current.id not in allowed:
        source = current.row_source
        if source is None:
            candidates = row_source_candidates(current, ctx, consumer)
            if len(candidates) != 1:
                names = ", ".join(f"'{inp.id}'" for inp in candidates) or "none"
                raise ValueError(
                    f"{consumer}: op '{current.id}' is not 1:1 over one row column, so"
                    f" its row source is ambiguous ({names})"
                )
            source = candidates[0]
        current = source
    return current


def build_join_step(
    builder: _RuntimeGraphBuilderBase, op: JoinOp, ctx: LoweringContext
) -> RuntimeOp:
    """Build a join's python step."""
    ctx.visited.add(op.id)
    plan: list[ColumnPlanEntry] = []
    dependencies: list[str] = []
    columns: list[int] = []
    for column in op.columns:
        columns.append(_planned(builder, op, column, ctx, plan, dependencies))
    attach: list[tuple[str, int, int]] = []
    for field, target in op.attach_ops.items():
        consumer = f"JoinOp '{op.id}' attach field {field!r} reading '{target.id}'"
        if isinstance(target, FormatOp):
            raise ValueError(
                f"{consumer}, but a format op is a row source only as a hop on a chain;"
                " it has no runtime node to read directly"
            )
        source = resolve_row_source(target, set(op.column_ids), ctx, consumer)
        attach.append(
            (
                field,
                _planned(builder, op, target, ctx, plan, dependencies),
                _planned(builder, op, source, ctx, plan, dependencies),
            )
        )
    node = builder._create_runtime_op(
        name=op.id,
        task_type=PYTHON_TASK_TYPE,
        data_spec=PythonData(
            mode="join",
            plan=plan,
            join=JoinDescription(
                node=op.id,
                by=op.by,
                inputs=columns,
                attach=attach,
                missing=op.missing,
            ),
            timeout_s=python_step.DEFAULT_TIMEOUT_SECONDS,
        ),
        model_spec={},
        inference_spec={},
        backend=python_step.BACKEND,
        model="",
        dependencies=dependencies or None,
    )
    return node


def _planned(
    builder: _RuntimeGraphBuilderBase,
    join: JoinOp,
    column: Op,
    ctx: LoweringContext,
    plan: list[ColumnPlanEntry],
    dependencies: list[str],
) -> int:
    """The plan index reading one column's values, one per row."""
    ctx.visited.add(column.id)
    runtime_ids = ctx.dsl_to_runtime.get(column.id)
    if not runtime_ids:
        raise ValueError(
            f"JoinOp '{join.id}' reads '{column.id}', which compiled to no runtime node"
            " before the join"
        )
    node = runtime_ids[-1]
    entry = ColumnPlanEntry(kind="stage", node=node)
    output = builder._op_output(column)
    if output.needs_flatten:
        entry.flatten = True
    elif output.kind is OutputKind.ROW_LAMBDA:
        # A row step writes the typed value under ``value``.
        entry.path = "value"
    else:
        entry.path = item_relative_path(output.path)
    plan.append(entry)
    if node not in dependencies:
        dependencies.append(node)
    return len(plan) - 1
