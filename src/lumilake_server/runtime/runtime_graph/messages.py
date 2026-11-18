"""Lowering for a chat op whose conversation comes from a per-row messages
column."""

import json
from typing import Any

from lumilake_server.ops import DataOp, InputOp
from lumilake_server.ops.llm_ops import LLMChatOp, RowwiseMessages
from lumilake_server.runtime.runtime_graph.lowering import (
    LoweringContext,
    LoweringResult,
)
from lumilake_server.runtime.runtime_ops import RuntimeOp

_WHOLE_OUTPUT = "items.output"


def lower_rowwise_messages(
    builder: Any, llm_op: LLMChatOp, ctx: LoweringContext
) -> LoweringResult:
    """Lower a chat op that reads each row's whole conversation from its
    ``rowwise_messages`` column instead of a rendered template."""
    column = llm_op.rowwise_messages
    if column is None:
        raise ValueError(f"LLMChatOp '{llm_op.id}' has no rowwise_messages column")
    if llm_op.condition is not None:
        raise ValueError(
            f"LLMChatOp '{llm_op.id}' combines rowwise_messages with condition;"
            " the column supplies one conversation per row, so a condition"
            " would be ignored. Route on the condition's field instead, or"
            " drop the messages column."
        )
    # Trace the message chain: it validates the op's row shape and marks the
    # ops it names visited. The column, not the rendered template, is what the
    # model reads.
    builder._infer_structural_messages(
        llm_op.id,
        ctx.graph_dict,
        ctx.inputs_dict,
        ctx.visited,
        dsl_to_runtime=ctx.dsl_to_runtime,
        api_mode=llm_op.config.api is not None,
    )
    data_spec, dependencies = _messages_data_spec(builder, llm_op, column, ctx)
    node = _messages_op(builder, llm_op, data_spec, dependencies, ctx)
    return LoweringResult(nodes=[node])


def _messages_data_spec(
    builder: Any, llm_op: LLMChatOp, column: RowwiseMessages, ctx: LoweringContext
) -> tuple[dict[str, Any], list[str]]:
    """The ``list`` data spec one row per conversation, plus the runtime node
    that produces the column."""
    consumer = f"LLMChatOp '{llm_op.id}' rowwise_messages"
    upstream = ctx.graph_dict.get(column.node)
    if upstream is None:
        raise ValueError(
            f"{consumer} references {column.node!r}, which is not in the"
            " workflow graph"
        )
    ctx.visited.add(column.node)
    if isinstance(upstream, (InputOp, DataOp)):
        if column.path is not None:
            raise ValueError(
                f"{consumer} reads the literal {column.node!r} with drill path"
                f" {column.path!r}; literal values bind whole, so drop the path"
            )
        values = (
            ctx.inputs_dict.get(upstream.name)
            if isinstance(upstream, InputOp)
            else upstream.data
        )
        if not values:
            raise ValueError(
                f"{consumer} reads {column.node!r}, which has no values supplied"
            )
        return {"type": "list", "items": _decoded(values, consumer)}, []
    output = builder._op_output(upstream)
    if output.needs_flatten:
        raise ValueError(
            f"{consumer} reads '{column.node}', which groups its rows into one"
            " item; a messages column needs one serialized conversation per row"
        )
    runtime_ids = ctx.dsl_to_runtime.get(column.node)
    if not runtime_ids:
        raise ValueError(
            f"{consumer} reads '{column.node}', which compiles to no runtime"
            " node; a messages column reads an op the graph materializes, such"
            " as a list-mode LambdaOp"
        )
    path = output.read(column.path or _WHOLE_OUTPUT, consumer)
    return (
        {"type": "list", "node": runtime_ids[-1], "path": path},
        [runtime_ids[-1]],
    )


def _decoded(values: list[str], consumer: str) -> list[Any]:
    """Decode each literal value as the JSON message array it holds."""
    decoded: list[Any] = []
    for value in values:
        try:
            decoded.append(json.loads(value))
        except ValueError as exc:
            raise ValueError(
                f"{consumer} values must be JSON message arrays; got {value!r}"
            ) from exc
    return decoded


def _messages_op(
    builder: Any,
    llm_op: LLMChatOp,
    data_spec: dict[str, Any],
    dependencies: list[str],
    ctx: LoweringContext,
) -> RuntimeOp:
    """The local inference task, or the API task when the op targets an
    external endpoint."""
    skip_empty = bool(dependencies)
    if llm_op.config.api is not None:
        return builder._build_api_task(
            llm_op_id=llm_op.id,
            llm_op=llm_op,
            api_config=llm_op.config.api,
            data_spec=data_spec,
            dependencies=dependencies or None,
            output_spec=None,
            condition=None,
            graph_dict=ctx.graph_dict,
            inputs_dict=ctx.inputs_dict,
            dsl_to_runtime=ctx.dsl_to_runtime,
            skip_when_all_inputs_empty=skip_empty,
        )[0]
    inference_spec = llm_op.config.inference_spec()
    if llm_op.structural_outputs:
        inference_spec["templates"] = llm_op.structural_outputs
    return builder._create_runtime_op(
        name=llm_op.id,
        task_type="inference",
        data_spec=data_spec,
        model_spec=builder._build_model_spec(llm_op.config),
        inference_spec=inference_spec,
        backend="vllm",
        model=llm_op.config.model,
        dependencies=dependencies or None,
        output_enforcement=llm_op.config.output_enforcement_spec(),
        skip_when_all_inputs_empty=skip_empty,
    )
