"""Lowering for a chat op whose conversation comes from a per-row messages
column."""

import json
from typing import Any

from lumilake_server.ops import DataOp, DataRetrievalOp, InputOp, LambdaOp, Op
from lumilake_server.ops.llm_ops import LLMChatOp, RowwiseMessages
from lumilake_server.runtime.runtime_graph.derived import _DerivedBuilder
from lumilake_server.runtime.runtime_graph.lowering import LoweringContext
from lumilake_server.runtime.runtime_graph.outputs import _WHOLE
from lumilake_server.runtime.runtime_ops import RuntimeOp


class _MessagesBuilder(_DerivedBuilder):
    """Builds the task a chat op runs when its conversations come from a
    per-row messages column."""

    def lower_rowwise_messages(
        self, llm_op: LLMChatOp, ctx: LoweringContext
    ) -> RuntimeOp:
        """The local inference task, or the API task when the op targets an
        external endpoint, reading each row's whole conversation from its
        ``rowwise_messages`` column instead of a rendered template."""
        column = llm_op.rowwise_messages
        if column is None:
            raise ValueError(f"LLMChatOp '{llm_op.id}' has no rowwise_messages column")
        # Trace the message chain: it validates the op's row shape and marks the
        # ops it names visited. The column, not the rendered message, is what
        # the model reads.
        self._infer_structural_messages(
            llm_op.id,
            ctx.graph_dict,
            ctx.inputs_dict,
            ctx.visited,
            dsl_to_runtime=ctx.dsl_to_runtime,
            api_mode=llm_op.config.api is not None,
        )
        data_spec, dependencies = self._messages_data_spec(llm_op, column, ctx)
        return self._messages_op(llm_op, data_spec, dependencies, ctx)

    def _messages_data_spec(
        self, llm_op: LLMChatOp, column: RowwiseMessages, ctx: LoweringContext
    ) -> tuple[dict[str, Any], list[str]]:
        """The ``list`` data spec one row per conversation, plus the runtime
        node that produces the column."""
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
                    f"{consumer} reads the literal {column.node!r} with drill"
                    f" path {column.path!r}; literal values bind whole, so drop"
                    " the path"
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
        output = self._op_output(upstream)
        if output.needs_flatten:
            raise ValueError(
                f"{consumer} reads '{column.node}', which groups its rows into"
                " one item; a messages column needs one conversation per row"
            )
        if not _producer_kind_supported(upstream):
            raise ValueError(
                f"{consumer} reads '{column.node}' ({type(upstream).__name__});"
                " a messages column reads a list-mode LambdaOp, a data"
                " retrieval or a literal input"
            )
        runtime_ids = ctx.dsl_to_runtime.get(column.node)
        if not runtime_ids:
            raise ValueError(
                f"{consumer} reads '{column.node}', which compiles to no runtime"
                " node even though the build order should have materialized it"
            )
        path = output.read(column.path or _WHOLE, consumer)
        return (
            {"type": "list", "node": runtime_ids[-1], "path": path},
            [runtime_ids[-1]],
        )

    def _messages_op(
        self,
        llm_op: LLMChatOp,
        data_spec: dict[str, Any],
        dependencies: list[str],
        ctx: LoweringContext,
    ) -> RuntimeOp:
        """The local inference task, or the API task when the op targets an
        external endpoint."""
        if llm_op.config.api is not None:
            return self._build_api_task(
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
                skip_when_all_inputs_empty=bool(dependencies),
            )[0]
        inference_spec = llm_op.config.inference_spec()
        if llm_op.structural_outputs:
            inference_spec["templates"] = llm_op.structural_outputs
        return self._create_runtime_op(
            name=llm_op.id,
            task_type="inference",
            data_spec=data_spec,
            model_spec=self._build_model_spec(llm_op.config),
            inference_spec=inference_spec,
            backend="vllm",
            model=llm_op.config.model,
            dependencies=dependencies or None,
            output_enforcement=llm_op.config.output_enforcement_spec(),
            skip_when_all_inputs_empty=bool(dependencies),
        )


def _producer_kind_supported(upstream: Op) -> bool:
    """Whether the op is one the build materializes before the chat op lowers:
    a list-mode Lambda or a data retrieval."""
    if isinstance(upstream, DataRetrievalOp):
        return True
    return isinstance(upstream, LambdaOp) and upstream.takes_whole_columns


def _decoded(values: list[str], consumer: str) -> list[Any]:
    """Decode each literal value as the message list it holds. Only a literal
    carries JSON text; a runtime producer emits decoded lists as they are."""
    decoded: list[Any] = []
    for value in values:
        try:
            messages = json.loads(value)
        except ValueError as exc:
            raise ValueError(
                f"{consumer} values must be JSON message arrays; got {value!r}"
            ) from exc
        if not isinstance(messages, list) or not messages:
            raise ValueError(
                f"{consumer} values must be non-empty JSON message arrays;"
                f" got {value!r}"
            )
        decoded.append(messages)
    return decoded
