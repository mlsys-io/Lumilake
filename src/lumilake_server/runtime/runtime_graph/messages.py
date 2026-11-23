"""Lowering for a chat op whose conversation comes from a per-row messages
column."""

import json
from typing import Any

from lumilake_server.data_specs import ListData
from lumilake_server.ops import DataOp, DataRetrievalOp, InputOp, LambdaOp, Op
from lumilake_server.ops.join_ops import JoinOp
from lumilake_server.ops.llm_ops import LLMChatOp, RowwiseMessages
from lumilake_server.runtime.runtime_graph.derived import _DerivedBuilder
from lumilake_server.runtime.runtime_graph.lowering import (
    LoweringContext,
    LoweringResult,
    Phase,
    register,
)
from lumilake_server.runtime.runtime_graph.outputs import _WHOLE
from lumilake_server.runtime.runtime_ops import RuntimeOp


class _MessagesBuilder(_DerivedBuilder):
    """Builds the task a chat op runs when its conversations come from a
    per-row messages column."""

    @register(LLMChatOp, Phase.LLM)
    def _lower_chat(self, op: LLMChatOp, ctx: LoweringContext) -> LoweringResult | None:
        """The list-fed task reading each row's whole conversation from its
        ``rowwise_messages`` column, or the rendered-prompt lowering for a
        chat op without one."""
        column = op.rowwise_messages
        if column is None:
            return self._lower_llm(op, ctx)
        # The column, not the empty message op, is what the model reads; that
        # op is marked visited for the graph traversal check.
        ctx.visited.add(op.messages.id)
        data_spec, dependencies = self._messages_data_spec(op, column, ctx)
        node = self._messages_op(op, data_spec, dependencies, ctx)
        return LoweringResult(nodes=[node])

    def _messages_data_spec(
        self, llm_op: LLMChatOp, column: RowwiseMessages, ctx: LoweringContext
    ) -> tuple[ListData, list[str]]:
        """The ``list`` data spec one row per conversation, plus the runtime
        node that produces the column."""
        consumer = f"LLMChatOp '{llm_op.id}' rowwise_messages"
        upstream = llm_op.rowwise_messages_op
        assert upstream is not None
        node = upstream.id
        ctx.visited.add(node)
        if isinstance(upstream, (InputOp, DataOp)):
            if column.path is not None:
                raise ValueError(
                    f"{consumer} reads the literal {node!r} with drill"
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
                    f"{consumer} reads {node!r}, which has no values supplied"
                )
            return ListData(items=_decoded(values, consumer)), []
        output = self._op_output(upstream)
        if output.needs_flatten:
            raise ValueError(
                f"{consumer} reads '{node}', which groups its rows into"
                " one item; a messages column needs one conversation per row"
            )
        if not _producer_kind_supported(upstream):
            raise ValueError(
                f"{consumer} reads '{node}' ({type(upstream).__name__}); a messages"
                " column reads a list-mode LambdaOp, a join, a data retrieval or a"
                " literal input"
            )
        runtime_ids = ctx.dsl_to_runtime.get(node)
        if isinstance(upstream, JoinOp):
            # A join's runtime node is its op id; the join phase builds it.
            runtime_ids = [node]
        if not runtime_ids:
            raise ValueError(
                f"{consumer} reads '{node}', which compiles to no runtime node"
            )
        path = output.read(column.path or _WHOLE, consumer)
        return (
            ListData(node=runtime_ids[-1], path=path),
            [runtime_ids[-1]],
        )

    def _messages_op(
        self,
        llm_op: LLMChatOp,
        data_spec: ListData,
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
            source_op=llm_op,
        )


def _producer_kind_supported(upstream: Op) -> bool:
    """Whether a chat op reads the op by node id: a whole-list producer or retrieval."""
    if isinstance(upstream, (DataRetrievalOp, JoinOp)):
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
