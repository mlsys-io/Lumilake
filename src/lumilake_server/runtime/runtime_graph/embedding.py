from collections.abc import Sequence
from typing import Any

from lumilake_server.ops import DataOp, InputOp, Op
from lumilake_server.ops.embedding_ops import EmbeddingOp
from lumilake_server.runtime.runtime_graph.base import _default_output_destination
from lumilake_server.runtime.runtime_graph.lambdas import _LambdaBuilder
from lumilake_server.runtime.runtime_ops import RuntimeOp


class _EmbeddingBuilder(_LambdaBuilder):
    def _validate_embedding_texts(self, op_id: str, items: Sequence[Any]) -> None:
        for idx, text in enumerate(items):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"EmbeddingOp '{op_id}' text input[{idx}] must be a non-empty "
                    f"string (got {text!r})"
                )

    def _build_node_from_embedding_op(
        self,
        llm_op_id: str,
        llm_op: EmbeddingOp,
        inputs_dict: dict[str, list[str]],
        visited_node_ids: set[str],
        graph_dict: dict[str, Op] | None = None,
        dsl_to_runtime: dict[str, list[str]] | None = None,
    ) -> RuntimeOp:
        visited_node_ids.add(llm_op_id)
        content_op = llm_op.content
        visited_node_ids.add(content_op.id)

        if isinstance(content_op, InputOp):
            items = inputs_dict.get(content_op.name)
            if not items:
                raise ValueError(
                    f"EmbeddingOp '{llm_op_id}' content references InputOp "
                    f"{content_op.name!r} with no values supplied."
                )
            self._validate_embedding_texts(llm_op_id, items)
            data_spec: dict[str, Any] = {"type": "list", "items": list(items)}
            dependencies: list[str] = []
        elif isinstance(content_op, DataOp):
            if not content_op.data:
                raise ValueError(f"EmbeddingOp '{llm_op_id}' has empty text input")
            self._validate_embedding_texts(llm_op_id, content_op.data)
            data_spec = {"type": "list", "items": list(content_op.data)}
            dependencies = []
        else:
            if graph_dict is not None:
                self._guard_single_row_node_ref(
                    consumer_id=llm_op_id,
                    upstream_id=content_op.id,
                    upstream=content_op,
                    kind="content",
                )
            content_path = self._op_output(content_op).path
            data_spec = {
                "type": "list",
                "node": content_op.id,
                "path": content_path,
            }
            dependencies = [content_op.id]

        # ``model.vllm`` presence routes to the vLLM embedding executor; the
        # executor sets ``runner`` internally, so it is intentionally omitted.
        model_spec = self._build_model_spec(llm_op.config, "vllm", {"convert": "embed"})
        return self._create_runtime_op(
            name=llm_op_id,
            task_type="embedding",
            data_spec=data_spec,
            model_spec=model_spec,
            inference_spec={},
            backend="vllm",
            model=llm_op.config.model,
            dependencies=dependencies or None,
            # Vectors are returned as a safetensors artifact, not inline; the
            # tensor is row-aligned to the input texts.
            output_spec=self._build_output_spec(
                _default_output_destination(),
                ["results.json", "embeddings.safetensors"],
            ),
        )
