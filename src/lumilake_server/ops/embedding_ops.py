from dataclasses import dataclass
from typing import Any

from lumilake_server.common import GenerationConfig
from lumilake_server.ops.data_ops import DataOp
from lumilake_server.ops.llm_ops import LLMOp
from lumilake_server.ops.ops import Op


@dataclass(frozen=True)
class EmbeddingArtifact:
    """Reference to a FlowMesh embedding artifact plus its metadata.

    The ``"embeddings"`` tensor lives in ``embedding_file``; its durable
    location is recorded under the top-level ``_artifacts`` context.
    """

    model: str | None
    embedding_file: str
    usage: dict[str, Any]
    artifacts: dict[str, Any]


def _is_nonempty_text(text: Any) -> bool:
    return isinstance(text, str) and bool(text.strip())


def _as_text_list(content: str | list[str]) -> list[str]:
    if isinstance(content, str):
        items = [content]
    else:
        items = list(content)
    if not items or not all(_is_nonempty_text(text) for text in items):
        raise ValueError("EmbeddingOp requires non-empty text input")
    return items


@Op.registry.register("EmbeddingOp")
class EmbeddingOp(LLMOp):
    """Op that embeds text with a FlowMesh-served embedding model.

    Output is one vector per input text; config carries the model id plus
    optional vLLM engine kwargs (``gpu_memory_utilization``,
    ``tensor_parallel_size``).
    """

    def __init__(
        self,
        content: str | list[str] | Op,
        config: GenerationConfig | None = None,
        cacheable: bool = False,
    ) -> None:
        content_op = (
            content if isinstance(content, Op) else DataOp(data=_as_text_list(content))
        )
        super().__init__([content_op], config, cacheable)

    @property
    def content(self) -> Op:
        return self.inputs[0]

    def _serialize(self) -> dict[str, Any]:
        return dict(
            content=self.content.id,
            config=self.config.to_dict(),
            cacheable=self.cacheable,
        )

    @classmethod
    def _from_json(
        cls, data: dict[str, Any], other_ops: dict[str, "Op"]
    ) -> "EmbeddingOp":
        config = GenerationConfig(**data["config"])
        return cls(other_ops[data["content"]], config, data["cacheable"])

    @staticmethod
    def parse_response(result: dict[str, Any]) -> EmbeddingArtifact:
        """Parse a FlowMesh embedding result envelope into an artifact ref.

        Fails fast if ``embedding_file``, ``_artifacts``, or ``usage`` is
        missing, so a malformed response never silently yields empty vectors.
        """
        if not isinstance(result, dict):
            raise ValueError(
                f"EmbeddingOp response must be a mapping (got {type(result).__name__})"
            )
        embedding_file = result.get("embedding_file")
        if not isinstance(embedding_file, dict) or not embedding_file.get("path"):
            raise ValueError(
                "EmbeddingOp response is missing 'embedding_file' with a 'path'"
            )
        artifacts = result.get("_artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError("EmbeddingOp response is missing the '_artifacts' context")
        usage = result.get("usage")
        if not isinstance(usage, dict):
            raise ValueError("EmbeddingOp response is missing the 'usage' block")
        model = result.get("model")
        return EmbeddingArtifact(
            model=str(model) if model is not None else None,
            embedding_file=str(embedding_file["path"]),
            usage=usage,
            artifacts=artifacts,
        )


def embedding(
    content: str | list[str] | Op,
    config: GenerationConfig | None = None,
    cacheable: bool = False,
) -> EmbeddingOp:
    return EmbeddingOp(content, config, cacheable)
