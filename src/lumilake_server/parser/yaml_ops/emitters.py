"""YAML op names, shared with ``Op.registry``, mapped to their emitters."""

from .classification import emit_classification_op
from .common import OpEmitter
from .data import emit_data_op, emit_data_retrieval_op, emit_format_op, emit_message_op
from .embedding import emit_embedding_op
from .http_fetch import emit_http_fetch_op
from .image import emit_image_generation_op
from .lambdas import emit_lambda_op
from .llm import emit_llm_chat_op, emit_llm_completion_op, emit_llm_vision_op
from .routing import emit_router_op

EMITTERS: dict[str, OpEmitter] = {
    "ClassificationOp": OpEmitter("classification", emit_classification_op),
    "DataOp": OpEmitter("data", emit_data_op),
    "DataRetrievalOp": OpEmitter("retrieval", emit_data_retrieval_op),
    "EmbeddingOp": OpEmitter("embedding", emit_embedding_op),
    "FormatOp": OpEmitter("format", emit_format_op),
    "HttpFetchOp": OpEmitter("httpfetch", emit_http_fetch_op),
    "ImageGenerationOp": OpEmitter("imagegen", emit_image_generation_op),
    "LLMChatOp": OpEmitter("llm", emit_llm_chat_op),
    "LLMCompletionOp": OpEmitter("llmcompletion", emit_llm_completion_op),
    "LLMVisionOp": OpEmitter("llmvision", emit_llm_vision_op),
    "LambdaOp": OpEmitter("lambda", emit_lambda_op),
    "MessageOp": OpEmitter("message", emit_message_op),
    "RouterOp": OpEmitter("router", emit_router_op),
}


def op_id_prefix(op_type: str) -> str:
    """Id prefix for an op type, matching ``EMITTERS``; falls back to the name."""
    emitter = EMITTERS.get(op_type)
    return emitter.prefix if emitter is not None else op_type.lower()
