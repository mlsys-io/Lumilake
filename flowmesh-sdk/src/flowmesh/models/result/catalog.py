# Necessary for the recursive ``children`` forward reference.
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    SerializeAsAny,
    Tag,
)

from ..artifacts import ArtifactRef
from ..common import TaskType
from ._base import BaseExecutorResult, StrictExecutorResult
from .payloads import (
    AgentItem,
    AgentMetadata,
    AgentUsage,
    CostEstimates,
    DataRetrievalItem,
    EchoItem,
    EmbeddingUsage,
    GenerationUsage,
    InferenceItem,
    OmniAudioItem,
    OmniGeneralItem,
    OmniImageItem,
    OmniSpeechItem,
    RagEmbedding,
    RagQdrant,
    RagQuery,
    RagSearch,
    RagUsage,
)


class InferenceResult(StrictExecutorResult):
    task_type: Literal[TaskType.INFERENCE] = TaskType.INFERENCE
    model: str | None = None
    items: list[InferenceItem] = Field(default_factory=list)
    usage: GenerationUsage | None = None


class EmbeddingResult(StrictExecutorResult):
    task_type: Literal[TaskType.EMBEDDING] = TaskType.EMBEDDING
    model: str | None = None
    embedding_file: ArtifactRef | None = None
    usage: EmbeddingUsage | None = None
    count: int | None = None
    image_group_sizes: list[int] | None = None


class DiffusionResult(StrictExecutorResult):
    task_type: Literal[TaskType.DIFFUSION] = TaskType.DIFFUSION
    model: str | None = None
    images: list[ArtifactRef] = Field(default_factory=list)


class ServeResult(StrictExecutorResult):
    task_type: Literal[TaskType.SERVE] = TaskType.SERVE
    model: str
    port: int


class _TrainingResult(StrictExecutorResult):
    training_time_seconds: float | None = None
    error_message: str | None = None
    model_name: str | None = None
    dataset_size: int = 0
    output_dir: str | None = None
    checkpoints_dir: ArtifactRef | None = None


class SFTResult(_TrainingResult):
    task_type: Literal[TaskType.SFT] = TaskType.SFT
    resume_from_path: str | None = None
    final_model: ArtifactRef | None = None
    final_model_archive: ArtifactRef | None = None
    spawned_torchrun: bool = False


class LoRAResult(_TrainingResult):
    task_type: Literal[TaskType.LORA_SFT] = TaskType.LORA_SFT
    resume_from_path: str | None = None
    final_lora: ArtifactRef | None = None
    final_lora_archive: ArtifactRef | None = None


class PPOResult(_TrainingResult):
    task_type: Literal[TaskType.PPO] = TaskType.PPO
    final_model: ArtifactRef | None = None
    final_model_archive: ArtifactRef | None = None
    spawned_torchrun: bool = False


class DPOResult(_TrainingResult):
    task_type: Literal[TaskType.DPO] = TaskType.DPO
    final_model: ArtifactRef | None = None
    final_model_archive: ArtifactRef | None = None
    spawned_torchrun: bool = False


class ImageClassificationTrainingResult(_TrainingResult):
    task_type: Literal[TaskType.IMAGE_CLASSIFICATION_TRAINING] = (
        TaskType.IMAGE_CLASSIFICATION_TRAINING
    )
    num_labels: int = 0
    eval_accuracy: float | None = None
    train_losses: list[float] = Field(default_factory=list)
    resume_from_path: str | None = None
    final_model: ArtifactRef | None = None
    final_model_archive: ArtifactRef | None = None


class OmniResult(StrictExecutorResult):
    executor: str
    mode: str
    model: str


class OmniText2ImageResult(OmniResult):
    task_type: Literal[TaskType.OMNI_TEXT2IMAGE] = TaskType.OMNI_TEXT2IMAGE
    executor: str = "omni_text2image"
    mode: str = "image"
    image: ArtifactRef | None
    items: list[OmniImageItem]


class OmniText2SpeechResult(OmniResult):
    task_type: Literal[TaskType.OMNI_TEXT2SPEECH] = TaskType.OMNI_TEXT2SPEECH
    executor: str = "omni_text2speech"
    mode: str = "tts"
    audio: ArtifactRef | None
    sample_rate: int
    storyboard: dict[str, Any] | None = None
    items: list[OmniSpeechItem]


class OmniText2AudioResult(OmniResult):
    task_type: Literal[TaskType.OMNI_TEXT2AUDIO] = TaskType.OMNI_TEXT2AUDIO
    executor: str = "omni_text2audio"
    mode: str = "bgm"
    audio: ArtifactRef | None
    sample_rate: int
    num_waveforms: int
    audio_length: float
    storyboard: dict[str, Any] | None = None
    items: list[OmniAudioItem]


class OmniText2GeneralResult(OmniResult):
    task_type: Literal[TaskType.OMNI_TEXT2GENERAL] = TaskType.OMNI_TEXT2GENERAL
    executor: str = "omni_text2general"
    mode: str = "narration"
    audio: ArtifactRef | None
    sample_rate: int
    storyboard: dict[str, Any] | None = None
    items: list[OmniGeneralItem]


class DataProfilingResult(StrictExecutorResult):
    task_type: Literal[TaskType.DATA_PROFILING] = TaskType.DATA_PROFILING
    type: str = "sql"
    template: str | None = None
    cost_estimates: CostEstimates | None = None


class DataRetrievalResult(StrictExecutorResult):
    task_type: Literal[TaskType.DATA_RETRIEVAL] = TaskType.DATA_RETRIEVAL
    type: str | None = None
    items: list[DataRetrievalItem] = Field(default_factory=list)
    count: int | None = None
    metadata: dict[str, Any] | None = None


class AgentResult(StrictExecutorResult):
    task_type: Literal[TaskType.AGENT] = TaskType.AGENT
    model: str
    items: list[AgentItem] = Field(default_factory=list)
    usage: AgentUsage | None = None
    metadata: AgentMetadata | None = None
    agent_output: ArtifactRef | None = None
    batch_summary_file: ArtifactRef | None = None


class RAGResult(StrictExecutorResult):
    task_type: Literal[TaskType.RAG] = TaskType.RAG
    executor: str = "rag"
    qdrant: RagQdrant
    embedding: RagEmbedding
    search: RagSearch
    queries: list[RagQuery] = Field(default_factory=list)
    usage: RagUsage | None = None


class EchoResult(StrictExecutorResult):
    task_type: Literal[TaskType.ECHO] = TaskType.ECHO
    items: list[EchoItem] = Field(default_factory=list)
    count: int = 0


class APIResult(StrictExecutorResult):
    task_type: Literal[TaskType.API] = TaskType.API
    executor: str
    method: str
    url: str
    status_code: int
    truncated: bool = False
    headers: dict[str, str] | None = None
    response_json: Any = Field(default=None, alias="json")
    usage: dict[str, Any] | None = None
    text: str | None = None


class SSHResult(StrictExecutorResult):
    task_type: Literal[TaskType.SSH] = TaskType.SSH
    session_id: str
    exit_code: int
    command: list[str] | None = None
    entrypoint: list[str] | None = None
    expires_at: str | None = None
    host: str | None = None
    port: int | None = None


_BASE_TAG = "__base__"

_RESULT_TAGS: frozenset[str] = frozenset(
    {
        TaskType.INFERENCE.value,
        TaskType.EMBEDDING.value,
        TaskType.DIFFUSION.value,
        TaskType.SERVE.value,
        TaskType.SFT.value,
        TaskType.LORA_SFT.value,
        TaskType.PPO.value,
        TaskType.DPO.value,
        TaskType.IMAGE_CLASSIFICATION_TRAINING.value,
        TaskType.OMNI_TEXT2IMAGE.value,
        TaskType.OMNI_TEXT2SPEECH.value,
        TaskType.OMNI_TEXT2AUDIO.value,
        TaskType.OMNI_TEXT2GENERAL.value,
        TaskType.DATA_PROFILING.value,
        TaskType.DATA_RETRIEVAL.value,
        TaskType.AGENT.value,
        TaskType.RAG.value,
        TaskType.ECHO.value,
        TaskType.API.value,
        TaskType.SSH.value,
    }
)


def _result_discriminator(value: Any) -> str:
    if isinstance(value, dict):
        tag = value.get("task_type")
    else:
        tag = getattr(value, "task_type", None)
    if tag is None:
        return _BASE_TAG
    tag = str(tag)
    return tag if tag in _RESULT_TAGS else _BASE_TAG


AnyExecutorResult = Annotated[
    (
        Annotated[InferenceResult, Tag(TaskType.INFERENCE.value)]
        | Annotated[EmbeddingResult, Tag(TaskType.EMBEDDING.value)]
        | Annotated[DiffusionResult, Tag(TaskType.DIFFUSION.value)]
        | Annotated[ServeResult, Tag(TaskType.SERVE.value)]
        | Annotated[SFTResult, Tag(TaskType.SFT.value)]
        | Annotated[LoRAResult, Tag(TaskType.LORA_SFT.value)]
        | Annotated[PPOResult, Tag(TaskType.PPO.value)]
        | Annotated[DPOResult, Tag(TaskType.DPO.value)]
        | Annotated[
            ImageClassificationTrainingResult,
            Tag(TaskType.IMAGE_CLASSIFICATION_TRAINING.value),
        ]
        | Annotated[OmniText2ImageResult, Tag(TaskType.OMNI_TEXT2IMAGE.value)]
        | Annotated[OmniText2SpeechResult, Tag(TaskType.OMNI_TEXT2SPEECH.value)]
        | Annotated[OmniText2AudioResult, Tag(TaskType.OMNI_TEXT2AUDIO.value)]
        | Annotated[OmniText2GeneralResult, Tag(TaskType.OMNI_TEXT2GENERAL.value)]
        | Annotated[DataProfilingResult, Tag(TaskType.DATA_PROFILING.value)]
        | Annotated[DataRetrievalResult, Tag(TaskType.DATA_RETRIEVAL.value)]
        | Annotated[AgentResult, Tag(TaskType.AGENT.value)]
        | Annotated[RAGResult, Tag(TaskType.RAG.value)]
        | Annotated[EchoResult, Tag(TaskType.ECHO.value)]
        | Annotated[APIResult, Tag(TaskType.API.value)]
        | Annotated[SSHResult, Tag(TaskType.SSH.value)]
        | Annotated[BaseExecutorResult, Tag(_BASE_TAG)]
    ),
    Discriminator(_result_discriminator),
]


class ResultEnvelope(BaseModel):
    """Canonical on-disk shape of ``results.json`` (mirrors the server)."""

    task_id: str
    result: SerializeAsAny[AnyExecutorResult]
    worker_id: str | None = None
    metadata: dict[str, Any] | None = None
    received_at: str | None = Field(default=None)
