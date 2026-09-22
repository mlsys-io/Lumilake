"""Common models shared across the FlowMesh SDK."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class OkResponse(BaseModel):
    ok: bool


class VersionResponse(BaseModel):
    version: str


class WorkflowStatus(StrEnum):
    PENDING = "PENDING"
    DISPATCHED = "DISPATCHED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    DONE = "DONE"


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    DISPATCHED = "DISPATCHED"
    CANCELLING = "CANCELLING"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    DONE = "DONE"


TERMINAL_TASK_STATUSES = frozenset(
    {TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.DONE}
)

TERMINAL_WORKFLOW_STATUSES = frozenset(
    {WorkflowStatus.FAILED, WorkflowStatus.CANCELLED, WorkflowStatus.DONE}
)


class WorkerStatus(StrEnum):
    UNKNOWN = "UNKNOWN"
    STARTING = "STARTING"
    IDLE = "IDLE"
    BUSY = "BUSY"
    UNAVAILABLE = "UNAVAILABLE"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        return cls.UNKNOWN


class TaskType(StrEnum):
    INFERENCE = "inference"
    RAG = "rag"
    DIFFUSION = "diffusion"
    API = "api"
    SFT = "sft"
    LORA_SFT = "lora_sft"
    PPO = "ppo"
    DPO = "dpo"
    IMAGE_CLASSIFICATION_TRAINING = "image_classification_training"
    ECHO = "echo"
    AGENT = "agent"
    DATA_PROFILING = "data_profiling"
    DATA_RETRIEVAL = "data_retrieval"
    EMBEDDING = "embedding"
    SSH = "ssh"
    OMNI_TEXT2IMAGE = "omni_text2image"
    OMNI_TEXT2SPEECH = "omni_text2speech"
    OMNI_TEXT2AUDIO = "omni_text2audio"
    OMNI_TEXT2GENERAL = "omni_text2general"
    SERVE = "serve"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LogStream(StrEnum):
    STDOUT = "stdout"
    STDERR = "stderr"
    SYSTEM = "system"


class LogEvent(BaseModel):
    ts: str | None = None
    workflow_id: str | None = None
    task_id: str | None = None
    worker_id: str | None = None
    node_id: str | None = None
    level: str | None = None
    stream: str | None = None
    source: str | None = None
    message: str | None = None
    fields: dict[str, Any] | None = None


class LogEntry(BaseModel):
    cursor: str
    event: LogEvent


class LogQueryResponse(BaseModel):
    entries: list[LogEntry]
    next_cursor: str | None = None
    prev_cursor: str | None = None
