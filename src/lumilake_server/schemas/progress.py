from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProgressDetails(BaseModel):
    succeeded: int = Field(default=0, description="Succeeded count.")
    failed: int = Field(default=0, description="Failed count.")
    pending: int = Field(default=0, description="Pending count.")
    dispatched: int = Field(default=0, description="Dispatched count.")

    @field_validator("succeeded", "failed", "pending", "dispatched", mode="before")
    @classmethod
    def _coerce_none_to_zero(cls, value: int | None) -> int:
        return 0 if value is None else value


class ProgressStep(BaseModel):
    completed: bool = Field(default=False, description="Whether the step completed.")
    details: ProgressDetails | None = Field(
        default=None, description="Optional status breakdown for the step."
    )


class BatchNodeStats(BaseModel):
    model_config = ConfigDict(extra="allow")

    succeeded: int = Field(default=0, description="Succeeded node count.")
    failed: int = Field(default=0, description="Failed node count.")
    pending: int = Field(default=0, description="Pending node count.")
    dispatched: int = Field(default=0, description="Dispatched node count.")
    total: int = Field(default=0, description="Total node count.")

    @field_validator(
        "succeeded", "failed", "pending", "dispatched", "total", mode="before"
    )
    @classmethod
    def _coerce_none_to_zero(cls, value: int | None) -> int:
        return 0 if value is None else value


BatchStatus = Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"]


class BatchProgressEntry(BaseModel):
    batch_id: str = Field(description="Batch identifier.")
    status: BatchStatus = Field(
        description=(
            "Batch status from FlowMesh: `PENDING`, `RUNNING`, `COMPLETED`, "
            "or `FAILED`."
        )
    )
    nodes: BatchNodeStats = Field(description="Node status breakdown.")
    elapsed_time: float | None = Field(
        default=None, description="Elapsed time in seconds."
    )


class OverallProgress(BaseModel):
    total_nodes: int = Field(default=0, description="Total nodes observed.")
    completed_nodes: int = Field(default=0, description="Completed node count.")
    percentage: float = Field(default=0.0, description="Overall completion percent.")
    total_inputs: int = Field(default=0, description="Total input item count.")
    completed_inputs: int = Field(default=0, description="Completed input item count.")
    raw_nodes: int = Field(default=0, description="Raw runtime node count.")
    flowmesh_nodes: int = Field(default=0, description="FlowMesh node count.")
    total_nodes_runtime: int = Field(default=0, description="Total runtime nodes.")
    pending_runtime_nodes_raw: int = Field(
        default=0, description="Runtime nodes pending batch aggregation (raw)."
    )
    processing_runtime_nodes_raw: int = Field(
        default=0, description="Runtime nodes currently being processed (raw)."
    )
    processing_runtime_nodes_optimized: int = Field(
        default=0, description="Runtime nodes currently being processed (optimized)."
    )
    processed_runtime_nodes_raw: int = Field(
        default=0, description="Runtime nodes already processed (raw)."
    )
    processed_runtime_nodes_optimized: int = Field(
        default=0, description="Runtime nodes already processed (optimized)."
    )

    @field_validator(
        "total_nodes_runtime",
        "total_inputs",
        "completed_inputs",
        "raw_nodes",
        "flowmesh_nodes",
        "pending_runtime_nodes_raw",
        "processing_runtime_nodes_raw",
        "processing_runtime_nodes_optimized",
        "processed_runtime_nodes_raw",
        "processed_runtime_nodes_optimized",
        mode="before",
    )
    @classmethod
    def _coerce_none_to_zero(cls, value: int | None) -> int:
        return 0 if value is None else value


class BatchProgress(BaseModel):
    total: int = Field(default=0, description="Total batch count.")
    completed: int = Field(default=0, description="Completed batch count.")
    running: int = Field(default=0, description="Running batch count.")
    pending: int = Field(default=0, description="Pending batch count.")
    failed: int = Field(default=0, description="Failed batch count.")
    batches: list[BatchProgressEntry] = Field(
        default_factory=list, description="Per-batch progress entries."
    )
    overall_progress: OverallProgress = Field(
        default_factory=OverallProgress, description="Overall progress aggregation."
    )
    eta_seconds: float | None = Field(
        default=None, description="Estimated time remaining in seconds."
    )


class JobProgress(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    queuing: ProgressStep = Field(default_factory=ProgressStep)
    query_parsing: ProgressStep = Field(
        default_factory=ProgressStep, alias="query parsing"
    )
    data_probing: ProgressStep = Field(
        default_factory=ProgressStep, alias="data probing"
    )
    execution: ProgressStep = Field(default_factory=ProgressStep)
    outputs: ProgressStep = Field(default_factory=ProgressStep)
    batch_progress: BatchProgress = Field(default_factory=BatchProgress)

    def apply_status(self, status: dict[str, Any]) -> None:
        if "data probing" in status:
            self.data_probing = ProgressStep.model_validate(status["data probing"])
        if "execution" in status:
            self.execution = ProgressStep.model_validate(status["execution"])
        if "outputs" in status:
            self.outputs = ProgressStep.model_validate(status["outputs"])
        if "batch_progress" in status:
            self.batch_progress = BatchProgress.model_validate(status["batch_progress"])
