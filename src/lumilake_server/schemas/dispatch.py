"""Raw per-workflow scheduler dispatch telemetry.

One ``WorkflowDispatch`` row is recorded for every actual dispatch of a
workflow from the scheduler queue to the runtime. Rows carry raw facts only;
consumers aggregate later.
"""

from pydantic import BaseModel, ConfigDict, Field


class WorkflowDispatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_id: str = Field(description="Queue item workflow id.")
    graph_name: str = Field(description="Runtime graph name.")
    public_graph_name: str = Field(description="Public workflow graph name.")
    slice_index: int = Field(description="Index of this slice within the workflow.")
    slice_start: int = Field(description="Start offset of this slice.")
    slice_length: int = Field(description="Number of rows in this slice.")
    total_length: int = Field(description="Total rows across all slices.")
    enqueued_at: float = Field(
        description="Epoch seconds the workflow entered the scheduler queue."
    )
    dispatched_at: float = Field(
        description="Epoch seconds the workflow was dispatched to the runtime."
    )
    miss_count: int = Field(
        description="Times the workflow was passed over before this dispatch."
    )
    batch_id: str = Field(description="Scheduler batch id for this dispatch.")
    execution_request_id: str = Field(
        description="Execution request id the batch was submitted under."
    )
    batch_workflow_ids: list[str] = Field(
        description="Queue item ids of every workflow in the same batch."
    )
    workers: list[str] = Field(description="Worker ids claimed for the batch.")
    flowmesh_workflow_id: str | None = Field(
        default=None,
        description="FlowMesh workflow id the batch was submitted as, once known.",
    )
