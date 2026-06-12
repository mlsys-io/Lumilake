from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DataProfileCostEstimate(BaseModel):
    plan_id: str = Field(min_length=1)
    description: str | None = None
    raw_cost: float = Field(ge=0)
    estimated_rows: int | None = Field(default=None, ge=0)
    estimated_files: int | None = Field(default=None, ge=0)
    total_size_bytes: int | None = Field(default=None, ge=0)
    avg_file_size_bytes: float | None = Field(default=None, ge=0)
    footprints: dict[str, int] = Field(default_factory=dict)
    explain_json: Any | None = None

    model_config = ConfigDict(extra="ignore")


class DataProfileResultRow(BaseModel):
    node_id: str = Field(min_length=1)
    raw_node_id: str = Field(min_length=1)
    query_name: str = Field(min_length=1)
    table: str
    cost_estimates: list[DataProfileCostEstimate]

    model_config = ConfigDict(extra="ignore")


class DataProfileResultsPayload(BaseModel):
    data_profile_results: dict[str, list[DataProfileResultRow]] = Field(
        default_factory=dict
    )

    model_config = ConfigDict(extra="forbid")
