"""Pydantic schemas for the schedule protocol endpoint.

Field names are the canonical wire contract; remote implementations must match exactly.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict

from lumilake_server.runtime.runtime_graph import RuntimeGraphSchema


class ScheduleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph: RuntimeGraphSchema
    worker_names: list[str]
    worker_profiles: dict[str, Any]
    data_profile_results: dict[str, list[Any]] | None = None
    optimizer_type: str = "halo"


class ScheduleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_assignment: dict[str, list[str]]


class OptimizerListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    types: list[str]


__all__ = [
    "OptimizerListResponse",
    "ScheduleRequest",
    "ScheduleResponse",
]
