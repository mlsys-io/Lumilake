from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from lumilake_server.data_specs import DataSpecModel, EmptyData, GpuRequest


@dataclass(frozen=True, slots=True)
class DBQuery:
    """Represents a templated SQL query defined in the YAML graph."""

    name: str
    sql: str
    parameters: dict[str, Any]
    post_llm: bool = False
    result_mappings: dict[str, str] = field(default_factory=dict)
    required_inputs: Sequence[str] = field(default_factory=tuple)
    param_types: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueryPlanChoice:
    """Plan outcome (cost/explain)."""

    plan_id: str
    description: str
    cost: float | None
    raw_cost: float | None = None
    explain_json: Any | None = None
    footprints: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Node:
    """A computation or IO unit from the YAML graph."""

    id: str
    type: str
    engine: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    inputs: Sequence[str] = field(default_factory=tuple)
    outputs: Sequence[str] = field(default_factory=tuple)
    db_queries: Sequence[DBQuery] = field(default_factory=tuple)
    gpu_memory_utilization: float | None = None
    gpu_request: GpuRequest | None = None
    data_spec: DataSpecModel = field(default_factory=EmptyData)
    inference_spec: Mapping[str, Any] = field(default_factory=dict)
    api_spec: Mapping[str, Any] = field(default_factory=dict)
    parent: str | None = None


@dataclass(frozen=True, slots=True)
class Edge:
    """Directed data dependency between nodes."""

    source: str
    target: str
    mapping: dict[str, str]


@dataclass(frozen=True, slots=True)
class GraphSpec:
    """In-memory representation of the YAML graph template."""

    name: str
    description: str
    nodes: dict[str, Node]
    edges: Sequence[Edge]


@dataclass(frozen=True, slots=True)
class Worker:
    """An execution worker (GPU or CPU) with its task types and owned GPU fraction."""

    id: str
    kind: str  # "gpu" or "cpu"
    device: str
    capacity: float = 1.0
    task_types: frozenset[str] = frozenset()
    gpu_memory_fraction: float | None = None
