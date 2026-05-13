from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


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
    plans: Sequence["QueryPlanOption"] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class QueryPlanOption:
    """Represents a hint-based plan candidate for a DBQuery."""

    id: str
    description: str
    settings: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueryPlanChoice:
    """Plan outcome (cost/explain)."""

    plan_id: str
    description: str
    cost: float | None
    raw_cost: float | None = None
    explain_json: Any | None = None
    samples: Sequence["PlanMetric"] = field(default_factory=tuple)
    footprints: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlanMetric:
    runtime_ms: float | None = None
    shared_hit_blocks: int | None = None
    shared_read_blocks: int | None = None
    local_hit_blocks: int | None = None
    local_read_blocks: int | None = None
    relations: Mapping[str, int] = field(default_factory=dict)
    indexes: Mapping[str, int] = field(default_factory=dict)


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
    raw: dict[str, Any] = field(default_factory=dict)


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
    """Describes an execution worker (GPU or CPU)."""

    id: str
    kind: str  # "gpu" or "cpu"
    device: str
    capacity: float = 1.0


@dataclass(frozen=True, slots=True)
class ExecutionTask:
    """A single unit of execution assigned to a worker."""

    node_id: str
    worker_id: str | Sequence[str]
    dependencies: Sequence[str]
    epoch: int
