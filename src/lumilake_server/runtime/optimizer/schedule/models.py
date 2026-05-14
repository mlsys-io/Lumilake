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
