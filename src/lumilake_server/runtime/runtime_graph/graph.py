import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from lumilake_server.data_specs import (
    PYTHON_TASK_TYPE,
    DataFrame,
    ListData,
    PythonData,
    remap_model,
    remap_node_dict,
)
from lumilake_server.runtime import python_step
from lumilake_server.runtime.runtime_ops import RuntimeOp, RuntimeOpSchema
from lumilake_server.runtime.sensitive import redact_sensitive
from lumilake_server.utils.graph import topological_sort

_API_CREDENTIAL_PLACEHOLDER = "${credential}"


class RuntimeGraphSchema(BaseModel):
    """Pydantic schema for a serialized ``RuntimeGraph``.

    The dataclass below stays the in-process representation. This model
    is used at boundaries (HTTP, IPC) to enforce field shape, types,
    required fields, and reject unknown keys.
    """

    model_config = ConfigDict(extra="forbid")

    nodes: dict[str, RuntimeOpSchema]
    node_order: list[str]
    output_node_map: dict[str, str] = Field(default_factory=dict)
    output_paths: dict[str, str] = Field(default_factory=dict)
    dsl_to_runtime: dict[str, list[str]] = Field(default_factory=dict)


class Roles(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


def _sanitize_node_prefix(prefix: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", prefix).strip("_")
    return safe or "graph"


def make_node_prefix(name: str) -> str:
    safe = _sanitize_node_prefix(name)
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return f"{safe}_{digest}"


_CHUNK_GROUP_SEPARATOR = "::"


def make_chunk_group_key(parent: str, index: int) -> str:
    """The graph key of one chunk group of ``parent``: the first group keeps
    the parent key, later ones append their index."""
    if index == 0:
        return parent
    return f"{parent}{_CHUNK_GROUP_SEPARATOR}{index}"


def chunk_group_parent(group_key: str) -> str:
    """The parent workflow key a chunk group was split from."""
    parent, _, chunk = group_key.rpartition(_CHUNK_GROUP_SEPARATOR)
    if parent and chunk.isdigit():
        return parent
    return group_key


@dataclass
class RuntimeGraph:
    nodes: dict[str, RuntimeOp]
    node_order: list[str]
    output_node_map: dict[str, str]
    output_paths: dict[str, str] = field(default_factory=dict)
    dsl_to_runtime: dict[str, list[str]] = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def serialize(self) -> dict[str, Any]:
        payload = RuntimeGraphSchema(
            nodes={
                nid: RuntimeOpSchema.model_validate(op.serialize())
                for nid, op in self.nodes.items()
            },
            node_order=self.node_order,
            output_node_map=self.output_node_map,
            output_paths=self.output_paths,
            dsl_to_runtime=self.dsl_to_runtime,
        ).model_dump(exclude_none=True)
        return redact_sensitive(payload)

    @classmethod
    def from_schema(cls, schema: RuntimeGraphSchema) -> Self:
        nodes = {
            nid: RuntimeOp.from_schema(nid, op_schema)
            for nid, op_schema in schema.nodes.items()
        }
        missing = [nid for nid in schema.node_order if nid not in nodes]
        if missing:
            raise ValueError(f"node_order references unknown nodes: {missing}")
        return cls(
            nodes=nodes,
            node_order=schema.node_order,
            output_node_map=schema.output_node_map,
            output_paths=schema.output_paths,
            dsl_to_runtime=schema.dsl_to_runtime,
        )

    @classmethod
    def deserialize(cls, payload: Mapping[str, Any]) -> Self:
        return cls.from_schema(RuntimeGraphSchema.model_validate(payload))

    def to_flowmesh_nodes(self) -> list[dict[str, Any]]:
        # ``RuntimeOp.dependencies`` may reference DSL op ids that aren't
        # materialized as runtime/FlowMesh nodes (e.g. an ``InputOp``
        # consumed only as a retrieval param). FlowMesh schedules tasks
        # whose deps it can resolve — including a non-FlowMesh id in
        # ``dependsOn`` pins the task in PENDING forever. Filter the
        # exported deps to real runtime nodes here. ``topological_order``
        # already preserves ordering across these deps.
        valid_node_ids = set(self.nodes)
        exported: list[dict[str, Any]] = []
        for node_id in self.node_order:
            node_payload = self.nodes[node_id].to_flowmesh_node()
            if "dependsOn" in node_payload:
                filtered = [
                    dep for dep in node_payload["dependsOn"] if dep in valid_node_ids
                ]
                if filtered:
                    node_payload["dependsOn"] = filtered
                else:
                    node_payload.pop("dependsOn", None)
            exported.append(node_payload)
        return exported

    def topological_order(self) -> list[str]:
        # A runtime node's ``dependencies`` may reference DSL-layer op ids
        # that never materialized as runtime nodes (e.g. an ``InputOp``
        # consumed as a retrieval param). We include those in the
        # topological-sort graph so their ordering constraints are
        # respected, then filter the result back down to ids that
        # actually exist in ``self.nodes`` — callers iterate the result
        # expecting to index into ``self.nodes``.
        graph: dict[str, set[str]] = {node_id: set() for node_id in self.nodes}
        for node_id, node in self.nodes.items():
            for dep in node.dependencies:
                if dep not in graph:
                    graph[dep] = set()
                graph[dep].add(node_id)
        order_index = {node_id: idx for idx, node_id in enumerate(self.node_order)}
        return [
            node_id
            for node_id in topological_sort(
                graph,
                secondary_key=lambda node_id: order_index.get(
                    node_id,
                    len(order_index),
                ),
            )
            if node_id in self.nodes
        ]

    def list_lambda_cardinality_nodes(self) -> set[str]:
        """Node ids emitting one whole-list value: whole-result Lambdas, dependents."""
        list_lambda_nodes = {
            node_id
            for node_id, node in self.nodes.items()
            if node.task_type == PYTHON_TASK_TYPE
            and isinstance(node.data_spec, PythonData)
            and node.data_spec.mode in python_step.WHOLE_RESULT_MODES
        }
        list_lambda_dependent: set[str] = set()
        for node_id in self.node_order:
            node = self.nodes[node_id]
            if node_id in list_lambda_nodes:
                list_lambda_dependent.add(node_id)
                continue
            if node.task_type == PYTHON_TASK_TYPE and not (
                isinstance(node.data_spec, PythonData)
                and node.data_spec.mode in python_step.WHOLE_RESULT_MODES
            ):
                continue
            if any(dep in list_lambda_dependent for dep in node.dependencies):
                list_lambda_dependent.add(node_id)
        return list_lambda_dependent

    def list_lambda_fanout_nodes(self) -> set[str]:
        """Node ids whose output count a list-mode Lambda's fan-out sets, not
        the job's input rows: every node below a list-mode Lambda that reads
        nothing but nodes of that fan-out."""
        seeds = {
            node_id
            for node_id, node in self.nodes.items()
            if node.task_type == PYTHON_TASK_TYPE
            and isinstance(node.data_spec, PythonData)
            and node.data_spec.mode == "list"
        }
        reached: set[str] = set()
        for node_id in self.node_order:
            node = self.nodes[node_id]
            if node_id in seeds:
                reached.add(node_id)
                continue
            reads = self._fanout_graph_reads(node)
            if reads and all(read in reached for read in reads):
                reached.add(node_id)
        return reached - self.list_lambda_cardinality_nodes()

    @staticmethod
    def _fanout_graph_reads(node: RuntimeOp) -> set[str] | None:
        """The node ids a node reads, or ``None`` when it also reads a literal
        value such as the job's input rows."""
        data_spec = node.data_spec
        if isinstance(data_spec, PythonData):
            reads: set[str] = set()
            for entry in data_spec.plan:
                if entry.kind != "stage" or entry.node is None:
                    return None
                reads.add(entry.node)
            return reads
        if isinstance(data_spec, DataFrame) and any(
            column.data is not None for column in data_spec.columns
        ):
            return None
        return set(node.dependencies)

    def list_lambda_direct_read_nodes(self) -> set[str]:
        """Node ids whose list column binds a whole-result ``LambdaOp`` one hop away."""
        reads: dict[str, str | None] = {}
        for node_id, node in self.nodes.items():
            data_spec = node.data_spec
            if isinstance(data_spec, ListData):
                reads[node_id] = data_spec.node
        return {
            node_id
            for node_id, node_ref in reads.items()
            if node_ref is not None
            and (upstream := self.nodes.get(node_ref)) is not None
            and upstream.task_type == PYTHON_TASK_TYPE
            and isinstance(upstream.data_spec, PythonData)
            and upstream.data_spec.mode in python_step.WHOLE_RESULT_MODES
        }

    def with_node_prefix(self, prefix: str, separator: str = "__") -> Self:
        if not prefix:
            return self

        mapping = {node_id: f"{prefix}{separator}{node_id}" for node_id in self.nodes}

        nodes: dict[str, RuntimeOp] = {}
        for old_id, op in self.nodes.items():
            new_id = mapping[old_id]
            rewritable = {
                dep: mapping[dep] for dep in op.dependencies if dep in mapping
            }
            nodes[new_id] = RuntimeOp(
                node_id=new_id,
                task_type=op.task_type,
                backend=op.backend,
                model=op.model,
                data_spec=remap_model(op.data_spec, rewritable),
                model_spec=remap_node_dict(op.model_spec, rewritable),
                inference_spec=remap_node_dict(op.inference_spec, rewritable),
                api_spec=remap_node_dict(op.api_spec, rewritable),
                output_enforcement=op.output_enforcement,
                dependencies=tuple(mapping.get(dep, dep) for dep in op.dependencies),
                output_spec=(
                    remap_node_dict(op.output_spec, rewritable)
                    if op.output_spec is not None
                    else None
                ),
                condition=(
                    remap_node_dict(op.condition, rewritable)
                    if op.condition is not None
                    else None
                ),
                skip_when_all_inputs_empty=op.skip_when_all_inputs_empty,
                samples=op.samples,
            )

        node_order = [mapping[node_id] for node_id in self.node_order]
        output_node_map = {
            mapping[node_id]: output_name
            for node_id, output_name in self.output_node_map.items()
        }
        output_paths = {
            mapping[node_id]: path for node_id, path in self.output_paths.items()
        }
        dsl_to_runtime = {
            op_id: [mapping.get(node_id, node_id) for node_id in runtime_ids]
            for op_id, runtime_ids in self.dsl_to_runtime.items()
        }

        return type(self)(
            nodes=nodes,
            node_order=node_order,
            output_node_map=output_node_map,
            output_paths=output_paths,
            dsl_to_runtime=dsl_to_runtime,
        )


def merge_runtime_graphs(
    graphs: dict[str, RuntimeGraph],
) -> tuple[RuntimeGraph, dict[str, tuple[str, str]]]:
    nodes: dict[str, RuntimeOp] = {}
    node_order: list[str] = []
    output_node_map: dict[str, str] = {}
    output_paths: dict[str, str] = {}
    dsl_to_runtime: dict[str, list[str]] = {}
    output_mapping: dict[str, tuple[str, str]] = {}

    for graph_name, graph in graphs.items():
        for node_id in graph.node_order:
            if node_id in nodes:
                raise ValueError(f"Duplicate runtime node id across graphs: {node_id}")
            nodes[node_id] = graph.nodes[node_id]
            node_order.append(node_id)
        for op_id, runtime_ids in graph.dsl_to_runtime.items():
            existing = dsl_to_runtime.get(op_id)
            if existing is None:
                dsl_to_runtime[op_id] = list(runtime_ids)
                continue
            for runtime_id in runtime_ids:
                if runtime_id not in existing:
                    existing.append(runtime_id)
        for node_id, output_name in graph.output_node_map.items():
            output_node_map[node_id] = output_name
            output_mapping[node_id] = (graph_name, output_name)
        for node_id, path in graph.output_paths.items():
            output_paths[node_id] = path

    return (
        RuntimeGraph(
            nodes=nodes,
            node_order=node_order,
            output_node_map=output_node_map,
            output_paths=output_paths,
            dsl_to_runtime=dsl_to_runtime,
        ),
        output_mapping,
    )
