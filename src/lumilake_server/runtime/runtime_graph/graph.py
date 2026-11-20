import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from lumilake_server.runtime import python_step
from lumilake_server.runtime.runtime_ops import RuntimeOp, RuntimeOpSchema
from lumilake_server.runtime.sensitive import redact_sensitive
from lumilake_server.utils.graph import topological_sort

_PLACEHOLDER_RE = re.compile(r"\$\{([^}.]+)\.([^}]+)\}")
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
    def from_schema(cls, schema: "RuntimeGraphSchema") -> "RuntimeGraph":
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
    def deserialize(cls, payload: Mapping[str, Any]) -> "RuntimeGraph":
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
        """Node ids that emit one whole-list value per run: list, expand and
        reduce mode Lambdas and their dependents, up to a row-mode python step.
        Aligned-mode Lambdas are row-cardinality, so they are neither seeds nor
        barriers here."""
        list_lambda_nodes = {
            node_id
            for node_id, node in self.nodes.items()
            if node.task_type == python_step.TASK_TYPE
            and node.data_spec.get("mode") in ("list", "expand", "reduce")
        }
        list_lambda_dependent: set[str] = set()
        for node_id in self.node_order:
            node = self.nodes[node_id]
            if node_id in list_lambda_nodes:
                list_lambda_dependent.add(node_id)
                continue
            if node.task_type == python_step.TASK_TYPE and node.data_spec.get(
                "mode"
            ) not in ("list", "expand", "reduce"):
                continue
            if any(dep in list_lambda_dependent for dep in node.dependencies):
                list_lambda_dependent.add(node_id)
        return list_lambda_dependent

    def list_lambda_direct_read_nodes(self) -> set[str]:
        """Node ids whose list column binds a list, expand or reduce mode
        ``LambdaOp`` one hop away. FlowMesh runs such a task once per element,
        so the task's items are the whole list and the aggregator folds them
        into one value."""
        return {
            node_id
            for node_id, node in self.nodes.items()
            if node.data_spec.get("type") == "list"
            and (upstream := self.nodes.get(node.data_spec.get("node", ""))) is not None
            and upstream.task_type == python_step.TASK_TYPE
            and upstream.data_spec.get("mode") in ("list", "expand", "reduce")
        }

    def with_node_prefix(self, prefix: str, separator: str = "__") -> Self:
        if not prefix:
            return self

        mapping = {node_id: f"{prefix}{separator}{node_id}" for node_id in self.nodes}

        def _remap_placeholder(text: str, rewritable: dict[str, str]) -> str:
            def _sub(match: re.Match[str]) -> str:
                node = match.group(1)
                prefixed = rewritable.get(node)
                if prefixed is None:
                    return match.group(0)
                return f"${{{prefixed}.{match.group(2)}}}"

            return _PLACEHOLDER_RE.sub(_sub, text)

        def remap(value: Any, rewritable: dict[str, str]) -> Any:
            if isinstance(value, dict):
                updated: dict[str, Any] = {}
                for key, item in value.items():
                    if key == "node" and isinstance(item, str) and item in mapping:
                        updated[key] = mapping[item]
                    else:
                        updated[key] = remap(item, rewritable)
                return updated
            if isinstance(value, list):
                return [remap(item, rewritable) for item in value]
            if isinstance(value, tuple):
                return tuple(remap(item, rewritable) for item in value)
            if isinstance(value, str):
                return _remap_placeholder(value, rewritable)
            return value

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
                data_spec=remap(op.data_spec, rewritable),
                model_spec=remap(op.model_spec, rewritable),
                inference_spec=remap(op.inference_spec, rewritable),
                api_spec=remap(op.api_spec, rewritable),
                output_enforcement=op.output_enforcement,
                dependencies=tuple(mapping.get(dep, dep) for dep in op.dependencies),
                output_spec=(
                    remap(op.output_spec, rewritable)
                    if op.output_spec is not None
                    else None
                ),
                condition=(
                    remap(op.condition, rewritable)
                    if op.condition is not None
                    else None
                ),
                skip_when_all_inputs_empty=op.skip_when_all_inputs_empty,
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
