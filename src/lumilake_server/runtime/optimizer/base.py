"""Base optimizer class for Lumilake runtime supporting multiple implementations."""

import json
from abc import ABC, abstractmethod
from typing import Any, cast

from lumilake.log import Logger, LogLevel, init_child_logger
from lumilake_hook import Schedule

from lumilake_server.runtime.runtime_graph import RuntimeGraph, merge_runtime_graphs
from lumilake_server.runtime.runtime_ops import RuntimeOp

__all__ = ["BaseOptimizer", "Schedule"]


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _normalize_label_tokens(value: object) -> object:
    labels: set[str] = set()

    def collect(node: object) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                if key == "label" and isinstance(item, str):
                    labels.add(item)
                collect(item)
        elif isinstance(node, (list, tuple)):
            for item in node:
                collect(item)

    collect(value)
    if not labels:
        return value

    sorted_labels = sorted(labels)
    label_map = {label: f"__lbl{idx}__" for idx, label in enumerate(sorted_labels)}
    replacement_pairs = sorted(
        label_map.items(),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )

    def replace(node: object) -> object:
        if isinstance(node, dict):
            updated: dict[str, object] = {}
            for key, item in node.items():
                if key == "label" and isinstance(item, str):
                    updated[key] = label_map.get(item, item)
                else:
                    updated[key] = replace(item)
            return updated
        if isinstance(node, list):
            return [replace(item) for item in node]
        if isinstance(node, tuple):
            return tuple(replace(item) for item in node)
        if isinstance(node, str):
            normalized = node
            for old, new in replacement_pairs:
                normalized = normalized.replace(old, new)
            return normalized
        return node

    return replace(value)


def _normalize_spec_for_signature(
    spec: dict[str, Any], mapping: dict[str, str]
) -> dict[str, Any]:
    remapped = _remap_node_spec(spec, mapping)
    normalized = _normalize_label_tokens(remapped)
    if not isinstance(normalized, dict):
        raise TypeError("Expected dict after normalizing runtime spec")
    return cast(dict[str, Any], normalized)


def _runtime_signature(op: RuntimeOp, mapping: dict[str, str]) -> tuple[object, ...]:
    normalized_data_spec = _normalize_spec_for_signature(op.data_spec, mapping)
    normalized_model_spec = _normalize_spec_for_signature(op.model_spec, mapping)
    normalized_inference_spec = _normalize_spec_for_signature(
        op.inference_spec, mapping
    )
    normalized_output_spec = (
        _normalize_spec_for_signature(op.output_spec, mapping)
        if op.output_spec is not None
        else None
    )
    normalized_dependencies = tuple(mapping.get(dep, dep) for dep in op.dependencies)
    return (
        op.task_type,
        op.backend,
        op.model,
        _stable_json(normalized_data_spec),
        _stable_json(normalized_model_spec),
        _stable_json(normalized_inference_spec),
        (
            _stable_json(normalized_output_spec)
            if normalized_output_spec is not None
            else None
        ),
        normalized_dependencies,
    )


def _dedupe_ordered(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _remap_node_refs(value: object, mapping: dict[str, str]) -> object:
    if isinstance(value, dict):
        updated: dict[str, object] = {}
        for key, item in value.items():
            if key == "node" and isinstance(item, str) and item in mapping:
                updated[key] = mapping[item]
            else:
                updated[key] = _remap_node_refs(item, mapping)
        return updated
    if isinstance(value, list):
        return [_remap_node_refs(item, mapping) for item in value]
    if isinstance(value, tuple):
        return tuple(_remap_node_refs(item, mapping) for item in value)
    return value


def _remap_node_spec(spec: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    remapped = _remap_node_refs(spec, mapping)
    if not isinstance(remapped, dict):
        raise TypeError("Expected dict after remapping runtime spec")
    return cast(dict[str, Any], remapped)


def _dedupe_runtime_graph(graph: RuntimeGraph) -> RuntimeGraph:
    """Merge identical runtime ops with identical dependencies (non-output nodes)."""
    output_nodes = set(graph.output_node_map)
    canonical_by_signature: dict[tuple[object, ...], str] = {}
    mapping: dict[str, str] = {}

    for node_id in graph.topological_order():
        if node_id in output_nodes:
            mapping[node_id] = node_id
            continue
        signature = _runtime_signature(graph.nodes[node_id], mapping)
        canonical = canonical_by_signature.get(signature)
        if canonical is None:
            canonical_by_signature[signature] = node_id
            mapping[node_id] = node_id
        else:
            mapping[node_id] = canonical

    if all(node_id == mapped for node_id, mapped in mapping.items()):
        return graph

    new_nodes: dict[str, RuntimeOp] = {}
    new_order: list[str] = []
    for node_id in graph.node_order:
        canonical = mapping[node_id]
        if canonical in new_nodes:
            continue
        op = graph.nodes[canonical]
        remapped_deps = [mapping.get(dep, dep) for dep in op.dependencies]
        new_nodes[canonical] = RuntimeOp(
            node_id=canonical,
            task_type=op.task_type,
            backend=op.backend,
            model=op.model,
            data_spec=_remap_node_spec(op.data_spec, mapping),
            model_spec=_remap_node_spec(op.model_spec, mapping),
            inference_spec=_remap_node_spec(op.inference_spec, mapping),
            dependencies=tuple(_dedupe_ordered(remapped_deps)),
            output_spec=(
                _remap_node_spec(op.output_spec, mapping)
                if op.output_spec is not None
                else None
            ),
        )
        new_order.append(canonical)

    new_output_node_map = {
        mapping[node_id]: output_name
        for node_id, output_name in graph.output_node_map.items()
    }
    new_dsl_to_runtime = {
        op_id: _dedupe_ordered([mapping.get(rid, rid) for rid in runtime_ids])
        for op_id, runtime_ids in graph.dsl_to_runtime.items()
    }

    return RuntimeGraph(
        nodes=new_nodes,
        node_order=new_order,
        output_node_map=new_output_node_map,
        dsl_to_runtime=new_dsl_to_runtime,
    )


class BaseOptimizer(ABC):
    """Abstract base class for Lumilake optimizers."""

    def __init__(
        self,
        logger: Logger | None = None,
        log_level: LogLevel | None = None,
    ) -> None:
        self.logger: Logger = init_child_logger(
            f"Optimizer.{self.__class__.__name__}", logger, log_level
        )

    @abstractmethod
    def generate_schedule(
        self,
        graph: RuntimeGraph,
        worker_names: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        data_profile_results: dict[str, list[dict[str, Any]]] | None = None,
    ) -> Schedule:
        """Generate execution schedule for operations in the batch.

        Parameters
        ----------
        graph : RuntimeGraph
            Runtime graph in the batch
        worker_names : list[str]
            Workers selected for this batch.
        worker_profiles : dict[str, dict[str, Any]]
            Worker hardware/profile information keyed by worker id.
        data_profile_results : dict[str, list[dict[str, Any]]] | None, optional
            Data profile outputs keyed by "data_profile::<node_id>::<query_name>".

        Returns
        -------
        Schedule
            Worker assignment mapping.
        """
        raise NotImplementedError

    def optimize_graphs(
        self,
        graphs: dict[str, RuntimeGraph],
    ) -> tuple[RuntimeGraph, dict[str, tuple[str, str]]]:
        """Optimize and merge runtime graphs prior to scheduling."""
        merged_graph, output_mapping = merge_runtime_graphs(graphs)
        optimized_graph = _dedupe_runtime_graph(merged_graph)
        return optimized_graph, output_mapping
