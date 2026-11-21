"""Base optimizer class for Lumilake runtime supporting multiple implementations."""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, cast

from lumilake.log import Logger, LogLevel, init_child_logger
from lumilake_hook import Schedule
from lumilake_server.runtime.runtime_graph import RuntimeGraph, merge_runtime_graphs
from lumilake_server.runtime.runtime_ops import SAMPLING_TASK_TYPES, RuntimeOp

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
    normalized_api_spec = _normalize_spec_for_signature(op.api_spec, mapping)
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
        _stable_json(normalized_api_spec),
        (
            _stable_json(normalized_output_spec)
            if normalized_output_spec is not None
            else None
        ),
        normalized_dependencies,
        _stable_json(op.condition) if op.condition is not None else None,
        (
            _stable_json(op.output_enforcement)
            if op.output_enforcement is not None
            else None
        ),
    )


_TEXT_TASK_TYPES: frozenset[str] = frozenset({"inference", "api"})


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _sampling_spec(op: RuntimeOp) -> dict[str, Any] | None:
    """The sampler settings of a node that generates text or images, or
    ``None`` when the node's output is a pure function of its inputs: a
    Python step, a retrieval, an embedding, a classification, or an API
    request other than a chat completion."""
    if op.task_type in SAMPLING_TASK_TYPES:
        return op.inference_spec
    if op.task_type == "api":
        body = op.api_spec.get("json")
        if isinstance(body, dict) and "messages" in body:
            return body
    return None


def _decodes_deterministically(op: RuntimeOp, sampler: dict[str, Any]) -> bool:
    """Whether the node's text decoder is greedy or beam search. An absent
    temperature is not greedy: the engine then samples at its own default, so
    only an explicit ``temperature: 0`` or a beam width above 1 counts. Image
    samplers read neither setting, so they never count."""
    if op.task_type not in _TEXT_TASK_TYPES:
        return False
    num_beams = _number(sampler.get("num_beams"))
    if num_beams is not None and num_beams > 1:
        return True
    return _number(sampler.get("temperature")) == 0.0


def _dedupe_scope(
    op: RuntimeOp, owners: tuple[tuple[str, str], ...] | None
) -> tuple[Any, ...] | None:
    """The per-node part of the dedupe key, or ``None`` when the node must not
    be merged. A node that samples draws from a distribution rather than
    computing a function of its inputs, so it merges only with the copies of
    its own logical op (``owners``: one ``(workflow, op id)`` pair per copy),
    never with a different op of equal settings, seeded or not. Python steps
    stay eligible, as their output function is the user's to keep pure."""
    sampler = _sampling_spec(op)
    if sampler is None or _decodes_deterministically(op, sampler):
        return ("pure",)
    if not owners:
        return None
    return ("sampled", *owners)


def _workflow_scope(group_name: str) -> str:
    """The parent workflow a graph group was built for. A request whose slices
    were split into chunk groups ``parent``, ``parent::1``, … is one workflow,
    so its chunk groups share a scope."""
    parent, _, chunk = group_name.rpartition("::")
    if parent and chunk.isdigit():
        return parent
    return group_name


def _logical_ops(
    graphs: dict[str, RuntimeGraph],
) -> dict[str, tuple[tuple[str, str], ...]]:
    """Map each runtime node to the ``(workflow, op id)`` pairs it was built
    from. Read from the per-group graphs, before ``merge_runtime_graphs``
    folds their ``dsl_to_runtime`` by bare op id, so two workflows that both
    name an op ``Ask`` keep separate owners."""
    owners: dict[str, list[tuple[str, str]]] = {}
    for group_name, graph in graphs.items():
        scope = _workflow_scope(group_name)
        for op_id, runtime_ids in graph.dsl_to_runtime.items():
            for runtime_id in runtime_ids:
                entry = (scope, op_id)
                pairs = owners.setdefault(runtime_id, [])
                if entry not in pairs:
                    pairs.append(entry)
    return {node_id: tuple(sorted(pairs)) for node_id, pairs in owners.items()}


def _dedupe_ordered(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


_PLACEHOLDER_RE = re.compile(r"\$\{([^}.]+)\.([^}]+)\}")


def _remap_placeholder(text: str, mapping: dict[str, str]) -> str:
    def _sub(match: re.Match[str]) -> str:
        node = match.group(1)
        remapped = mapping.get(node)
        if remapped is None:
            return match.group(0)
        return f"${{{remapped}.{match.group(2)}}}"

    return _PLACEHOLDER_RE.sub(_sub, text)


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
    if isinstance(value, str):
        return _remap_placeholder(value, mapping)
    return value


def _remap_node_spec(spec: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    remapped = _remap_node_refs(spec, mapping)
    if not isinstance(remapped, dict):
        raise TypeError("Expected dict after remapping runtime spec")
    return cast(dict[str, Any], remapped)


def _dedupe_runtime_graph(
    graph: RuntimeGraph,
    logical_ops: dict[str, tuple[tuple[str, str], ...]],
) -> tuple[RuntimeGraph, dict[str, str]]:
    """Merge identical runtime ops with identical dependencies (non-output nodes).

    ``logical_ops`` maps a runtime node to the ``(workflow, op id)`` pairs it
    was built from; ``_dedupe_scope`` needs it to keep sampling nodes of
    different ops apart.

    Returns the deduplicated graph and the ``node_id -> canonical node_id``
    mapping (identity for nodes that were not deduplicated).
    """
    output_nodes = set(graph.output_node_map)
    canonical_by_signature: dict[tuple[object, ...], str] = {}
    mapping: dict[str, str] = {}

    for node_id in graph.topological_order():
        if node_id in output_nodes:
            mapping[node_id] = node_id
            continue
        op = graph.nodes[node_id]
        scope = _dedupe_scope(op, logical_ops.get(node_id))
        if scope is None:
            mapping[node_id] = node_id
            continue
        signature = (*scope, *_runtime_signature(op, mapping))
        canonical = canonical_by_signature.get(signature)
        if canonical is None:
            canonical_by_signature[signature] = node_id
            mapping[node_id] = node_id
        else:
            mapping[node_id] = canonical

    if all(node_id == mapped for node_id, mapped in mapping.items()):
        return graph, mapping

    new_nodes: dict[str, RuntimeOp] = {}
    new_order: list[str] = []
    for node_id in graph.node_order:
        canonical = mapping[node_id]
        if canonical in new_nodes:
            continue
        op = graph.nodes[canonical]
        remapped_deps = [mapping.get(dep, dep) for dep in op.dependencies]
        new_nodes[canonical] = replace(
            op,
            node_id=canonical,
            data_spec=_remap_node_spec(op.data_spec, mapping),
            model_spec=_remap_node_spec(op.model_spec, mapping),
            inference_spec=_remap_node_spec(op.inference_spec, mapping),
            api_spec=_remap_node_spec(op.api_spec, mapping),
            dependencies=tuple(_dedupe_ordered(remapped_deps)),
            output_spec=(
                _remap_node_spec(op.output_spec, mapping)
                if op.output_spec is not None
                else None
            ),
            condition=(
                _remap_node_spec(op.condition, mapping)
                if op.condition is not None
                else None
            ),
        )
        new_order.append(canonical)

    new_output_node_map = {
        mapping[node_id]: output_name
        for node_id, output_name in graph.output_node_map.items()
    }
    new_output_paths = {
        mapping[node_id]: path for node_id, path in graph.output_paths.items()
    }
    new_dsl_to_runtime = {
        op_id: _dedupe_ordered([mapping.get(rid, rid) for rid in runtime_ids])
        for op_id, runtime_ids in graph.dsl_to_runtime.items()
    }

    return (
        RuntimeGraph(
            nodes=new_nodes,
            node_order=new_order,
            output_node_map=new_output_node_map,
            output_paths=new_output_paths,
            dsl_to_runtime=new_dsl_to_runtime,
        ),
        mapping,
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
    ) -> tuple[RuntimeGraph, dict[str, tuple[str, str]], dict[str, str]]:
        """Optimize and merge runtime graphs prior to scheduling.

        Returns the merged, deduplicated graph, the output mapping, and the
        dedup ``node_id -> canonical node_id`` mapping (identity for nodes
        that were not deduplicated)."""
        logical_ops = _logical_ops(graphs)
        merged_graph, output_mapping = merge_runtime_graphs(graphs)
        optimized_graph, dedup_mapping = _dedupe_runtime_graph(
            merged_graph, logical_ops
        )
        return optimized_graph, output_mapping, dedup_mapping
