"""Affinity-based clustering utilities for batch selection.

This module clusters workflow runtime graphs using agglomerative hierarchical
clustering (average linkage). Pairwise distance is derived from model, data,
and size affinities. The selector then chooses a batch-sized subset that
minimizes average intra-batch distance, with an enqueue-time tie-breaker.
"""

import re
from dataclasses import dataclass
from typing import Any

from lumilake.runtime.runtime_graph import RuntimeGraph
from lumilake.runtime.runtime_ops import RuntimeOp

MODEL_WEIGHT = 0.7
DATA_WEIGHT = 0.2
SIZE_WEIGHT = 0.1
# The weights above correspond to alpha/beta/gamma in the distance definition,
# with model affinity dominating data and size affinities.


@dataclass(frozen=True)
class WorkflowAffinity:
    """Lightweight feature bundle used by the distance calculator."""

    models: set[str]
    system_prompt_tokens: set[str]
    node_count: int


def select_affinity_batch_ids(
    graphs: dict[str, RuntimeGraph],
    enqueued_at: dict[str, float],
    batch_size: int,
    *,
    pinned_ids: list[str] | None = None,
    model_weight: float = MODEL_WEIGHT,
    data_weight: float = DATA_WEIGHT,
    size_weight: float = SIZE_WEIGHT,
) -> list[str]:
    """Select workflow IDs by affinity using agglomerative clustering.

    Distance between workflows i and j is:
        D = 1 - (alpha*S_model + beta*S_data + gamma*S_size)
    where:
        S_model: Jaccard of base model sets.
        S_data: Jaccard of tokenized system prompts.
        S_size: 1 - |N_i - N_j| / max(N_i, N_j).

    The function merges clusters by average linkage and, for any cluster with
    size >= batch_size, chooses the subset with minimal average distance.
    Enqueue time is used as a tie-breaker (older first). When ``pinned_ids``
    are provided, the selector always includes those IDs (up to batch size)
    and fills remaining slots by affinity.
    """
    if batch_size <= 0:
        batch_size = 1

    candidate_ids = list(graphs.keys())
    if len(candidate_ids) <= batch_size:
        return candidate_ids
    candidate_set = set(candidate_ids)
    normalized_pinned: list[str] = []
    if pinned_ids:
        seen: set[str] = set()
        for workflow_id in pinned_ids:
            if workflow_id not in candidate_set or workflow_id in seen:
                continue
            seen.add(workflow_id)
            normalized_pinned.append(workflow_id)
    if len(normalized_pinned) >= batch_size:
        return sorted(
            normalized_pinned,
            key=lambda workflow_id: enqueued_at.get(workflow_id, float("inf")),
        )[:batch_size]

    affinity = {
        workflow_id: _build_affinity(graphs[workflow_id])
        for workflow_id in candidate_ids
    }
    distance_cache: dict[tuple[str, str], float] = {}

    def distance(first: str, second: str) -> float:
        if first == second:
            return 0.0
        key = (first, second) if first < second else (second, first)
        cached = distance_cache.get(key)
        if cached is not None:
            return cached
        first_aff = affinity[first]
        second_aff = affinity[second]
        score = (
            model_weight * _jaccard(first_aff.models, second_aff.models)
            + data_weight
            * _jaccard(first_aff.system_prompt_tokens, second_aff.system_prompt_tokens)
            + size_weight * _size_affinity(first_aff, second_aff)
        )
        distance_value = 1.0 - max(0.0, min(1.0, score))
        distance_cache[key] = distance_value
        return distance_value

    def avg_distance(candidate_id: str, selected_ids: list[str]) -> float:
        if not selected_ids:
            return 1.0
        total = 0.0
        for selected_id in selected_ids:
            total += distance(candidate_id, selected_id)
        return total / len(selected_ids)

    def batch_score(selected_ids: list[str]) -> float:
        if len(selected_ids) <= 1:
            return 0.0
        total = 0.0
        count = 0
        for idx, first_id in enumerate(selected_ids):
            for second_id in selected_ids[idx + 1 :]:
                total += distance(first_id, second_id)
                count += 1
        return total / count if count else 0.0

    def select_subset(candidate_ids: list[str], pinned: list[str]) -> list[str]:
        if len(candidate_ids) <= batch_size:
            return list(candidate_ids)
        pinned_set = set(pinned)
        if len(pinned) >= batch_size:
            return sorted(
                pinned,
                key=lambda workflow_id: enqueued_at.get(workflow_id, float("inf")),
            )[:batch_size]
        if pinned:
            selected = list(pinned)
            remaining = [cid for cid in candidate_ids if cid not in pinned_set]
            while len(selected) < batch_size and remaining:
                next_id = min(
                    remaining,
                    key=lambda cid: (avg_distance(cid, selected), enqueued_at[cid]),
                )
                selected.append(next_id)
                remaining.remove(next_id)
            return selected

        best_subset: list[str] | None = None
        best_score: float | None = None
        best_oldest: float | None = None

        for seed_id in candidate_ids:
            selected = [seed_id]
            remaining = [cid for cid in candidate_ids if cid != seed_id]
            while len(selected) < batch_size and remaining:
                next_id = min(
                    remaining,
                    key=lambda cid: (avg_distance(cid, selected), enqueued_at[cid]),
                )
                selected.append(next_id)
                remaining.remove(next_id)

            score = batch_score(selected)
            oldest = min(enqueued_at[cid] for cid in selected)
            if (
                best_score is None
                or score < best_score
                or (score == best_score and oldest < (best_oldest or float("inf")))
            ):
                best_score = score
                best_oldest = oldest
                best_subset = selected

        if best_subset is None:
            return candidate_ids[:batch_size]
        return best_subset

    def cluster_distance(cluster_a: list[str], cluster_b: list[str]) -> float:
        total = 0.0
        count = 0
        for first_id in cluster_a:
            for second_id in cluster_b:
                total += distance(first_id, second_id)
                count += 1
        return total / count if count else 0.0

    best_ids: list[str] | None = None
    best_score: float | None = None
    best_oldest: float | None = None

    def consider_cluster(cluster_ids: list[str]) -> None:
        nonlocal best_ids, best_score, best_oldest
        if any(workflow_id not in cluster_ids for workflow_id in normalized_pinned):
            return
        if len(cluster_ids) < batch_size:
            return
        subset = select_subset(cluster_ids, normalized_pinned)
        score = batch_score(subset)
        oldest = min(enqueued_at[cid] for cid in subset)
        if (
            best_score is None
            or score < best_score
            or (score == best_score and oldest < (best_oldest or float("inf")))
        ):
            best_score = score
            best_oldest = oldest
            best_ids = subset

    clusters: list[list[str]] = [[workflow_id] for workflow_id in candidate_ids]
    for cluster in clusters:
        consider_cluster(cluster)

    while len(clusters) > 1:
        best_pair: tuple[int, int] | None = None
        best_pair_distance = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = cluster_distance(clusters[i], clusters[j])
                if dist < best_pair_distance:
                    best_pair_distance = dist
                    best_pair = (i, j)

        if best_pair is None:
            break
        first_idx, second_idx = best_pair
        merged = clusters[first_idx] + clusters[second_idx]
        for idx in sorted((first_idx, second_idx), reverse=True):
            clusters.pop(idx)
        clusters.append(merged)
        consider_cluster(merged)

    if best_ids is None:
        return select_subset(candidate_ids, normalized_pinned)
    return best_ids


def _build_affinity(graph: RuntimeGraph) -> WorkflowAffinity:
    """Extract model, system prompt tokens, and size features from a graph."""
    models: set[str] = set()
    system_prompts: set[str] = set()
    for op in graph.nodes.values():
        if _is_llm_backend(op):
            if op.model:
                models.add(op.model)
        system_prompts.update(_extract_system_prompts(op.data_spec))
    tokens = _tokenize_prompts(system_prompts)
    return WorkflowAffinity(
        models=models,
        system_prompt_tokens=tokens,
        node_count=graph.node_count,
    )


def _is_llm_backend(op: RuntimeOp) -> bool:
    """Return True if the runtime op is associated with an LLM backend."""
    return op.backend in {"vllm", "transformers", "diffusers"}


def _extract_system_prompts(spec: Any) -> set[str]:
    """Collect system messages from nested message specs."""
    prompts: set[str] = set()
    if isinstance(spec, dict):
        messages = spec.get("messages")
        if isinstance(messages, list):
            prompts.update(_extract_system_prompts(messages))
        for key, value in spec.items():
            if key == "messages":
                continue
            prompts.update(_extract_system_prompts(value))
    elif isinstance(spec, list):
        for item in spec:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if role == "system" and isinstance(content, str):
                    prompts.add(content)
            else:
                prompts.update(_extract_system_prompts(item))
    return prompts


def _tokenize_prompts(prompts: set[str]) -> set[str]:
    """Tokenize prompts into lowercased word tokens for Jaccard overlap."""
    tokens: set[str] = set()
    for prompt in prompts:
        if not prompt:
            continue
        tokens.update(re.findall(r"[A-Za-z0-9_]+", prompt.lower()))
    return tokens


def _jaccard(left: set[str], right: set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _size_affinity(first: WorkflowAffinity, second: WorkflowAffinity) -> float:
    """Compute normalized size affinity based on node counts."""
    max_nodes = max(first.node_count, second.node_count)
    if max_nodes <= 0:
        return 1.0
    return 1.0 - abs(first.node_count - second.node_count) / max_nodes
