"""Cross-job result cache for standalone ``LambdaOp`` python tasks.

An opt-in ``cache: true`` LambdaOp (with a pinned image) stores each row's
output under a Merkle-style lineage key. A later job that computes the same
key reuses the stored result instead of re-running the simulation. Row-mode
ops cache per row; list/aligned ops cache one whole-task entry per key.

Entries live under the archive's ``cache/`` namespace via
``JobStorage.save_artifact``/``get_artifact`` (``job_storage.py``), so cache
growth never bloats the per-job ``record.json`` or ``jobs_index.json``.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from lumilake_server.runtime import python_step
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.utils.job_storage import JobStorage

# Bump to invalidate every previously stored entry when the materializer or
# the key's meaning changes.
CACHE_VERSION = 1

# The archive namespace (used as the ``job_id`` argument to save/get_artifact)
# that cache entries live under.
CACHE_NAMESPACE = "cache"

# A cached result larger than this is not stored (the run still succeeds).
CACHE_MAX_BYTES = 8 * 1024 * 1024


def is_cacheable_node(node: Any) -> bool:
    """Return whether a runtime node is a cacheable standalone python task.

    Only ``python`` tasks that opted in with ``cache: true`` are cacheable.
    """
    return (
        node is not None
        and node.task_type == python_step.TASK_TYPE
        and bool(node.data_spec.get("cache"))
    )


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _row_key(
    node: Any,
    row_index: int,
    literal_values: list[list[Any]],
    upstream_keys: list[str],
) -> str:
    """The per-row lineage key for one row of a cacheable row-mode node.

    Excludes ``timeout_s``/``memory_mb`` (they don't change the result) and
    includes the op spec, this row's resolved literal values, and the per-row
    keys of cacheable upstream rows.
    """
    spec = {
        "code": node.data_spec["code"],
        "fn_name": node.data_spec["fn_name"],
        "mode": node.data_spec.get("mode", "row"),
        "image": node.data_spec.get("image"),
        "row_values": [col[row_index] for col in literal_values],
        "upstream_keys": upstream_keys,
        "cache_version": CACHE_VERSION,
    }
    return hashlib.sha256(_canonical(spec).encode("utf-8")).hexdigest()


def _task_key(
    node: Any, literal_values: list[list[Any]], upstream_keys: list[str]
) -> str:
    """The whole-task lineage key for a cacheable list/aligned-mode node."""
    spec = {
        "code": node.data_spec["code"],
        "fn_name": node.data_spec["fn_name"],
        "mode": node.data_spec.get("mode", "row"),
        "image": node.data_spec.get("image"),
        "literal_values": literal_values,
        "upstream_keys": upstream_keys,
        "cache_version": CACHE_VERSION,
    }
    return hashlib.sha256(_canonical(spec).encode("utf-8")).hexdigest()


def _plan_columns(node: Any) -> tuple[list[list[Any]], list[str]]:
    """Split a node's plan into literal columns and upstream node ids.

    Returns ``(literal_values, upstream_node_ids)``. Literal columns are the
    ``{"kind": "literal", "values": [...]}`` entries; stage entries contribute
    their upstream node id to ``upstream_node_ids``.
    """
    literal_values: list[list[Any]] = []
    upstream_node_ids: list[str] = []
    for entry in node.data_spec["plan"]:
        if entry["kind"] == "literal":
            literal_values.append(entry["values"])
        elif entry["kind"] == "stage":
            upstream_node_ids.append(entry["node"])
    return literal_values, upstream_node_ids


def _row_count(node: Any, literal_values: list[list[Any]]) -> int:
    """The number of rows a row-mode node processes (max literal length)."""
    return max((len(col) for col in literal_values), default=1)


def compute_keys(
    graph: RuntimeGraph, node_id: str
) -> tuple[list[str] | None, str | None]:
    """Compute the cache key(s) for a cacheable node.

    Returns ``(row_keys, task_key)``: for row mode, ``row_keys`` is one key per
    row and ``task_key`` is ``None``; for list/aligned mode, ``row_keys`` is
    ``None`` and ``task_key`` is the single whole-task key. Returns
    ``(None, None)`` if the node is not cacheable or any upstream is not
    cacheable (fail closed).
    """
    node = graph.nodes.get(node_id)
    if not is_cacheable_node(node):
        return None, None
    assert node is not None
    literal_values, upstream_node_ids = _plan_columns(node)
    # Fail closed: every stage upstream must itself be a cacheable python task.
    upstream_keys: list[str] = []
    for upstream_id in upstream_node_ids:
        upstream = graph.nodes.get(upstream_id)
        if not is_cacheable_node(upstream):
            return None, None
        upstream_row_keys, upstream_task_key = compute_keys(graph, upstream_id)
        if upstream_task_key is not None:
            upstream_keys.append(upstream_task_key)
        elif upstream_row_keys is not None:
            upstream_keys.extend(upstream_row_keys)
        else:
            return None, None
    mode = node.data_spec.get("mode", "row")
    if mode == "row":
        n = _row_count(node, literal_values)
        row_keys = [_row_key(node, i, literal_values, upstream_keys) for i in range(n)]
        return row_keys, None
    task_key = _task_key(node, literal_values, upstream_keys)
    return None, task_key


def _object_name(key: str) -> str:
    return f"{key}.json"


@dataclass
class CachePlan:
    """The rewrite decided for one cacheable node.

    ``emit`` replays a fully-cached result without running the computation;
    ``merge`` runs a partial task on the missed rows and splices the cached
    rows back in; ``run`` submits the node as normal (all-miss or not
    cacheable).
    """

    kind: str  # "emit" | "merge" | "run"
    # For "emit": the cached items to replay.
    items: list[Any] | None = None
    # For "merge": the cached value per row (None marks a row the partial task
    # runs) and the id of the generated partial node.
    cached: list[Any] | None = None
    partial_node: str | None = None
    # The row indices the partial task runs (missed rows).
    missed_rows: list[int] | None = None
    # Telemetry: rows hit and missed, and the producer job ids of the hits.
    rows_hit: int = 0
    rows_missed: int = 0
    producer_job_ids: set[str] = field(default_factory=set)


def plan_node(
    graph: RuntimeGraph,
    node_id: str,
    entries: list[dict[str, Any] | None],
) -> CachePlan:
    """Decide the cache rewrite for one node from its loaded entries.

    ``entries`` is one entry per row (row mode) or a single entry (list/aligned
    mode), in key order; ``None`` marks a miss. Returns a :class:`CachePlan`
    describing how to rewrite the node.
    """
    node = graph.nodes.get(node_id)
    if not is_cacheable_node(node):
        return CachePlan(kind="run")
    assert node is not None
    mode = node.data_spec.get("mode", "row")
    if mode == "row":
        return _plan_row_node(graph, node_id, entries)
    return _plan_task_node(node_id, entries)


def _plan_row_node(
    graph: RuntimeGraph, node_id: str, entries: list[dict[str, Any] | None]
) -> CachePlan:
    hits = [e for e in entries if e is not None]
    if not hits:
        return CachePlan(kind="run", rows_missed=len(entries))
    cached = [e["items"][0]["output"] if e is not None else None for e in entries]
    producers = {e["producer_job_id"] for e in hits if e.get("producer_job_id")}
    if len(hits) == len(entries):
        return CachePlan(
            kind="emit",
            items=cached,
            rows_hit=len(entries),
            producer_job_ids=producers,
        )
    missed_rows = [i for i, e in enumerate(entries) if e is None]
    return CachePlan(
        kind="merge",
        cached=cached,
        partial_node=f"{node_id}__partial",
        missed_rows=missed_rows,
        rows_hit=len(hits),
        rows_missed=len(missed_rows),
        producer_job_ids=producers,
    )


def _plan_task_node(node_id: str, entries: list[dict[str, Any] | None]) -> CachePlan:
    entry = entries[0] if entries else None
    if entry is None:
        return CachePlan(kind="run", rows_missed=1)
    items = entry.get("items")
    if not isinstance(items, list):
        return CachePlan(kind="run", rows_missed=1)
    producers = {entry["producer_job_id"]} if entry.get("producer_job_id") else set()
    return CachePlan(
        kind="emit",
        items=items,
        rows_hit=1,
        producer_job_ids=producers,
    )


def partial_plan(node: Any, missed_rows: list[int]) -> list[dict[str, Any]]:
    """The plan for a partial task: every literal column filtered to the
    missed rows; stage columns are left as-is."""
    plan: list[dict[str, Any]] = []
    for entry in node.data_spec["plan"]:
        if entry["kind"] == "literal":
            values = entry["values"]
            plan.append({"kind": "literal", "values": [values[i] for i in missed_rows]})
        else:
            plan.append(dict(entry))
    return plan


def emit_data_spec(node: Any, items: list[Any]) -> dict[str, Any]:
    """The data spec for an emit step that replays a fully-cached result.

    The step only returns cached items — it runs no user code — so it needs no
    image and runs in the default image.
    """
    return {
        "mode": "emit",
        "items": items,
        "plan": [],
    }


def merge_data_spec(node: Any, cached: list[Any], partial_node: str) -> dict[str, Any]:
    """The data spec for a merge step that splices a partial task's fresh rows
    into a cached list, restoring the original row order."""
    return {
        "mode": "merge",
        "cached": cached,
        "partial_node": partial_node,
        "plan": [{"kind": "stage", "node": partial_node}],
    }


def partial_data_spec(node: Any, missed_rows: list[int]) -> dict[str, Any]:
    """The data spec for a partial task running the computation on the missed
    rows only."""
    spec = {
        "code": node.data_spec["code"],
        "fn_name": node.data_spec["fn_name"],
        "mode": node.data_spec.get("mode", "row"),
        "plan": partial_plan(node, missed_rows),
        "timeout_s": node.data_spec.get("timeout_s"),
        "memory_mb": node.data_spec.get("memory_mb"),
    }
    if node.data_spec.get("image"):
        spec["image"] = node.data_spec["image"]
    return spec


def load_entry(job_storage: JobStorage, key: str) -> dict[str, Any] | None:
    """Load a cached entry, or ``None`` on a miss."""
    try:
        data, _ = job_storage.get_artifact(CACHE_NAMESPACE, _object_name(key))
    except KeyError:
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def store_entry(
    job_storage: JobStorage, key: str, items: list[Any], producer_job_id: str
) -> None:
    """Store a cached entry under ``key``, skipping oversized results."""
    body = json.dumps(
        {"items": items, "producer_job_id": producer_job_id},
        ensure_ascii=False,
    ).encode("utf-8")
    if len(body) > CACHE_MAX_BYTES:
        return
    job_storage.save_artifact(
        CACHE_NAMESPACE, _object_name(key), body, "application/json"
    )
