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

from lumilake.log import get_default_logger
from lumilake_server.runtime import python_step
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.utils.job_storage import JobStorage

logger = get_default_logger()

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
    includes the op spec, this row's resolved literal values, and the aligned
    upstream keys for this row. A one-value literal column broadcasts to every
    row, matching the row-mode wrapper in ``python_step``.
    """
    spec = {
        "code": node.data_spec["code"],
        "fn_name": node.data_spec["fn_name"],
        "mode": node.data_spec.get("mode", "row"),
        "image": node.data_spec.get("image"),
        "row_values": [
            col[0] if len(col) == 1 else col[row_index] for col in literal_values
        ],
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


def _aligned_upstream_keys(
    graph: RuntimeGraph,
    node: Any,
    upstream_node_ids: list[str],
    keys: dict[str, tuple[list[str] | None, str | None]],
    mode: str,
) -> tuple[list[list[str]], int] | None:
    """The upstream key(s) that align with this node's rows.

    A row-mode upstream contributes its row ``i`` key to row ``i``; a
    list/aligned upstream contributes its whole-task key to every row. Returns
    ``(per_row_upstream_keys, row_count)`` where ``per_row_upstream_keys[i]`` is
    the upstream key list for row ``i``, or ``None`` (fail closed) if an
    upstream is not cacheable, if a row-mode upstream's row count neither
    matches this node's nor is 1 (a broadcast), or if a row-mode node's row
    count cannot be known before the run (its rows come only from a list/aligned
    upstream).
    """
    literal_values, _ = _plan_columns(node)
    # A one-value literal column broadcasts to any row count, so it does not
    # determine n.
    n = max((len(col) for col in literal_values if len(col) != 1), default=0)
    upstream_keys: list[tuple[str | None, list[str] | None]] = []
    for upstream_id in upstream_node_ids:
        upstream = graph.nodes.get(upstream_id)
        if not is_cacheable_node(upstream):
            return None
        upstream_row_keys, upstream_task_key = keys.get(upstream_id, (None, None))
        if upstream_task_key is not None:
            upstream_keys.append((upstream_task_key, None))
        elif upstream_row_keys is not None:
            upstream_keys.append((None, upstream_row_keys))
        else:
            return None
    # The row count is the max over multi-value literal columns and row-mode
    # upstreams.
    row_counts = [len(rk) for _, rk in upstream_keys if rk is not None]
    if row_counts:
        n = max(n, max(row_counts))
    if n == 0:
        if mode == "row" and upstream_keys:
            # A row-mode node whose row count depends only on a list/aligned
            # upstream is unknowable before the run; fail closed.
            return None
        n = 1
    # Every literal column must broadcast (1) or match the row count.
    for col in literal_values:
        if len(col) not in (1, n):
            return None
    per_row: list[list[str]] = [[] for _ in range(n)]
    for upstream_key, upstream_row_keys in upstream_keys:
        if upstream_row_keys is None:
            assert upstream_key is not None
            for row in per_row:
                row.append(upstream_key)
        elif len(upstream_row_keys) == 1:
            for row in per_row:
                row.append(upstream_row_keys[0])
        elif len(upstream_row_keys) == n:
            for i, row in enumerate(per_row):
                row.append(upstream_row_keys[i])
        else:
            return None
    return per_row, n


def compute_all_keys(
    graph: RuntimeGraph,
) -> dict[str, tuple[list[str] | None, str | None]]:
    """Compute every cacheable node's key(s) on the unmodified graph.

    Runs in ``graph.topological_order()`` so an upstream's keys are known
    before its dependents, and memoizes the result. Returns a map of node id to
    ``(row_keys, task_key)``; a node that is not cacheable, or whose upstream is
    not cacheable, maps to ``(None, None)`` (fail closed).
    """
    keys: dict[str, tuple[list[str] | None, str | None]] = {}
    for node_id in graph.topological_order():
        node = graph.nodes.get(node_id)
        if not is_cacheable_node(node):
            keys[node_id] = (None, None)
            continue
        assert node is not None
        literal_values, upstream_node_ids = _plan_columns(node)
        mode = node.data_spec.get("mode", "row")
        aligned = _aligned_upstream_keys(graph, node, upstream_node_ids, keys, mode)
        if aligned is None:
            keys[node_id] = (None, None)
            continue
        per_row_upstream_keys, n = aligned
        if mode == "row":
            row_keys = [
                _row_key(node, i, literal_values, per_row_upstream_keys[i])
                for i in range(n)
            ]
            keys[node_id] = (row_keys, None)
        else:
            keys[node_id] = (
                None,
                _task_key(node, literal_values, per_row_upstream_keys[0]),
            )
    return keys


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
    return compute_all_keys(graph).get(node_id, (None, None))


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
    # A partial rewrite filters the literal columns to the missed rows but
    # leaves staged upstream columns at their full length, which the row-mode
    # wrapper rejects. Only merge when every plan column is literal; otherwise
    # run every row normally (all-miss) and store every row's key on success.
    node = graph.nodes[node_id]
    assert node is not None
    _, upstream_node_ids = _plan_columns(node)
    if upstream_node_ids:
        return CachePlan(kind="run", rows_missed=len(entries))
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
    # Unwrap the stored ``{"output": v}`` items to plain values so the emit
    # step replays the same shape as the original computation (row mode already
    # unwraps per-row entries).
    values = [
        i["output"] if isinstance(i, dict) and "output" in i else i for i in items
    ]
    producers = {entry["producer_job_id"]} if entry.get("producer_job_id") else set()
    return CachePlan(
        kind="emit",
        items=values,
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
    image and runs in the default image. It keeps the original mode so the
    output aggregation sees the same shape as an uncached run, and marks itself
    with ``cache_emit`` so the step compiler selects the emit source (and skips
    the timeout) rather than treating the mode as a user computation.
    """
    return {
        "mode": node.data_spec.get("mode", "row"),
        "cache_emit": True,
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
    """Load a cached entry, or ``None`` on a miss.

    A missing entry is a miss. A storage transport or HTTP error is also
    treated as a miss (the job still runs normally) but is logged, so a cache
    outage never fails a request.
    """
    try:
        data, _ = job_storage.get_artifact(CACHE_NAMESPACE, _object_name(key))
    except KeyError:
        return None
    except Exception:
        logger.warning("cache lookup failed for key %s; treating as a miss", key)
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
