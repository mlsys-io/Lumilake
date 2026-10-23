"""Cross-job LambdaOp result cache: key stability, pinned-image validation,
row/list granularity, partial-hit merge rewrites, and telemetry."""

from typing import Any

import pytest

from lumilake_server.ops import LambdaOp
from lumilake_server.ops.util_ops import is_pinned_image
from lumilake_server.runtime import cache, python_step
from lumilake_server.runtime.runtime_graph import RuntimeGraph, RuntimeOp
from lumilake_server.utils.job_storage import InMemoryJobStorage

_IMAGE = "ghcr.io/example/sim@sha256:" + "a" * 64
_IMAGE_ID = "sha256:" + "b" * 64


def _op(
    node_id: str,
    *,
    code: str = "def f(row):\n    return row",
    fn_name: str = "f",
    mode: str = "row",
    image: str = _IMAGE,
    cache_flag: bool = True,
    timeout_s: int = 600,
    memory_mb: int | None = None,
    plan: list[dict[str, Any]] | None = None,
    dependencies: tuple[str, ...] = (),
) -> RuntimeOp:
    data_spec: dict[str, Any] = {
        "code": code,
        "fn_name": fn_name,
        "plan": plan if plan is not None else [{"kind": "literal", "values": ["a"]}],
        "mode": mode,
        "timeout_s": timeout_s,
    }
    if memory_mb is not None:
        data_spec["memory_mb"] = memory_mb
    if image is not None:
        data_spec["image"] = image
    if cache_flag:
        data_spec["cache"] = True
    return RuntimeOp(
        node_id=node_id,
        task_type=python_step.TASK_TYPE,
        backend=python_step.BACKEND,
        model="",
        data_spec=data_spec,
        model_spec={},
        inference_spec={},
        dependencies=dependencies,
    )


def _graph(*nodes: RuntimeOp) -> RuntimeGraph:
    return RuntimeGraph(
        nodes={n.node_id: n for n in nodes},
        node_order=[n.node_id for n in nodes],
        output_node_map={},
    )


def _store(
    job_storage: InMemoryJobStorage, key: str, items: list[Any], producer: str
) -> None:
    cache.store_entry(job_storage, key, [{"output": v} for v in items], producer)


def _load(job_storage: InMemoryJobStorage, key: str) -> dict[str, Any] | None:
    return cache.load_entry(job_storage, key)


def _row_keys(graph: RuntimeGraph, node_id: str) -> list[str]:
    row_keys, task_key = cache.compute_keys(graph, node_id)
    assert row_keys is not None
    assert task_key is None
    return row_keys


# --- pinned-image validation -------------------------------------------------


def test_is_pinned_image() -> None:
    assert is_pinned_image("name@sha256:" + "a" * 64)
    assert is_pinned_image("sha256:" + "b" * 64)
    assert not is_pinned_image("name:latest")
    assert not is_pinned_image("name")
    assert not is_pinned_image("sha256:short")
    assert not is_pinned_image("name@sha256:short")
    # The digest must be lowercase hex, not just 64 arbitrary characters.
    assert not is_pinned_image("name@sha256:" + "g" * 64)
    assert not is_pinned_image("sha256:" + "G" * 64)
    assert not is_pinned_image("name@sha256:" + "A" * 64)


def test_lambda_op_cache_requires_pinned_image() -> None:
    with pytest.raises(ValueError, match="requires an image"):
        LambdaOp([], fn=lambda row: str(row), cache=True)
    with pytest.raises(ValueError, match="requires a pinned image"):
        LambdaOp([], fn=lambda row: str(row), image="name:latest", cache=True)
    # Pinned references are accepted.
    LambdaOp([], fn=lambda row: str(row), image=_IMAGE, cache=True)
    LambdaOp([], fn=lambda row: str(row), image=_IMAGE_ID, cache=True)


# --- key stability -----------------------------------------------------------


def test_row_key_ignores_limits_and_changes_with_inputs() -> None:
    node = _op("n", timeout_s=600, memory_mb=512)
    graph = _graph(node)
    row_keys = _row_keys(graph, "n")
    assert len(row_keys) == 1
    key = row_keys[0]

    # Changing timeout/memory does not change the key.
    node2 = _op("n", timeout_s=1200, memory_mb=1024)
    row_keys2 = _row_keys(_graph(node2), "n")
    assert row_keys2 == [key]

    # Changing the code, image, or row value changes the key.
    for changed in (
        _op("n", code="def f(row):\n    return row + 'x'"),
        _op("n", image=_IMAGE_ID),
        _op(
            "n",
            plan=[{"kind": "literal", "values": ["b"]}],
        ),
    ):
        row_keys_c = _row_keys(_graph(changed), "n")
        assert row_keys_c != [key]


def test_row_key_per_row() -> None:
    node = _op(
        "n",
        plan=[{"kind": "literal", "values": ["a", "b", "c"]}],
    )
    row_keys = _row_keys(_graph(node), "n")
    assert len(row_keys) == 3
    assert len(set(row_keys)) == 3


def test_row_key_broadcasts_one_value_column() -> None:
    # A one-value literal column broadcasts to every row (matching the row-mode
    # wrapper); it must not raise IndexError on later rows.
    node = _op(
        "n",
        plan=[
            {"kind": "literal", "values": ["a", "b", "c"]},
            {"kind": "literal", "values": ["const"]},
        ],
    )
    row_keys = _row_keys(_graph(node), "n")
    assert len(row_keys) == 3
    assert len(set(row_keys)) == 3


def test_row_key_uses_aligned_upstream_row() -> None:
    # A row-mode upstream contributes its row i key to row i, so adding an
    # unrelated upstream row does not change the other rows' keys.
    up = _op("up", plan=[{"kind": "literal", "values": ["a", "b"]}])
    down = _op(
        "down",
        plan=[{"kind": "stage", "node": "up"}],
        dependencies=("up",),
    )
    graph = _graph(up, down)
    down_keys = _row_keys(graph, "down")
    assert len(down_keys) == 2

    up2 = _op("up", plan=[{"kind": "literal", "values": ["a", "b", "c"]}])
    down2 = _op(
        "down",
        plan=[{"kind": "stage", "node": "up"}],
        dependencies=("up",),
    )
    graph2 = _graph(up2, down2)
    down_keys2 = _row_keys(graph2, "down")
    # Row 0 and row 1 keys are unchanged; a third row was added.
    assert down_keys2[:2] == down_keys
    assert len(down_keys2) == 3


def test_upstream_key_changes_row_key() -> None:
    up = _op("up", plan=[{"kind": "literal", "values": ["a"]}])
    down = _op(
        "down",
        plan=[{"kind": "stage", "node": "up"}],
        dependencies=("up",),
    )
    graph = _graph(up, down)
    down_row_keys = _row_keys(graph, "down")
    up_row_keys = _row_keys(graph, "up")
    assert down_row_keys != up_row_keys


def test_list_mode_whole_task_key() -> None:
    node = _op(
        "n",
        mode="list",
        plan=[{"kind": "literal", "values": ["a", "b"]}],
    )
    row_keys, task_key = cache.compute_keys(_graph(node), "n")
    assert row_keys is None
    assert task_key is not None


# --- fail-closed upstreams ---------------------------------------------------


def test_non_cacheable_upstream_fails_closed() -> None:
    # An LLM upstream (not a python task) makes the node not cacheable.
    llm = RuntimeOp(
        node_id="llm",
        task_type="inference",
        backend="vllm",
        model="m",
        data_spec={},
        model_spec={},
        inference_spec={},
    )
    node = _op(
        "n",
        plan=[{"kind": "stage", "node": "llm"}],
        dependencies=("llm",),
    )
    row_keys, task_key = cache.compute_keys(_graph(llm, node), "n")
    assert row_keys is None and task_key is None


def test_cacheable_upstream_keeps_node_cacheable() -> None:
    up = _op("up", plan=[{"kind": "literal", "values": ["a"]}])
    down = _op(
        "down",
        plan=[{"kind": "stage", "node": "up"}],
        dependencies=("up",),
    )
    row_keys, _ = cache.compute_keys(_graph(up, down), "down")
    assert row_keys is not None


# --- plan rewrites -----------------------------------------------------------


def test_plan_all_hit_emits() -> None:
    node = _op("n", plan=[{"kind": "literal", "values": ["a", "b"]}])
    graph = _graph(node)
    keys = _row_keys(graph, "n")
    storage = InMemoryJobStorage()
    for i, key in enumerate(keys):
        _store(storage, key, [f"out{i}"], "job-1")
    entries = [cache.load_entry(storage, k) for k in keys]
    plan = cache.plan_node(graph, "n", entries)
    assert plan.kind == "emit"
    assert plan.items == ["out0", "out1"]
    assert plan.rows_hit == 2 and plan.rows_missed == 0
    assert plan.producer_job_ids == {"job-1"}


def test_plan_all_miss_runs() -> None:
    node = _op("n", plan=[{"kind": "literal", "values": ["a", "b"]}])
    graph = _graph(node)
    keys = _row_keys(graph, "n")
    storage = InMemoryJobStorage()
    entries = [cache.load_entry(storage, k) for k in keys]
    plan = cache.plan_node(graph, "n", entries)
    assert plan.kind == "run"
    assert plan.rows_missed == 2


def test_plan_partial_hit_merges() -> None:
    node = _op("n", plan=[{"kind": "literal", "values": ["a", "b", "c"]}])
    graph = _graph(node)
    keys = _row_keys(graph, "n")
    storage = InMemoryJobStorage()
    _store(storage, keys[1], ["cached-b"], "job-1")
    entries = [cache.load_entry(storage, k) for k in keys]
    plan = cache.plan_node(graph, "n", entries)
    assert plan.kind == "merge"
    assert plan.missed_rows == [0, 2]
    assert plan.cached == [None, "cached-b", None]
    assert plan.partial_node == "n__partial"
    assert plan.rows_hit == 1 and plan.rows_missed == 2


def test_plan_partial_hit_with_stage_column_runs_all() -> None:
    # A partial rewrite cannot filter staged upstream columns, so a row node
    # with a stage input and a partial hit runs every row normally (all-miss)
    # instead of merging.
    up = _op("up", plan=[{"kind": "literal", "values": ["a", "b", "c"]}])
    node = _op(
        "n",
        plan=[{"kind": "stage", "node": "up"}],
        dependencies=("up",),
    )
    graph = _graph(up, node)
    keys = _row_keys(graph, "n")
    storage = InMemoryJobStorage()
    _store(storage, keys[1], ["cached-b"], "job-1")
    entries = [cache.load_entry(storage, k) for k in keys]
    plan = cache.plan_node(graph, "n", entries)
    assert plan.kind == "run"
    assert plan.rows_missed == 3


def test_partial_plan_filters_literal_columns() -> None:
    node = _op(
        "n",
        plan=[
            {"kind": "literal", "values": ["a", "b", "c"]},
            {"kind": "literal", "values": ["x", "y", "z"]},
        ],
    )
    filtered = cache.partial_plan(node, [0, 2])
    assert filtered == [
        {"kind": "literal", "values": ["a", "c"]},
        {"kind": "literal", "values": ["x", "z"]},
    ]


def test_emit_data_spec_has_no_image() -> None:
    node = _op("n")
    spec = cache.emit_data_spec(node, ["out"])
    assert spec["mode"] == "emit"
    assert spec["items"] == ["out"]
    assert "image" not in spec


def test_merge_data_spec() -> None:
    node = _op("n")
    spec = cache.merge_data_spec(node, [None, "cached"], "n__partial")
    assert spec["mode"] == "merge"
    assert spec["cached"] == [None, "cached"]
    assert spec["plan"] == [{"kind": "stage", "node": "n__partial"}]


def test_partial_data_spec_keeps_image() -> None:
    node = _op("n")
    spec = cache.partial_data_spec(node, [0])
    assert spec["image"] == _IMAGE
    assert spec["plan"] == [{"kind": "literal", "values": ["a"]}]


# --- store / load ------------------------------------------------------------


def test_store_then_load_roundtrip() -> None:
    storage = InMemoryJobStorage()
    node = _op("n")
    keys = _row_keys(_graph(node), "n")
    _store(storage, keys[0], ["out"], "job-9")
    entry = _load(storage, keys[0])
    assert entry == {"items": [{"output": "out"}], "producer_job_id": "job-9"}
    assert _load(storage, "missing") is None


def test_store_skips_oversized() -> None:
    storage = InMemoryJobStorage()
    big = ["x" * (cache.CACHE_MAX_BYTES + 1)]
    cache.store_entry(storage, "k", big, "job-1")
    assert _load(storage, "k") is None


def test_emit_source_replays_items() -> None:
    src = python_step.emit_source("n", ["a", "b"])
    assert "_ITEMS" in src
    assert "return" in src


def test_merge_source_splices_in_order() -> None:
    src = python_step.merge_source("n", [None, "cached", None], "n__partial")
    assert "_CACHED" in src
    assert "_PARTIAL_NODE" in src
