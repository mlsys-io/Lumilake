"""Cache rewrite and store methods on the FlowMesh runtime manager: all-hit /
partial-hit / all-miss graph rewrites, stores after success and none after
failure, and telemetry."""

from types import SimpleNamespace
from typing import Any

import pytest

from lumilake import envs
from lumilake_server.runtime import cache, python_step
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.request import RequestInfo
from lumilake_server.runtime.runtime_graph import RuntimeGraph, RuntimeOp
from lumilake_server.runtime.runtime_manager.flowmesh import (
    CacheStore,
    FlowmeshRuntimeManager,
)
from lumilake_server.utils.job_storage import InMemoryJobStorage

_IMAGE = "ghcr.io/example/sim@sha256:" + "a" * 64


def _op(
    node_id: str,
    *,
    plan: list[dict[str, Any]] | None = None,
    mode: str = "row",
    dependencies: tuple[str, ...] = (),
) -> RuntimeOp:
    data_spec: dict[str, Any] = {
        "code": "def f(row):\n    return row",
        "fn_name": "f",
        "plan": plan if plan is not None else [{"kind": "literal", "values": ["a"]}],
        "mode": mode,
        "timeout_s": 600,
        "image": _IMAGE,
        "cache": True,
    }
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


def _request_info(graph: RuntimeGraph, request_id: str = "req-1") -> RequestInfo:
    info = RequestInfo(
        request_id=request_id,
        runtime_graphs={"g": graph},
        data_profile_graphs={},
        member_request_ids={"req-1"},
    )
    info.batch_id = "batch-1"
    info.runtime_graph = graph
    info.data_profile_graph = RuntimeGraph(nodes={}, node_order=[], output_node_map={})
    return info


def _schedule(node_ids: list[str]) -> Schedule:
    return Schedule(worker_assignment={"w1": list(node_ids)})


@pytest.fixture(autouse=True)
def _job_storage(monkeypatch: pytest.MonkeyPatch) -> InMemoryJobStorage:
    storage = InMemoryJobStorage()
    import lumilake_server.utils.job_storage as js

    monkeypatch.setattr(js, "_job_storage", storage)
    return storage


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")


def _store(storage: InMemoryJobStorage, key: str, value: str, producer: str) -> None:
    cache.store_entry(storage, key, [{"output": value}], producer)


async def _async_dict(value: dict[str, Any]) -> dict[str, Any]:
    return value


def _row_keys(graph: RuntimeGraph, node_id: str) -> list[str]:
    row_keys, task_key = cache.compute_keys(graph, node_id)
    assert row_keys is not None
    assert task_key is None
    return row_keys


def _entry(storage: InMemoryJobStorage, key: str) -> dict[str, Any]:
    entry = cache.load_entry(storage, key)
    assert entry is not None
    return entry


@pytest.mark.asyncio
async def test_all_hit_rewrites_to_emit(_job_storage) -> None:
    node = _op("n", plan=[{"kind": "literal", "values": ["a", "b"]}])
    graph = _graph(node)
    storage = _job_storage
    keys = _row_keys(graph, "n")
    for i, key in enumerate(keys):
        _store(storage, key, f"out{i}", "job-0")

    manager = FlowmeshRuntimeManager()
    telemetry, store_plan = await manager._apply_cache_rewrites(
        _request_info(graph), _schedule(["n"])
    )
    assert graph.nodes["n"].data_spec["mode"] == "emit"
    assert graph.nodes["n"].data_spec["items"] == ["out0", "out1"]
    assert graph.nodes["n"].dependencies == ()
    assert store_plan == []
    assert telemetry["n"] == {
        "rows_hit": 2,
        "rows_missed": 0,
        "producer_job_ids": ["job-0"],
    }


@pytest.mark.asyncio
async def test_all_miss_runs_and_plans_store() -> None:
    node = _op("n", plan=[{"kind": "literal", "values": ["a", "b"]}])
    graph = _graph(node)
    manager = FlowmeshRuntimeManager()
    telemetry, store_plan = await manager._apply_cache_rewrites(
        _request_info(graph), _schedule(["n"])
    )
    assert graph.nodes["n"].data_spec["mode"] == "row"
    assert store_plan == [
        CacheStore(node_id="n", keys=store_plan[0].keys, whole_task=False)
    ]
    assert len(store_plan[0].keys) == 2
    assert telemetry["n"]["rows_missed"] == 2


@pytest.mark.asyncio
async def test_partial_hit_rewrites_to_merge_with_partial_node(_job_storage) -> None:
    node = _op("n", plan=[{"kind": "literal", "values": ["a", "b", "c"]}])
    graph = _graph(node)
    storage = _job_storage
    keys = _row_keys(graph, "n")
    _store(storage, keys[1], "cached-b", "job-0")

    manager = FlowmeshRuntimeManager()
    schedule = _schedule(["n"])
    telemetry, store_plan = await manager._apply_cache_rewrites(
        _request_info(graph), schedule
    )
    # Original node becomes a merge step.
    assert graph.nodes["n"].data_spec["mode"] == "merge"
    assert graph.nodes["n"].data_spec["cached"] == [None, "cached-b", None]
    assert graph.nodes["n"].data_spec["partial_node"] == "n__partial"
    assert graph.nodes["n"].dependencies == ("n__partial",)
    # Partial node runs the missed rows only.
    partial = graph.nodes["n__partial"]
    assert partial.data_spec["mode"] == "row"
    assert partial.data_spec["plan"] == [{"kind": "literal", "values": ["a", "c"]}]
    assert "n__partial" in graph.node_order
    # Schedule covers the partial node on the same worker.
    assert "n__partial" in schedule.worker_assignment["w1"]
    # Store plan caches the partial task's fresh rows under the missed keys.
    assert store_plan == [
        CacheStore(
            node_id="n__partial",
            keys=[keys[0], keys[2]],
            whole_task=False,
        )
    ]
    assert telemetry["n"] == {
        "rows_hit": 1,
        "rows_missed": 2,
        "producer_job_ids": ["job-0"],
    }


@pytest.mark.asyncio
async def test_list_mode_whole_task_key(_job_storage) -> None:
    node = _op("n", mode="list", plan=[{"kind": "literal", "values": ["a", "b"]}])
    graph = _graph(node)
    storage = _job_storage
    _, task_key = cache.compute_keys(graph, "n")
    assert task_key is not None
    _store(storage, task_key, "whole", "job-0")

    manager = FlowmeshRuntimeManager()
    telemetry, store_plan = await manager._apply_cache_rewrites(
        _request_info(graph), _schedule(["n"])
    )
    assert graph.nodes["n"].data_spec["mode"] == "emit"
    assert graph.nodes["n"].data_spec["items"] == [{"output": "whole"}]
    assert store_plan == []
    assert telemetry["n"]["rows_hit"] == 1


@pytest.mark.asyncio
async def test_fail_closed_upstream_not_cached() -> None:
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
    graph = _graph(llm, node)
    manager = FlowmeshRuntimeManager()
    telemetry, store_plan = await manager._apply_cache_rewrites(
        _request_info(graph), _schedule(["llm", "n"])
    )
    assert telemetry == {}
    assert store_plan == []
    assert graph.nodes["n"].data_spec["mode"] == "row"


@pytest.mark.asyncio
async def test_store_after_success_and_none_after_failure(
    _job_storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = _op("n", plan=[{"kind": "literal", "values": ["a", "b"]}])
    graph = _graph(node)
    storage = _job_storage
    keys = _row_keys(graph, "n")
    manager = FlowmeshRuntimeManager()
    _, store_plan = await manager._apply_cache_rewrites(
        _request_info(graph), _schedule(["n"])
    )

    # Task succeeds: results are stored.
    manager._execution_task_status[("req-1", "batch-1")] = {"t1": "DONE"}
    monkeypatch.setattr(
        FlowmeshRuntimeManager,
        "fm",
        property(
            lambda self: SimpleNamespace(
                results=SimpleNamespace(
                    retrieve=lambda tid: _async_dict(
                        {"items": [{"output": "o1"}, {"output": "o2"}]}
                    )
                )
            )
        ),
    )
    await manager._store_cache_results(_request_info(graph), store_plan, {"n": "t1"})
    assert _entry(storage, keys[0])["items"] == [{"output": "o1"}]
    assert _entry(storage, keys[1])["items"] == [{"output": "o2"}]

    # A failed task stores nothing.
    manager._execution_task_status[("req-1", "batch-1")] = {"t1": "FAILED"}
    await manager._store_cache_results(_request_info(graph), store_plan, {"n": "t1"})
    # Entries remain from the successful run (not overwritten by failure).
    assert _entry(storage, keys[0])["items"] == [{"output": "o1"}]
