"""Whole-op reuse: a reusing plan answers an op whose every record the store
already holds and dispatches the rest."""

import json
import logging
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from lumilake_server.data_specs import (
    ColumnPlanEntry,
    DataFrame,
    NodeRead,
    PythonData,
    parse_data_spec,
)
from lumilake_server.runtime import python_step
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.record_store import (
    RecordAddress,
    RecordEntry,
    RecordSource,
)
from lumilake_server.runtime.record_writer import RecordWriter, read_index
from lumilake_server.runtime.request import RequestInfo
from lumilake_server.runtime.reuse_planner import (
    ReusePlan,
    ReusePlanner,
    served_column,
    served_records,
)
from lumilake_server.runtime.runtime_manager import FlowmeshRuntimeManager
from lumilake_server.utils.job_storage import InMemoryJobStorage
from tests.runtime.runtime_manager.conftest import job_storage as job_storage
from tests.runtime.runtime_manager.conftest import serve
from tests.runtime.test_record_store import _WORKFLOW, _runtime_graph

_THIRD_CODE = (
    "def third(c):\n    import json\n    rows = [json.loads(v) for v in c[0]]\n"
    "    return [{'id': r['id'], 'text': r['text'] + '?'} for r in rows]\n"
)
_CHANGED_CODE = _THIRD_CODE.replace("+ '?'", "+ '!!'")
_FOUR_OPS: dict[str, Any] = {
    **_WORKFLOW,
    "ops": [
        *_WORKFLOW["ops"],
        {
            "id": "Final",
            "op": "LambdaOp",
            "inputs": ["Join"],
            "fn_name": "third",
            "mode": "row",
            "code": _THIRD_CODE,
            "key": {"node": "Join", "path": "id"},
        },
    ],
    "outputs": [{"name": "final", "ref": "Final"}],
}
_ROWS = {
    "Mark": [{"id": "a", "text": "ONE"}, {"id": "b", "text": "TWO"}],
    "Join": [{"id": "a", "text": "ONE!"}, {"id": "b", "text": "TWO!"}],
    "Final": [{"id": "a", "text": "ONE!?"}, {"id": "b", "text": "TWO!?"}],
}

_LOG = logging.getLogger("t")
_BASE_ROWS = [{"id": "a", "q": "one"}, {"id": "b", "q": "two"}]


def _changed(workflow: dict[str, Any]) -> dict[str, Any]:
    """The workflow with its last op's code changed."""
    ops = [*workflow["ops"][:-1], {**workflow["ops"][-1], "code": _CHANGED_CODE}]
    return {**workflow, "ops": ops}


def _stored(value: Any, op: str, record: str | None = None) -> RecordEntry:
    """One stored entry at ``op``, keyed by ``record`` when it names a row."""
    return RecordEntry(
        digest="d",
        address=RecordAddress(op=op, record=record, round=0),
        value=value,
        source=RecordSource(job="req-1", round=0, task_id=""),
        written_at=0.0,
    )


def _node(graph: Any, name: str) -> str:
    return next(node for node in graph.nodes if graph.op_names.get(node) == name)


def _chat_graph(rows: list[dict[str, str]], seeds: list[int] | None = None) -> Any:
    """The keyed workflow's middle op as a chat keyed by each row's ``id``."""
    graph = _runtime_graph(_FOUR_OPS)
    join = _node(graph, "Join")
    items = [json.dumps(row) for row in rows]
    data_spec: dict[str, Any] = {
        "type": "dataframe",
        "columns": [{"label": "q", "data": {"type": "list", "items": items}}],
        "messages": [{"role": "user", "content": "{q}"}],
    }
    if seeds is not None:
        data_spec["seeds"] = seeds
    graph.nodes[join] = replace(
        graph.nodes[join],
        dependencies=(),
        task_type="inference",
        data_spec=parse_data_spec("inference", data_spec),
        inference_spec={"temperature": 0},
        model_spec={},
    )
    return graph


async def _run(
    storage: InMemoryJobStorage,
    workflow: dict[str, Any],
    results: dict[str, Any],
    job_id: str,
    *,
    served: dict[str, Any] | None = None,
    graph: Any = None,
) -> None:
    """Write one job's finished tasks the way dispatch does."""
    graph = _runtime_graph(workflow) if graph is None else graph
    tasks = {
        f"task-{n}": {"items": [{"output": r} for r in v]} for n, v in results.items()
    }
    await RecordWriter(storage=storage, enabled=True, logger=_LOG).write(
        job_id=job_id,
        member_job_ids=[job_id],
        round_number=0,
        runtime_graph=graph,
        node_task_map={_node(graph, name): f"task-{name}" for name in results},
        task_worker={},
        results=tasks,
        served=served,
    )


def _reusing(graph: Any) -> RequestInfo:
    """A reusing request over ``graph`` that stores what it runs."""
    info = RequestInfo(
        request_id="req-2",
        runtime_graphs={"g": graph},
        data_profile_graphs={},
        store_records=True,
        reuse_records=True,
        member_reuse={"req-2": True},
    )
    info.runtime_graph = graph
    info.batch_id = "batch-1"
    info.output_node_map = dict.fromkeys(graph.output_node_map, ("g", "x"))
    return info


def _decisions(plan: ReusePlan) -> dict[str, tuple[str, str | None]]:
    return {d.op: (d.decision, d.reason) for d in plan.decisions}


def _planner(storage: InMemoryJobStorage) -> ReusePlanner:
    return ReusePlanner(storage=storage, logger=_LOG)


@pytest.mark.asyncio
async def test_one_changed_op_reruns_only_it_and_its_descendants() -> None:
    """Changing the last op misses it alone, and a resubmission hits everywhere."""
    storage = InMemoryJobStorage()
    await _run(storage, _FOUR_OPS, _ROWS, "req-1")
    changed = _changed(_FOUR_OPS)
    graph = _runtime_graph(changed)
    planner = _planner(storage)
    plan, dispatched = planner.plan(runtime_graph=graph, round_number=0, job_id="req-2")
    assert _decisions(plan) == {
        "Mark": ("hit", None),
        "Join": ("hit", None),
        "Final": ("miss", "no entry"),
    }
    assert list(dispatched.nodes) == [_node(graph, "Final")]
    await _run(
        storage,
        changed,
        {"Final": _ROWS["Final"]},
        "req-2",
        served=served_records(plan),
    )
    again, _ = planner.plan(runtime_graph=graph, round_number=0, job_id="req-3")
    assert {d.decision for d in again.decisions} == {"hit"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rows", "seeds", "expected"),
    [
        (_BASE_ROWS, None, ("hit", None)),
        ([{"id": "a", "q": "X"}, {"id": "b", "q": "two"}], None, ("miss", "no entry")),
        ([{"q": "x"}, {"q": "y"}], None, ("miss", "unkeyed row")),
        (_BASE_ROWS, [7, 9], ("hit", None)),
    ],
)
async def test_a_chat_op_hits_on_its_record_keys(
    rows: list[dict[str, str]],
    seeds: list[int] | None,
    expected: tuple[str, str | None],
) -> None:
    """The chat op hits when every row is keyed and its per-row seeds match; a
    changed row, an unkeyed row or another seed misses."""
    storage = InMemoryJobStorage()
    await _run(storage, _FOUR_OPS, _ROWS, "req-1", graph=_chat_graph(_BASE_ROWS, seeds))
    plan, _ = _planner(storage).plan(
        runtime_graph=_chat_graph(rows, seeds), round_number=0, job_id="req-2"
    )
    assert _decisions(plan)["Join"] == expected


@pytest.mark.asyncio
async def test_a_row_steps_served_item_carries_its_typed_value() -> None:
    """A row step's stored value rebuilds as its item, so a reader of a field
    under ``value`` gets it."""
    storage = InMemoryJobStorage()
    await _run(storage, _WORKFLOW, {"Mark": _ROWS["Mark"]}, "req-1")
    graph = _runtime_graph(_WORKFLOW)
    join = _node(graph, "Join")
    spec = graph.nodes[join].data_spec
    assert isinstance(spec, PythonData)
    path = spec.plan[0].model_copy(update={"path": "value.text"})
    graph.nodes[join] = replace(
        graph.nodes[join], data_spec=spec.model_copy(update={"plan": [path]})
    )
    plan, dispatched = _planner(storage).plan(
        runtime_graph=graph, round_number=0, job_id="req-2"
    )
    assert _decisions(plan)["Mark"] == ("hit", None)
    rewritten = dispatched.nodes[join].data_spec
    assert isinstance(rewritten, PythonData)
    assert rewritten.plan[0].values == ["ONE", "TWO"]


@pytest.mark.asyncio
async def test_a_node_bound_seed_answers_a_chat_op_under_reuse() -> None:
    """A chat sampling from a keyed seed node walks that node's planned records,
    so an unchanged rerun hits."""
    rows = [{"id": "a", "q": "one", "seed": 7}, {"id": "b", "q": "two", "seed": 9}]
    graph = _seeded_graph()
    join, mark = _node(graph, "Join"), _node(graph, "Mark")
    storage = InMemoryJobStorage()
    await RecordWriter(storage=storage, enabled=True, logger=_LOG).write(
        job_id="req-1",
        member_job_ids=["req-1"],
        round_number=0,
        runtime_graph=graph,
        node_task_map={mark: "task-mark", join: "task-join"},
        task_worker={},
        results={
            "task-mark": {"items": [{"output": r, "value": r} for r in rows]},
            "task-join": {"items": [{"output": t} for t in ("A", "B")]},
        },
    )
    plan, _ = _planner(storage).plan(
        runtime_graph=graph, round_number=0, job_id="req-2"
    )
    assert _decisions(plan)["Join"] == ("hit", None)


def _seeded_graph() -> Any:
    """The keyed workflow's middle op as a chat drawing each row's node-bound
    seed from the keyed row step before it."""
    graph = _chat_graph(_BASE_ROWS)
    join, mark = _node(graph, "Join"), _node(graph, "Mark")
    spec = graph.nodes[join].data_spec
    assert isinstance(spec, DataFrame)
    graph.nodes[join] = replace(
        graph.nodes[join],
        data_spec=spec.model_copy(
            update={"seeds": NodeRead(node=mark, path="value.items.value.seed")}
        ),
    )
    return graph


@pytest.mark.asyncio
async def test_an_all_hit_request_dispatches_no_task(
    monkeypatch: pytest.MonkeyPatch, job_storage: InMemoryJobStorage
) -> None:
    """A fully answered request submits nothing and indexes what it served."""
    graph = _runtime_graph(_WORKFLOW)
    await _run(job_storage, _WORKFLOW, {k: _ROWS[k] for k in ("Mark", "Join")}, "req-1")
    schedule = Schedule(worker_assignment={"w": list(graph.nodes)})

    async def _never(*_: Any) -> tuple[str, list[str]]:
        raise AssertionError("an all-hit request must submit nothing")

    monkeypatch.setattr(FlowmeshRuntimeManager, "_submit_workflow", _never)
    outputs = await FlowmeshRuntimeManager().process_request(
        _reusing(graph), schedule, worker_ids=["w"]
    )
    index = read_index(job_storage, "req-2")
    assert schedule.worker_assignment == {}
    assert outputs["flat_outputs"] == {
        _node(graph, "Join"): [json.dumps(row) for row in _ROWS["Join"]]
    }
    decisions = [(d.op, d.decision) for d in outputs["reuse_decisions"]]
    assert decisions == [("Mark", "hit"), ("Join", "hit")]
    assert index is not None and {e.op for e in index.entries} == {"Mark", "Join"}


def _list_workflow(consume_suffix: str) -> dict[str, Any]:
    """A list-mode Lambda producing one output per element, read whole by a
    second list-mode Lambda, so both ops are whole-op records."""
    expand = "def expand(c):\n    return [json.loads(v)['id'] for v in c[0]]\n"
    use = f"def consume(c):\n    return [str(v) + {consume_suffix!r} for v in c[0]]\n"
    ops = [
        {"id": "Expand", "inputs": ["Rows"], "fn_name": "expand", "code": expand},
        {"id": "Consume", "inputs": ["Expand"], "fn_name": "consume", "code": use},
    ]
    return {
        "name": "lists",
        "inputs": {"Rows": ['{"id": "a"}', '{"id": "b"}']},
        "ops": [{"op": "LambdaOp", "mode": "list", **o} for o in ops],
        "outputs": [{"name": "consumed", "ref": "Consume"}],
    }


@pytest.mark.asyncio
async def test_a_whole_op_hit_feeds_its_consumer_the_same_column() -> None:
    """A list-mode producer hit answers its consumer with one value per item."""
    storage = InMemoryJobStorage()
    await _run(storage, _list_workflow("!"), {"Expand": ["ab", 3]}, "req-1")
    graph = _runtime_graph(_list_workflow("?"))
    plan, dispatched = _planner(storage).plan(
        runtime_graph=graph, round_number=0, job_id="req-2"
    )
    assert _decisions(plan) == {
        "Expand": ("hit", None),
        "Consume": ("miss", "no entry"),
    }
    (consumer,) = dispatched.nodes.values()
    consumer_spec = consumer.data_spec
    assert isinstance(consumer_spec, PythonData)
    (entry,) = consumer_spec.plan
    (hit,) = plan.hits.values()
    staged = ColumnPlanEntry(kind="stage", node="x")
    assert entry.values == served_column(hit, staged, True) == ["ab", 3]
    blob = python_step.runtime_blob()
    run = PythonData(plan=[entry], code="", fn_name="", timeout_s=60)
    spec = python_step.flowmesh_spec("Consume", run, {blob.key: blob})
    assert "inputs" not in spec
    assert served_column(hit, staged.model_copy(update={"path": "output"}), True) == [
        "ab",
        3,
    ]


def test_a_served_output_item_is_wrapped_as_a_live_read_sees_it() -> None:
    """A python row whose output is itself a dict with an ``output`` key is
    stored unwrapped once, so the served column must wrap it back: unwrapping
    twice would hand the consumer a different value than a live read."""
    row = {"output": 1, "x": 2}
    entry = [_stored(row, "Final", "a")]
    staged = ColumnPlanEntry(kind="stage", node="x")
    assert served_column(entry, staged, True) == [row]
    assert served_column(entry, staged.model_copy(update={"path": "output"}), True) == [
        row
    ]
    flat_entry = ColumnPlanEntry(kind="stage", node="x", flatten=True)
    flat = [_stored([["x", "y"]], "Expand")]
    assert served_column(flat, flat_entry, True) == ["x", "y"]
    item = [_stored({"index": 0, "score": 0.5}, "E", "a")]
    assert served_column(
        item, ColumnPlanEntry(kind="stage", node="x", path="score"), False
    ) == [0.5]


@pytest.mark.asyncio
async def test_a_hit_output_survives_a_partial_run(
    monkeypatch: pytest.MonkeyPatch, job_storage: InMemoryJobStorage
) -> None:
    """Two outputs, one op changed: the hit output is rebuilt beside the miss."""
    outs = [{"name": "joined", "ref": "Join"}, {"name": "final", "ref": "Final"}]
    await _run(job_storage, {**_FOUR_OPS, "outputs": outs}, _ROWS, "req-1")
    graph = _runtime_graph(_changed({**_FOUR_OPS, "outputs": outs}))
    request_info = _reusing(graph)
    final_node, final_task = _node(graph, "Final"), "task-final"

    async def _submit(*_: Any) -> tuple[str, list[str]]:
        return "wf-1", [final_task]

    async def _maps(*_: Any) -> tuple[dict[str, str], dict[str, str]]:
        return {final_task: final_node}, {final_node: final_task}

    async def _retrieve(_: str) -> dict[str, Any]:
        return {"items": [{"output": row} for row in _ROWS["Final"]]}

    fm = SimpleNamespace(results=SimpleNamespace(retrieve=_retrieve))
    serve(monkeypatch, {final_task: "DONE"}, {final_task: final_node})
    monkeypatch.setattr(FlowmeshRuntimeManager, "_submit_workflow", _submit)
    monkeypatch.setattr(FlowmeshRuntimeManager, "_resolve_task_node_maps", _maps)
    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: fm))
    schedule = Schedule(worker_assignment={"w": [final_node]})
    outputs = await FlowmeshRuntimeManager().process_request(
        request_info, schedule, ["w"]
    )
    assert outputs["flat_outputs"] == {
        _node(graph, "Join"): [json.dumps(row) for row in _ROWS["Join"]],
        _node(graph, "Final"): [json.dumps(row) for row in _ROWS["Final"]],
    }
