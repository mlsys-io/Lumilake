"""Whole-op reuse: a reusing plan answers an op whose every record the store
already holds and dispatches the rest."""

import json
import logging
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from lumilake_server.data_specs import (
    ColumnPlanEntry,
    DataFrame,
    NodeRead,
    PythonData,
    parse_data_spec,
)
from lumilake_server.ops import RecordKey
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
    _seed_first_order,
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
_MARK_CODE = "lambda c: json.loads(c[0][0])"
_GRADE_CODE = "lambda c: ['ok' for v in c[0]]"
_FINAL: dict[str, Any] = {"id": "Final", "op": "LambdaOp", "inputs": ["Join"]}
_FINAL |= {"fn_name": "third", "code": _THIRD_CODE}
_FINAL |= {"mode": "row", "key": {"node": "Join", "path": "id"}}
_FOUR_OPS = {**_WORKFLOW, "outputs": [{"name": "final", "ref": "Final"}]}
_FOUR_OPS["ops"] = [*_WORKFLOW["ops"], _FINAL]
_ROWS = {
    k: [{"id": i, "text": t} for i, t in (("a", "ONE" + m), ("b", "TWO" + m))]
    for k, m in (("Mark", ""), ("Join", "!"), ("Final", "!?"))
}

_LOG = logging.getLogger("t")
_PARSE_TEXT = """\
name: parse
inputs: {Rows: %s}
ops:
  - {id: Mark, op: LambdaOp, inputs: [Rows], mode: row, fn_name: mark,
     key: {node: Rows, path: id}, code: "lambda c: json.loads(c[0][0])"}
  - {id: Parsed, op: ParseOp, inputs: [Mark], format: json}
  - {id: Grade, op: LambdaOp, inputs: [Parsed], mode: row, fn_name: grade,
     key: {node: Parsed, path: id}, code: "lambda c: ['ok' for v in c[0]]"}
outputs: [{name: g, ref: Grade}]
"""


_BASE_ROWS = [{"id": "a", "q": "one"}, {"id": "b", "q": "two"}]


def _changed(workflow: dict[str, Any]) -> dict[str, Any]:
    """The workflow with its last op's code changed."""
    ops = [*workflow["ops"][:-1], {**workflow["ops"][-1], "code": _CHANGED_CODE}]
    return {**workflow, "ops": ops}


def _stored(value: Any, op: str, record: str | None = None) -> RecordEntry:
    """One stored entry at ``op``, keyed by ``record`` when it names a row."""
    source = RecordSource(job="req-1", round=0, task_id="")
    address = RecordAddress(op=op, record=record, round=0)
    return RecordEntry(
        digest="d", value=value, written_at=0.0, source=source, address=address
    )


def _read_at(graph: Any, node: str, path: str) -> Any:
    """The op reading its first served column at ``path``."""
    assert isinstance(spec := graph.nodes[node].data_spec, PythonData)
    entry = spec.plan[0].model_copy(update={"path": path})
    return replace(
        graph.nodes[node], data_spec=spec.model_copy(update={"plan": [entry]})
    )


def _node(graph: Any, name: str) -> str:
    return next(n for n in graph.nodes if graph.op_names.get(n) == name)


def _chat_graph(rows: list[dict[str, Any]], seeds: list[int] | None = None) -> Any:
    """The keyed workflow's middle op as a chat keyed by each row's ``id``."""
    graph = _runtime_graph(_FOUR_OPS)
    join = _node(graph, "Join")
    items = [json.dumps(row) for row in rows]
    column = {"label": "q", "data": {"type": "list", "items": items}}
    message = {"role": "user", "content": "{q}"}
    data_spec: dict[str, Any] = {"type": "dataframe", "columns": [column]}
    data_spec |= {"messages": [message], "seeds": seeds}
    graph.nodes[join] = replace(
        graph.nodes[join],
        dependencies=(),
        task_type="inference",
        model_spec={},
        inference_spec={"temperature": 0},
        data_spec=parse_data_spec("inference", data_spec),
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
    items: dict[str, list[Any]] | None = None,
) -> None:
    """Write a job's finished tasks the way dispatch does; ``items`` gives raw items."""
    graph = _runtime_graph(workflow) if graph is None else graph
    raw = {n: {"items": [{"output": r} for r in v]} for n, v in results.items()}
    raw |= {n: {"items": r} for n, r in (items or {}).items()}
    await RecordWriter(storage=storage, enabled=True, logger=_LOG).write(
        job_id=job_id,
        member_job_ids=[job_id],
        round_number=0,
        runtime_graph=graph,
        task_worker={},
        served=served,
        node_task_map={_node(graph, n): f"task-{n}" for n in raw},
        results={f"task-{n}": r for n, r in raw.items()},
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
    info.runtime_graph, info.batch_id = graph, "batch-1"
    info.output_node_map = dict.fromkeys(graph.output_node_map, ("g", "x"))
    return info


def _decisions(plan: ReusePlan) -> dict[str, tuple[str, str | None]]:
    return {d.op: (d.decision, d.reason) for d in plan.decisions}


def _planner(storage: InMemoryJobStorage) -> ReusePlanner:
    return ReusePlanner(storage=storage, logger=_LOG)


def _planned(storage: InMemoryJobStorage, graph: Any, job_id: str = "req-2") -> Any:
    """One plan over ``graph`` at round 0."""
    return _planner(storage).plan(runtime_graph=graph, round_number=0, job_id=job_id)


@pytest.mark.asyncio
async def test_one_changed_op_reruns_only_it_and_its_descendants() -> None:
    """Changing the last op misses it alone, and a resubmission hits everywhere."""
    storage = InMemoryJobStorage()
    await _run(storage, _FOUR_OPS, _ROWS, "req-1")
    changed = _changed(_FOUR_OPS)
    graph = _runtime_graph(changed)
    planner = _planner(storage)
    plan, dispatched = planner.plan(runtime_graph=graph, round_number=0, job_id="req-2")
    the_miss = {"Final": ("miss", "no entry")}
    assert _decisions(plan) == {"Mark": ("hit", None), "Join": ("hit", None)} | the_miss
    assert list(dispatched.nodes) == [_node(graph, "Final")]
    served = served_records(plan)
    await _run(storage, changed, {"Final": _ROWS["Final"]}, "req-2", served=served)
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
    """A keyed chat hits; a changed, unkeyed or reseeded row misses."""
    storage = InMemoryJobStorage()
    await _run(storage, _FOUR_OPS, _ROWS, "req-1", graph=_chat_graph(_BASE_ROWS, seeds))
    plan, _ = _planned(storage, _chat_graph(rows, seeds))
    assert _decisions(plan)["Join"] == expected


@pytest.mark.asyncio
async def test_a_served_local_chat_keeps_its_grouped_item_shape() -> None:
    """A served local rowwise chat answers its reader with the live grouped
    item, one ``output`` list holding every row's reply, not one item per row."""
    graph = _chat_graph(_BASE_ROWS)
    join = _node(graph, "Join")
    replies = [row["id"] for row in _BASE_ROWS]
    storage = InMemoryJobStorage()
    await _run(
        storage,
        _FOUR_OPS,
        {"Join": replies},
        "req-1",
        graph=graph,
        items={"Join": [{"output": replies}]},
    )
    plan, dispatched = _planned(storage, graph)
    assert _decisions(plan)["Join"] == ("hit", None)
    assert not any(n == join for n in dispatched.nodes)
    final = dispatched.nodes[_node(graph, "Final")]
    assert isinstance(final.data_spec, PythonData)
    (entry,) = final.data_spec.plan
    assert entry.values == [["a", "b"]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [7, True, None, '{"a": 1}', {"a": 1}, ["a"], "plain", 0.5],
    ids=["int", "bool", "null", "json-text", "dict", "list", "text", "float"],
)
async def test_a_row_steps_served_item_equals_its_live_item(value: Any) -> None:
    """A served row step's item is the live one, typed ``value`` under it."""
    graph = _runtime_graph(_WORKFLOW)
    join = _node(graph, "Join")
    text = value if isinstance(value, str) else json.dumps(value)
    graph.nodes[join] = _read_at(graph, join, "value")
    storage, items = InMemoryJobStorage(), None
    items = {"Mark": [{"output": text, "value": value}] * 2}
    await _run(storage, _WORKFLOW, {"Mark": [value] * 2}, "req-1", items=items)
    plan, sent = _planned(storage, graph)
    served = sent.nodes[join].data_spec
    assert _decisions(plan)["Mark"] == ("hit", None)
    assert isinstance(served, PythonData) and served.plan[0].values == [value, value]


@pytest.mark.asyncio
async def test_a_parse_steps_field_is_served_under_reuse() -> None:
    """A parse step's stored record serves a reader of ``items.output.<field>``."""
    text = _PARSE_TEXT % [json.dumps({"id": "a", "text": "one"})]
    graph = _runtime_graph(yaml.safe_load(text))
    node = _node(graph, "Grade")
    graph.nodes[node] = _read_at(graph, node, "value.text")
    rows = [{"id": "a", "text": "one"}]
    storage = InMemoryJobStorage()
    await _run(storage, {}, {"Mark": rows, "Parsed": rows}, "req-1", graph=graph)
    plan, _ = _planned(storage, graph)
    assert _decisions(plan)["Parsed"] == ("hit", None), _decisions(plan)


@pytest.mark.asyncio
async def test_a_redispatched_parse_takes_a_served_inputs_keys() -> None:
    """A parse step rerun while its keyed input is served stores under that
    input's keys, so the next run hits it."""
    rows = [{"id": i, "text": t} for i, t in (("a", "one"), ("b", "two"))]
    parsed = [{"text": row["text"]} for row in rows]
    text = _PARSE_TEXT % [json.dumps(row) for row in rows]
    workflow, changed = yaml.safe_load(text), yaml.safe_load(text)
    changed["ops"][1] = {**changed["ops"][1], "on_fail": "raw"}
    storage = InMemoryJobStorage()
    await _run(storage, workflow, {"Mark": rows, "Parsed": parsed}, "req-1")
    planner, graph = _planner(storage), _runtime_graph(changed)
    plan, dispatched = planner.plan(runtime_graph=graph, round_number=0, job_id="req-2")
    assert _decisions(plan)["Parsed"] == ("miss", "no entry"), _decisions(plan)
    served = served_records(plan)
    await _run(
        storage, changed, {"Parsed": parsed}, "req-2", graph=dispatched, served=served
    )
    entries = read_index(storage, "req-2")
    assert entries is not None
    assert {e.record for e in entries.entries if e.op == "Parsed"} == {"a", "b"}
    again, _ = planner.plan(runtime_graph=graph, round_number=0, job_id="req-3")
    assert _decisions(again)["Parsed"] == ("hit", None), _decisions(again)


@pytest.mark.asyncio
async def test_a_bare_int_seed_op_is_served_under_reuse() -> None:
    """A seed op's bare ints serve its chat; a changed chat still holds the seed."""
    graph = _chat_graph(_BASE_ROWS)
    join, mark = _node(graph, "Join"), _node(graph, "Mark")
    assert isinstance(spec := graph.nodes[join].data_spec, DataFrame)
    read = NodeRead(node=mark, path="value.items.value")
    graph.nodes[join] = replace(
        graph.nodes[join], data_spec=spec.model_copy(update={"seeds": read})
    )
    graph.record_keys.pop(mark)
    items = {"Mark": [{"output": str(n), "value": n} for n in (7, 9)]}
    storage = InMemoryJobStorage()
    await _run(storage, _FOUR_OPS, _ROWS, "req-1", graph=graph, items=items)
    plan, _ = _planned(storage, graph)
    assert _decisions(plan)["Join"] == ("hit", None), _decisions(plan)
    changed = _chat_graph(_BASE_ROWS)
    changed.nodes[join] = replace(
        changed.nodes[join],
        model_spec={"top_p": 0.9},
        data_spec=spec.model_copy(update={"seeds": read}),
    )
    changed.record_keys.pop(mark)
    plan, dispatched = _planned(storage, changed)
    assert _decisions(plan)["Mark"] == ("miss", "read by dispatched 'Join'")
    assert {mark, join} <= set(dispatched.nodes)
    for node in dispatched.nodes.values():
        assert set(node.dependencies) <= set(dispatched.nodes)


@pytest.mark.asyncio
async def test_a_list_mode_seed_op_is_served_under_reuse() -> None:
    """A list-mode seed node serves its chat from the list's own item shape."""
    graph = _chat_graph(_BASE_ROWS)
    join, mark = _node(graph, "Join"), _node(graph, "Mark")
    assert isinstance(spec := graph.nodes[join].data_spec, DataFrame)
    read = NodeRead(node=mark, path="value.items.output")
    graph.nodes[join] = replace(
        graph.nodes[join], data_spec=spec.model_copy(update={"seeds": read})
    )
    assert isinstance(mark_spec := graph.nodes[mark].data_spec, PythonData)
    graph.nodes[mark] = replace(
        graph.nodes[mark], data_spec=mark_spec.model_copy(update={"mode": "list"})
    )
    graph.record_keys.pop(mark)
    items = {"Mark": [{"output": 11}, {"output": 12}]}
    storage = InMemoryJobStorage()
    await _run(storage, _FOUR_OPS, _ROWS, "req-1", graph=graph, items=items)
    plan, _ = _planned(storage, graph)
    assert _decisions(plan)["Join"] == ("hit", None), _decisions(plan)


@pytest.mark.asyncio
async def test_an_all_hit_request_dispatches_no_task(
    monkeypatch: pytest.MonkeyPatch,
    job_storage: InMemoryJobStorage,
) -> None:
    """A fully answered request submits nothing and indexes what it served."""
    graph = _runtime_graph(_WORKFLOW)
    await _run(job_storage, _WORKFLOW, {k: _ROWS[k] for k in ("Mark", "Join")}, "req-1")
    schedule = Schedule(worker_assignment={"w": list(graph.nodes)})

    async def _never(*_: Any) -> tuple[str, list[str]]:
        raise AssertionError("an all-hit request must submit nothing")

    monkeypatch.setattr(FlowmeshRuntimeManager, "_submit_workflow", _never)
    reusing = _reusing(graph)
    outputs = await FlowmeshRuntimeManager().process_request(reusing, schedule, ["w"])
    index = read_index(job_storage, "req-2")
    assert schedule.worker_assignment == {}
    assert outputs["flat_outputs"] == {
        _node(graph, "Join"): [json.dumps(row) for row in _ROWS["Join"]]
    }
    decisions = [(d.op, d.decision) for d in outputs["reuse_decisions"]]
    assert decisions == [("Mark", "hit"), ("Join", "hit")]
    assert index is not None and {e.op for e in index.entries} == {"Mark", "Join"}


def _list_workflow(consume_suffix: str) -> dict[str, Any]:
    """A list-mode Lambda read whole by a second, so both are whole-op records."""
    expand = "def expand(c):\n    return [json.loads(v)['id'] for v in c[0]]\n"
    use = f"def consume(c):\n    return [str(v) + {consume_suffix!r} for v in c[0]]\n"
    ops = [{"id": "Expand", "inputs": ["Rows"], "fn_name": "expand"}]
    ops += [{"id": "Consume", "inputs": ["Expand"], "fn_name": "consume"}]
    codes = (expand, use)
    ops = [
        o | {"op": "LambdaOp", "mode": "list", "code": c} for o, c in zip(ops, codes)
    ]
    outs = [{"name": "consumed", "ref": "Consume"}]
    inputs = {"Rows": ['{"id": "a"}', '{"id": "b"}']}
    return {"name": "lists", "inputs": inputs, "ops": ops, "outputs": outs}


@pytest.mark.asyncio
async def test_a_whole_op_hit_feeds_its_consumer_the_same_column() -> None:
    """A hit producer answers its consumer, flat or scored as the store holds it."""
    storage = InMemoryJobStorage()
    await _run(storage, _list_workflow("!"), {"Expand": ["ab", 3]}, "req-1")
    graph = _runtime_graph(_list_workflow("?"))
    plan, dispatched = _planned(storage, graph)
    assert _decisions(plan) == dict(Expand=("hit", None), Consume=("miss", "no entry"))
    (consumer,) = dispatched.nodes.values()
    assert isinstance(consumer.data_spec, PythonData)
    (entry,) = consumer.data_spec.plan
    (hit,) = plan.hits.values()
    staged = ColumnPlanEntry(kind="stage", node="x")
    assert entry.values == served_column(hit, staged, "output") == ["ab", 3]
    run = PythonData(plan=[entry], code="", fn_name="", timeout_s=60)
    blob = python_step.runtime_blob()
    assert "inputs" not in python_step.flowmesh_spec("Consume", run, {blob.key: blob})
    flat, flatten = [_stored([["x", "y"]], "Expand")], staged.model_copy(
        update={"flatten": True}
    )
    assert served_column(flat, flatten, "output") == ["x", "y"]
    score = staged.model_copy(update={"path": "score"})
    assert served_column([_stored({"score": 0.5}, "E", "a")], score, None) == [0.5]


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
        return {"items": [{"output": r} for r in _ROWS["Final"]]}

    fm = SimpleNamespace(results=SimpleNamespace(retrieve=_retrieve))
    serve(monkeypatch, {final_task: "DONE"}, {final_task: final_node})
    patch = monkeypatch.setattr
    patch(FlowmeshRuntimeManager, "_submit_workflow", _submit)
    patch(FlowmeshRuntimeManager, "_resolve_task_node_maps", _maps)
    patch(FlowmeshRuntimeManager, "fm", property(lambda self: fm))
    schedule = Schedule(worker_assignment={"w": [final_node]})
    manager = FlowmeshRuntimeManager()
    outputs = await manager.process_request(request_info, schedule, ["w"])
    want = {
        _node(graph, n): [json.dumps(r) for r in _ROWS[n]] for n in ("Join", "Final")
    }
    assert outputs["flat_outputs"] == want


@pytest.mark.asyncio
async def test_a_reuse_run_stores_the_seeded_chat_it_redispatched() -> None:
    """Keyed Cases served and a seeded keyed chat re-dispatched: the reuse write
    holds the chat's per-row records, so a third run serves the chat."""
    graph = _chat_graph(_BASE_ROWS)
    join, mark = _node(graph, "Join"), _node(graph, "Mark")
    seeds = "lambda_keyed_Seeds_1"
    graph = replace(
        graph,
        nodes={
            **graph.nodes,
            seeds: replace(
                graph.nodes[mark],
                node_id=seeds,
                dependencies=(),
                data_spec=PythonData(
                    mode="list", fn_name="seeds", code="lambda c: [7, 9]"
                ),
            ),
        },
        node_order=[*graph.node_order, seeds],
    )
    graph.op_names[seeds] = "Seeds"
    assert isinstance(spec := graph.nodes[join].data_spec, DataFrame)
    graph.nodes[join] = replace(
        graph.nodes[join],
        data_spec=spec.model_copy(
            update={"seeds": NodeRead(node=seeds, path="value.items.output")}
        ),
    )
    graph.record_keys[join] = RecordKey(node=mark, path="id")
    replies = ["A", "B"]
    items: dict[str, list[Any]] = {
        "Seeds": [{"output": 7}, {"output": 9}],
        "Join": [{"output": replies}],
    }
    storage = InMemoryJobStorage()
    await _run(
        storage,
        _FOUR_OPS,
        {"Mark": _BASE_ROWS, "Join": replies},
        "req-1",
        graph=graph,
        items=items,
    )
    changed = replace(
        graph,
        nodes={
            **graph.nodes,
            join: replace(
                graph.nodes[join],
                data_spec=graph.nodes[join].data_spec.model_copy(
                    update={"messages": [{"role": "user", "content": "{q}?"}]}
                ),
            ),
        },
    )
    plan, dispatched = _planned(storage, changed)
    assert _decisions(plan)["Mark"] == ("hit", None), _decisions(plan)
    assert _decisions(plan)["Join"] == ("miss", "no entry"), _decisions(plan)
    await _run(
        storage,
        _FOUR_OPS,
        {"Join": replies},
        "req-2",
        graph=dispatched,
        served=served_records(plan),
        items=items,
    )
    chat = graph.op_names[join]
    index = read_index(storage, "req-2")
    assert index is not None
    stored = [e for e in index.entries if e.op == chat]
    assert {e.record for e in stored} == {"a", "b"}
    again, _ = _planned(storage, changed, job_id="req-3")
    assert _decisions(again)["Join"] == ("hit", None), _decisions(again)


def test_a_seed_node_that_depends_on_a_later_node_stays_after_it() -> None:
    """A seed node ordered after its chat but reading a node between them keeps
    that node ahead of it, so the plan order stays topological."""
    base = _chat_graph(_BASE_ROWS)
    mark, join = _node(base, "Mark"), _node(base, "Join")
    between = replace(base.nodes[mark], node_id="b", dependencies=(mark,))
    seed = replace(base.nodes[mark], node_id="s", dependencies=("b",))
    graph = replace(
        base,
        nodes={**base.nodes, "b": between, "s": seed},
        node_order=[mark, join, "b", "s"],
    )
    graph.op_names["s"] = "Seed"
    spec = graph.nodes[join].data_spec
    assert isinstance(spec, DataFrame)
    graph.nodes[join] = replace(
        graph.nodes[join],
        data_spec=spec.model_copy(update={"seeds": NodeRead(node="s")}),
    )
    order = _seed_first_order(graph)
    assert order.index(mark) < order.index(join)
    assert order.index("b") < order.index("s")
    assert order.index("s") < order.index(join)
