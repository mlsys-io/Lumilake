"""Whole-op reuse: a reusing plan answers an op whose every record the store
already holds and dispatches the rest."""

import json
import logging
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from lumilake import envs
from lumilake_server.common import GenerationConfig
from lumilake_server.data_specs import (
    ColumnPlanEntry,
    DataFrame,
    GraphTemplateData,
    ListData,
    NodeRead,
    PythonData,
    parse_data_spec,
)
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    LLMChatOp,
    OpMessage,
    RecordKey,
    as_output,
    input_placeholder,
)
from lumilake_server.runtime import python_step
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.protocol import RequestCancelledError
from lumilake_server.runtime.record_store import (
    RecordAddress,
    RecordEntry,
    RecordSource,
)
from lumilake_server.runtime.record_writer import NodeRecords, RecordWriter, read_index
from lumilake_server.runtime.request import RequestInfo
from lumilake_server.runtime.reuse_planner import (
    ReusePlan,
    ReusePlanner,
    rebuilt_output,
    served_column,
    served_records,
    writer_records,
)
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.runtime.runtime_manager import FlowmeshRuntimeManager
from lumilake_server.utils.job_storage import InMemoryJobStorage
from tests.runtime.runtime_manager.conftest import job_storage as job_storage
from tests.runtime.runtime_manager.conftest import serve
from tests.runtime.test_record_store import _CODE_SECOND, _WORKFLOW, _runtime_graph

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


def _patient_workflow(ops: list[dict[str, Any]], output: str, rows: list[Any]) -> Any:
    """A workflow over ``rows`` that outputs ``output``."""
    return {
        "name": "patients",
        "inputs": {"Rows": [json.dumps(row) for row in rows]},
        "ops": ops,
        "outputs": [{"name": "o", "ref": output}],
    }


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


def _read_at(graph: Any, node: str, path: str) -> Any:
    """The op reading its first served column at ``path``."""
    assert isinstance(spec := graph.nodes[node].data_spec, PythonData)
    entry = spec.plan[0].model_copy(update={"path": path})
    return replace(
        graph.nodes[node], data_spec=spec.model_copy(update={"plan": [entry]})
    )


def _node(graph: Any, name: str) -> str:
    return next(node for node in graph.nodes if graph.op_names.get(node) == name)


def _chat_graph(rows: list[dict[str, Any]], seeds: list[int] | None = None) -> Any:
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
    content_nodes: frozenset[str] = frozenset(),
    task_ids: dict[str, str] | None = None,
    items: dict[str, list[Any]] | None = None,
) -> None:
    """Write one job's finished tasks the way dispatch does; ``items`` are raw items."""
    graph = _runtime_graph(workflow) if graph is None else graph
    tasks = {
        f"task-{n}": {"items": [{"output": r} for r in v]} for n, v in results.items()
    }
    tasks |= {f"task-{n}": {"items": v} for n, v in (items or {}).items()}
    await RecordWriter(storage=storage, enabled=True, logger=_LOG).write(
        job_id=job_id,
        member_job_ids=[job_id],
        round_number=0,
        runtime_graph=graph,
        node_task_map=task_ids
        or {
            _node(graph, name): f"task-{name}" for name in {**results, **(items or {})}
        },
        results=tasks,
        served=served,
        content_nodes=content_nodes,
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


def _read_sub_run(monkeypatch: pytest.MonkeyPatch, graph: Any, response: Any = None):
    """Patch a read sub-run over ``graph``: one read task, done, answering it."""
    read = _node(graph, "Read")
    serve(monkeypatch, {"task-read": "PENDING" if response is None else "DONE"})
    names = ("_submit_workflow", "_resolve_task_node_maps", "_read_task_result")
    answers = (("wf-sub", ["task-read"]), ({"task-read": read}, {read: "task-read"}))
    for name, answer in zip(names, (*answers, response)):
        monkeypatch.setattr(
            FlowmeshRuntimeManager, name, AsyncMock(return_value=answer)
        )
    monkeypatch.setattr(envs, "LUMILAKE_POLL_INTERVAL_SECONDS", 0.0)
    return read


def _decisions(plan: ReusePlan) -> dict[str, tuple[str, str | None]]:
    return {d.op: (d.decision, d.reason) for d in plan.decisions}


def _planner(storage: InMemoryJobStorage) -> ReusePlanner:
    return ReusePlanner(storage=storage, logger=_LOG)


def _planned(storage: InMemoryJobStorage, graph: Any, job_id="req-2", **kw: Any):
    """One plan over ``graph``, at round 0."""
    return _planner(storage).plan(
        runtime_graph=graph, round_number=0, job_id=job_id, **kw
    )


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
    items = {"Mark": [{"output": text, "value": value}] * 2}
    storage = InMemoryJobStorage()
    await _run(storage, _WORKFLOW, {"Mark": [value] * 2}, "req-1", items=items)
    plan, sent = _planned(storage, graph)
    served = sent.nodes[join].data_spec
    assert _decisions(plan)["Mark"] == ("hit", None)
    assert isinstance(served, PythonData) and served.plan[0].values == [value, value]


@pytest.mark.asyncio
async def test_a_row_steps_served_item_carries_its_typed_value() -> None:
    """A row step's served item keeps a typed ``value`` under it."""
    graph = _runtime_graph(_WORKFLOW)
    join = _node(graph, "Join")
    spec = graph.nodes[join].data_spec
    assert isinstance(spec, PythonData)
    entry = spec.plan[0].model_copy(update={"path": "value.text"})
    spec = spec.model_copy(update={"plan": [entry]})
    graph.nodes[join] = replace(graph.nodes[join], data_spec=spec)
    await _run(
        storage := InMemoryJobStorage(), _WORKFLOW, {"Mark": _ROWS["Mark"]}, "req-1"
    )
    plan, sent = _planner(storage).plan(
        runtime_graph=graph, round_number=0, job_id="req-2"
    )
    served = sent.nodes[join].data_spec
    assert _decisions(plan)["Mark"] == ("hit", None)
    assert isinstance(served, PythonData) and served.plan[0].values == ["ONE", "TWO"]


_MARK_OP: dict[str, Any] = {
    "id": "Mark",
    "op": "LambdaOp",
    "inputs": ["Rows"],
    "mode": "row",
    "fn_name": "mark",
    "code": "lambda c: json.loads(c[0][0])",
    "key": {"node": "Rows", "path": "id"},
}
_PARSE_OP = {"id": "Parsed", "op": "ParseOp", "inputs": ["Mark"]}
_CHANGED_PARSE = {**_PARSE_OP, "on_fail": "raw"}
_CODE_OUT = "def out(c):\n    return [str(v) + '!' for v in c[0]]\n"


def _parse_workflow(rows: list[dict[str, Any]], *more: dict[str, Any]) -> Any:
    """A keyed mark step, a parse step and further ops, over ``rows`` as rows."""
    ops = [_MARK_OP, _PARSE_OP, *more]
    ref = ops[-1]["id"]
    return {
        "name": "parse",
        "inputs": {"Rows": [json.dumps(row) for row in rows]},
        "ops": ops,
        "outputs": [{"name": "o", "ref": ref}],
    }


@pytest.mark.asyncio
async def test_a_parse_steps_field_is_served_under_reuse() -> None:
    """A parse step's stored record serves a reader of ``items.output.<field>``."""
    rows = [{"id": "a", "text": "one"}]
    grade = {
        "id": "Grade",
        "op": "LambdaOp",
        "inputs": ["Parsed"],
        "mode": "row",
        "fn_name": "grade",
        "key": {"node": "Parsed", "path": "id"},
        "code": "lambda c: ['ok' for v in c[0]]",
    }
    graph = _runtime_graph(_parse_workflow(rows, grade))
    node = _node(graph, "Grade")
    graph.nodes[node] = _read_at(graph, node, "value.text")
    storage = InMemoryJobStorage()
    await _run(storage, {}, {"Mark": rows, "Parsed": rows}, "req-1", graph=graph)
    plan, _ = _planned(storage, graph)
    assert _decisions(plan)["Parsed"] == ("hit", None), _decisions(plan)


@pytest.mark.asyncio
async def test_a_reader_keeps_its_declared_path_on_a_served_input() -> None:
    """A reader declaring one path of a several-path keyed input stores under
    that path's value both when it runs and when that input is served."""
    rows = [{"pid": f"p{n}", "nctid": f"N{n}", "note": f"n{n}"} for n in (1, 2)]
    workflow = _patient_workflow([_NOTES, _ASK], "Ask", rows)
    changed = {**workflow, "ops": [_NOTES, {**_ASK, "code": "lambda c: ['?']"}]}
    replies = [row["note"] + "!" for row in rows]
    storage = InMemoryJobStorage()
    await _run(storage, workflow, {"Notes": rows, "Ask": replies}, "req-1")
    plan, dispatched = _planned(storage, _runtime_graph(changed))
    assert _decisions(plan)["Notes"] == ("hit", None), _decisions(plan)
    await _run(
        storage,
        changed,
        {"Ask": replies},
        "req-2",
        graph=dispatched,
        served=served_records(plan),
    )
    produced, served = read_index(storage, "req-1"), read_index(storage, "req-2")
    assert produced is not None and served is not None
    for index in (produced, served):
        assert [e.record for e in index.entries if e.op == "Ask"] == ["N1", "N2"]


@pytest.mark.asyncio
async def test_a_redispatched_parse_keeps_its_served_inputs_keys() -> None:
    """A parse step re-dispatched over a served keyed input stores under that
    input's keys, though its own values carry no pid, so the next run hits it."""
    rows = [{"pid": f"p{n}", "note": f"n{n}"} for n in (1, 2)]
    notes = [json.dumps({"note": row["note"]}) for row in rows]
    parsed = [json.loads(note) for note in notes]
    workflow = _patient_workflow([_NOTED, _PARSE], "Parsed", rows)
    changed = {**workflow, "ops": [_NOTED, {**_PARSE, "on_fail": "raw"}]}
    storage = InMemoryJobStorage()
    await _run(storage, workflow, {"Notes": notes, "Parsed": parsed}, "req-1")
    plan, dispatched = _planned(storage, _runtime_graph(changed))
    assert _decisions(plan)["Parsed"][0] == "miss", _decisions(plan)
    await _run(
        storage,
        changed,
        {"Parsed": parsed},
        "req-2",
        graph=dispatched,
        served=served_records(plan),
    )
    entries = read_index(storage, "req-2")
    assert entries is not None
    assert {e.record for e in entries.entries if e.op == "Parsed"} == {"p1", "p2"}
    again, _ = _planned(storage, _runtime_graph(changed))
    assert _decisions(again)["Parsed"] == ("hit", None), _decisions(again)


@pytest.mark.asyncio
async def test_a_redispatched_parse_takes_a_served_inputs_keys() -> None:
    """A parse step rerun while its keyed input is served stores under that
    input's keys, so the next run hits it."""
    rows = [{"id": i, "text": t} for i, t in (("a", "one"), ("b", "two"))]
    parsed = [{"text": row["text"]} for row in rows]
    workflow = _parse_workflow(rows)
    changed = {**workflow, "ops": [_MARK_OP, _CHANGED_PARSE]}
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
async def test_a_second_redispatched_row_step_keeps_inheriting_keys() -> None:
    """An undeclared row step after a re-dispatched parse stores under the served
    keys, so a third run hits it instead of missing with no entry."""
    rows = [{"id": i, "text": t} for i, t in (("a", "one"), ("b", "two"))]
    parsed, out = [{"text": row["text"]} for row in rows], ["one", "two"]
    out_op = {
        "id": "Out",
        "op": "LambdaOp",
        "inputs": ["Parsed"],
        "mode": "row",
        "fn_name": "out",
        "code": _CODE_OUT,
    }
    workflow = _parse_workflow(rows, out_op)
    changed = {**workflow, "ops": [_MARK_OP, _CHANGED_PARSE, out_op]}
    storage = InMemoryJobStorage()
    await _run(storage, workflow, {"Mark": rows, "Parsed": parsed, "Out": out}, "req-1")
    planner, graph = _planner(storage), _runtime_graph(changed)
    plan, dispatched = planner.plan(runtime_graph=graph, round_number=0, job_id="req-2")
    assert _decisions(plan)["Parsed"] == ("miss", "no entry"), _decisions(plan)
    served = served_records(plan)
    await _run(
        storage,
        changed,
        {"Parsed": parsed, "Out": out},
        "req-2",
        graph=dispatched,
        served=served,
    )
    entries = read_index(storage, "req-2")
    assert entries is not None
    assert {e.record for e in entries.entries if e.op == "Out"} == {"a", "b"}
    again, _ = planner.plan(runtime_graph=graph, round_number=0, job_id="req-3")
    assert _decisions(again)["Out"] == ("hit", None), _decisions(again)


@pytest.mark.asyncio
async def test_a_bare_int_seed_op_is_served_under_reuse() -> None:
    """A seed op's bare ints serve its chat; a changed chat still holds the seed."""
    graph = _chat_graph(_BASE_ROWS)
    join, mark = _node(graph, "Join"), _node(graph, "Mark")
    assert isinstance(spec := graph.nodes[join].data_spec, DataFrame)
    seeds = NodeRead(node=mark, path="value.items.value")
    graph.nodes[join] = replace(
        graph.nodes[join], data_spec=spec.model_copy(update={"seeds": seeds})
    )
    graph.record_keys.pop(mark)
    ints = {"Mark": [{"output": str(n), "value": n} for n in (7, 9)]}
    storage = InMemoryJobStorage()
    rows = {"Join": _ROWS["Join"]}
    await _run(storage, _FOUR_OPS, rows, "req-1", graph=graph, items=ints)
    plan, _ = _planned(storage, graph)
    assert _decisions(plan)["Join"] == ("hit", None), _decisions(plan)
    changed = _chat_graph(_BASE_ROWS)
    changed.nodes[join] = replace(
        changed.nodes[join],
        model_spec={"top_p": 0.9},
        data_spec=graph.nodes[join].data_spec,
    )
    changed.record_keys.pop(mark)
    plan, dispatched = _planned(storage, changed)
    assert _decisions(plan)["Mark"] == ("miss", "read by dispatched 'Join'")
    assert {mark, join} <= set(dispatched.nodes)
    for node in dispatched.nodes.values():
        assert set(node.dependencies) <= set(dispatched.nodes)


@pytest.mark.asyncio
async def test_a_node_bound_seed_answers_a_chat_op_under_reuse() -> None:
    """A chat seeded from a keyed node's records hits on an unchanged rerun."""
    rows = [{"id": "a", "q": "one", "seed": 7}, {"id": "b", "q": "two", "seed": 9}]
    graph = _chat_graph(rows)
    join, mark = _node(graph, "Join"), _node(graph, "Mark")
    spec = graph.nodes[join].data_spec
    assert isinstance(spec, DataFrame)
    seeds = NodeRead(node=mark, path="value.items.output.seed")
    spec = spec.model_copy(update={"seeds": seeds})
    graph.nodes[join] = replace(graph.nodes[join], data_spec=spec)
    storage = InMemoryJobStorage()
    stored = {"Mark": rows, "Join": ["A", "B"]}
    await _run(storage, _FOUR_OPS, stored, "req-1", graph=graph)
    plan, _ = _planner(storage).plan(
        runtime_graph=graph, round_number=0, job_id="req-2"
    )
    assert _decisions(plan)["Join"] == ("hit", None)


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
    rows = {"Join": _ROWS["Join"]}
    storage = InMemoryJobStorage()
    await _run(storage, _FOUR_OPS, rows, "req-1", graph=graph, items=items)
    plan, _ = _planned(storage, graph)
    assert _decisions(plan)["Join"] == ("hit", None), _decisions(plan)


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
    ran = {"Expand": ["ab", 3]}
    await _run(storage := InMemoryJobStorage(), _list_workflow("!"), ran, "req-1")
    graph = _runtime_graph(_list_workflow("?"))
    plan, dispatched = _planned(storage, graph)
    assert _decisions(plan) == dict(Expand=("hit", None), Consume=("miss", "no entry"))
    (consumer,) = dispatched.nodes.values()
    assert isinstance(consumer.data_spec, PythonData)
    (entry,), (hit,) = consumer.data_spec.plan, plan.hits.values()
    staged = ColumnPlanEntry(kind="stage", node="x")
    assert entry.values == served_column(hit, staged, "output", False) == ["ab", 3]
    run = PythonData(plan=[entry], code="", fn_name="", timeout_s=60)
    blob = python_step.runtime_blob()
    assert "inputs" not in python_step.flowmesh_spec("Consume", run, {blob.key: blob})


def test_a_served_output_item_is_wrapped_as_a_live_read_sees_it() -> None:
    """A stored item is wrapped back once, so a served read equals a live one."""
    staged, row = ColumnPlanEntry(kind="stage", node="x"), {"output": 1, "x": 2}
    entry = [_stored(row, "Final", "a")]
    output = staged.model_copy(update={"path": "output"})
    assert served_column(entry, output, "output", False) == [row]
    assert served_column(entry, staged, "output", False) == [row]
    flat = [_stored([["x", "y"]], "Expand")]
    flatten = staged.model_copy(update={"flatten": True})
    assert served_column(flat, flatten, "output", False) == ["x", "y"]
    scored, score = [_stored({"index": 0, "score": 0.5}, "E", "a")], {"path": "score"}
    assert served_column(scored, staged.model_copy(update=score), None, False) == [0.5]


@pytest.mark.asyncio
async def test_a_list_mode_output_hit_rebuilds_its_whole_list() -> None:
    """A list-mode Lambda's whole-list output exports its stored list unwrapped."""
    workflow = {**_list_workflow("!"), "outputs": [{"name": "x", "ref": "Expand"}]}
    await _run(
        storage := InMemoryJobStorage(), workflow, {"Expand": ["ab", 3]}, "req-1"
    )
    graph = _runtime_graph(workflow)
    (out,) = graph.output_node_map
    plan, dispatched = _planned(storage, graph)
    assert _decisions(plan)["Expand"] == ("hit", None)
    assert rebuilt_output(plan, out) == [json.dumps(["ab", 3])]
    assert out not in dispatched.nodes


@pytest.mark.asyncio
async def test_a_hit_output_survives_a_partial_run(
    monkeypatch: pytest.MonkeyPatch, job_storage: InMemoryJobStorage
) -> None:
    """Two outputs, one op changed: the hit output is rebuilt beside the miss."""
    outs = [{"name": "joined", "ref": "Join"}, {"name": "final", "ref": "Final"}]
    workflow = {**_FOUR_OPS, "outputs": outs}
    await _run(job_storage, workflow, _ROWS, "req-1")
    graph = _runtime_graph(_changed(workflow))
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
        _node(graph, name): [json.dumps(row) for row in _ROWS[name]]
        for name in ("Join", "Final")
    }


_STEPS = "def prompt(c):\n    import json\n    rows = [json.loads(v) for v in c[0]]\n"
_RETURN = "    return [{'id': r['id'], 'q': '/' + r['text']} for r in rows]\n"
_CODE_PROMPT = _STEPS + _RETURN
_LIST_PROMPT = _CODE_PROMPT.replace("'/' + r['text']", "[r['text']]")
_ROW_LIST_PROMPT = _STEPS + "    return [[r['text']] for r in rows]\n"
_KEY_CODE = "def key(c):\n    return [json.loads(v)['text'] for v in c[0]]\n"
_INPUT = [json.dumps({"id": "a", "text": "one"})]
_ROWS_KEY = {"node": "Rows", "path": "id"}
_CONFIG, _SPEAK = {"model": "m", "temperature": 0}, {"role": "user", "content": "hi"}
_READER, _ONE = {"role": "user", "content": "Read"}, [{"id": "a", "q": "/one"}]


def _lambda(op_id: str, src: list[str], fn: str, code: str) -> dict[str, Any]:
    """One row Lambda op as YAML names it."""
    return {"id": op_id, "op": "LambdaOp", "inputs": src, "fn_name": fn, "code": code}


_PATIENT_CODE = "lambda c: [json.loads(v) for v in c[0]]"
_PATIENT_ASK = "lambda c: [r['note'] for r in c[0]]"
_NOTED_CODE = "lambda c: [json.dumps(json.loads(v)['note']) for v in c[0]]"
_NOTES = {**_lambda("Notes", ["Rows"], "notes", _PATIENT_CODE), "mode": "row"}
_NOTES |= {"key": {"node": "Rows", "path": ["pid", "nctid"]}}
_NOTED = {**_lambda("Notes", ["Rows"], "notes", _NOTED_CODE), "mode": "row"}
_NOTED |= {"key": {"node": "Rows", "path": "pid"}}
_ASK = {**_lambda("Ask", ["Notes"], "ask", _PATIENT_ASK), "mode": "row"}
_ASK |= {"key": {"node": "Notes", "path": "nctid"}}
_PARSE = {"id": "Parsed", "op": "ParseOp", "inputs": ["Notes"]}


def _ask(op: str, inputs: list[str], **fields: Any) -> dict[str, Any]:
    """One model op over ``inputs`` as YAML names it."""
    return {"op": op, "inputs": inputs, "config": _CONFIG, **fields}


def _model_workflow(path: str | None, code: str) -> Any:
    """A keyed python prompt step read by one model op at ``path``."""
    prompt = {**_lambda("Prompt", ["Rows"], "prompt", code), "key": _ROWS_KEY}
    ask = _ask("LLMCompletionOp", ["Prompt"], prompt="Prompt")
    if path is not None:
        ask = _ask(
            "LLMChatOp",
            ["Prompt"],
            messages=[_SPEAK],
            rowwise_template="{v}",
            rowwise_columns=[{"label": "v", "node": "Prompt", "path": path}],
        )
    return {
        "name": "model",
        "inputs": {"Rows": _INPUT},
        "ops": [prompt, {"id": "Ask", **ask}],
        "outputs": [{"name": "answer", "ref": "Ask"}],
    }


_MODEL_CASES = [
    ("items.output.q", _CODE_PROMPT, _ONE, ["/one"]),
    (None, _CODE_PROMPT, _ONE, [json.dumps(row) for row in _ONE]),
    ("items.output.q", _LIST_PROMPT, [{"id": "a", "q": ["one"]}], [["one"]]),
    ("items.output", _ROW_LIST_PROMPT, [["one"]], None),
]


_SQL = {"type": "lumid", "mode": "sql", "template": "SELECT 1"}
_PARAM_SPEC = {**_SQL, "template": "SELECT 1 WHERE {k}", "params": [{"label": "k"}]}
_READ = {"id": "Read", "op": "DataRetrievalOp", "data_spec": _SQL}
_READ_ASK = {"id": "Ask"} | _ask("LLMChatOp", ["Read"], messages=[_READER])
_RAG: dict[str, Any] = {
    "name": "rag",
    "ops": [_READ, _READ_ASK],
    "outputs": [{"name": "x", "ref": "Ask"}],
}
_CHAIN_ROWS = {"Read": _ROWS["Mark"], "Join": _ROWS["Join"], "Final": _ROWS["Final"]}

_READ_CHAIN: dict[str, Any] = {
    **_RAG,
    "name": "read",
    "ops": [
        _READ,
        _lambda("Join", ["Read"], "second", _CODE_SECOND),
        _lambda("Final", ["Join"], "third", _THIRD_CODE),
    ],
    "outputs": [{"name": "final", "ref": "Final"}],
}


@pytest.mark.asyncio
@pytest.mark.parametrize(("path", "code", "stored", "items"), _MODEL_CASES)
async def test_a_served_prompt_feeds_a_model_op(
    path: str | None, code: str, stored: list[Any], items: list[Any] | None
) -> None:
    """A served prompt fills the payload; a row Lambda returning a list stays out."""
    workflow = _model_workflow(path, code)
    await _run(storage := InMemoryJobStorage(), workflow, {"Prompt": stored}, "req-1")
    graph = _runtime_graph(workflow)
    plan, dispatched = _planned(storage, graph)
    reason = "read by a non-literal op" if items is None else None
    assert _decisions(plan)["Prompt"] == ("miss" if reason else "hit", reason)
    ask = _node(graph, "Ask")
    spec = dispatched.nodes[ask].data_spec
    if items is None:
        assert spec == graph.nodes[ask].data_spec
        return
    column = spec if isinstance(spec, ListData) else spec.columns[0].data
    assert isinstance(column, ListData) and column.items == items


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("returned", "join", "final"),
    [
        (_ROWS["Mark"], ("hit", None), ("hit", None)),
        (
            [{**_ROWS["Mark"][0], "text": "X"}, _ROWS["Mark"][1]],
            ("miss", "no entry"),
            ("miss", "upstream miss"),
        ),
    ],
)
async def test_an_unversioned_read_rekeys_only_what_changed(
    returned: list[dict[str, str]], join: Any, final: Any
) -> None:
    """Consumers key on the read's value, so unchanged rows still hit."""
    graph = _runtime_graph(_READ_CHAIN)
    read = _node(graph, "Read")
    await _run(
        storage := InMemoryJobStorage(), _READ_CHAIN, _CHAIN_ROWS, "req-1", graph=graph
    )
    content = {read: NodeRecords(keyed=False, values=returned, keys=None)}
    plan, _ = _planned(storage, graph, content_values=content)
    assert _decisions(plan) == dict(
        Read=("miss", "unversioned upstream"), Join=join, Final=final
    )


@pytest.mark.asyncio
async def test_a_cancelled_request_cancels_its_read_sub_run(monkeypatch: Any) -> None:
    """A request cancelled while its read sub-run polls stops as cancelled."""
    graph = _runtime_graph(_READ_CHAIN)
    read, manager = _read_sub_run(monkeypatch, graph), FlowmeshRuntimeManager()

    async def _cancelled(*_: Any, **__: Any) -> bool:
        manager._cancelled_requests.add("req-2")
        return False

    monkeypatch.setattr(FlowmeshRuntimeManager, "update_task_status", _cancelled)
    with pytest.raises(RequestCancelledError):
        await manager._run_unversioned_reads(_reusing(graph), Schedule({"w": [read]}))
    assert manager._batch_workflow_id[("req-2", "batch-1")] == "wf-sub"


@pytest.mark.asyncio
async def test_a_consumer_a_reusing_job_writes_reuses() -> None:
    """A reusing job keys its consumer on the read's value, so the next hits."""
    storage, graph = InMemoryJobStorage(), _runtime_graph(_READ_CHAIN)
    marks = _ROWS["Mark"]
    content = {_node(graph, "Read"): NodeRecords(keyed=False, values=marks, keys=None)}
    plan, dispatched = _planned(storage, graph, content_values=content)
    assert _decisions(plan)["Join"] == ("miss", "no entry")
    joined = {"Join": marks}
    served, content_nodes = writer_records(plan, content), frozenset(content)
    await _run(
        storage,
        _READ_CHAIN,
        joined,
        "req-2",
        graph=dispatched,
        served=served,
        content_nodes=content_nodes,
    )
    again, _ = _planned(storage, graph, "req-3", content_values=content)
    assert _decisions(again)["Join"] == ("hit", None)


@pytest.mark.asyncio
async def test_a_read_that_is_an_output_comes_back(monkeypatch: Any) -> None:
    """A retrieval that is an output comes back from the sub-run's response."""
    workflow = {**_READ_CHAIN, "outputs": [{"name": "rows", "ref": "Read"}]}
    answer = {"items": [{"run_id": "r", "table": json.dumps(_ROWS["Mark"])}]}
    manager = FlowmeshRuntimeManager()
    graph = _runtime_graph(workflow)
    info = _reusing(graph)
    read = _read_sub_run(monkeypatch, graph, answer)
    await manager._run_unversioned_reads(info, Schedule({"w": [read]}))
    info.runtime_graph = replace(
        graph, output_node_map={}, nodes={}, node_order=[], output_paths={}
    )
    manager._reuse_read_outputs["req-2"] = await manager._aggregate_read_outputs(info)
    flat: dict[str, Any] = {}
    manager._add_reused_outputs(info, ReusePlan(), flat)
    assert flat == {read: [json.dumps(_ROWS["Mark"])]}


_CLASSIFY: dict[str, Any] = {
    "name": "classify",
    "inputs": {"Rows": _INPUT, "Cands": ["c1"]},
    "ops": [
        {**_lambda("Key", ["Rows"], "key", _KEY_CODE), "key": _ROWS_KEY},
        {"id": "Rank"}
        | _ask("ClassificationOp", ["Key", "Cands"], query="Key", candidates="Cands")
        | {"config": {"model": "org/reranker"}},
    ],
    "outputs": [{"name": "labels", "ref": "Rank"}],
}


@pytest.mark.asyncio
async def test_a_pairs_step_reads_a_served_lambda_at_its_output_path() -> None:
    """A keyed Lambda a pairs step reads is served at its item-relative path."""
    graph = _runtime_graph(_CLASSIFY)
    pairs = next(node for node in graph.nodes if "pair" in node)
    key = graph.nodes[pairs].dependencies[0]
    await _run(
        storage := InMemoryJobStorage(),
        _CLASSIFY,
        {"Key": ["one"]},
        "req-1",
        graph=graph,
        task_ids={key: "task-Key"},
    )
    plan, dispatched = _planned(storage, graph)
    pairs_spec = dispatched.nodes[pairs].data_spec
    assert key in plan.hits and isinstance(pairs_spec, PythonData)
    (stage,) = [entry for entry in pairs_spec.plan if entry.node == key]
    assert stage.path == "output" and stage.values == ["one"]


@pytest.mark.asyncio
async def test_a_history_edge_keeps_its_upstream() -> None:
    """A chat's history reads a prompt no stored value holds, so it runs."""
    config = GenerationConfig(model="m", temperature=0)
    stock = input_placeholder("Stock")
    first = LLMChatOp([OpMessage(role="user", content=stock)], config, True)
    second = LLMChatOp([OpMessage(role="user", content=first)], config)
    graph = RuntimeGraphBuilder().build(
        Graph.from_ops([as_output("answer", second)]).compile(Stock=["a"])
    )
    node = graph.dsl_to_runtime[first.id][0]
    await _run(
        storage := InMemoryJobStorage(),
        {},
        {"a": ["A"]},
        "req-1",
        graph=graph,
        task_ids={node: "task-a"},
    )
    plan, dispatched = _planned(storage, graph)
    assert plan.decisions[0].reason == "read by a non-literal op"
    assert set(dispatched.nodes) == set(graph.nodes)


_PARAM_READ: dict[str, Any] = {
    **_RAG,
    "ops": [
        _lambda("Key", [], "key", "def key(c):\n    return ['a']\n"),
        {**_READ, "inputs": ["Key"], "data_spec": _PARAM_SPEC},
        _READ_ASK,
    ],
}


@pytest.mark.asyncio
async def test_a_chat_reading_a_table_keeps_its_read_dispatched(
    job_storage: InMemoryJobStorage,
) -> None:
    """A chat reading a retrieval keeps the retrieval dispatched."""
    graph = _runtime_graph(_RAG)
    read, chat = _node(graph, "Read"), _node(graph, "Ask")
    table = {"table": {"df": json.dumps({"id": {"0": "a"}})}}
    content = {read: NodeRecords(keyed=False, values=[table], keys=None)}
    plan, dispatched = _planned(job_storage, graph, content_values=content)
    chat_spec = dispatched.nodes[chat].data_spec
    assert read not in plan.content_served
    assert isinstance(chat_spec, GraphTemplateData)
    assert chat_spec.template.columns[0].node == read


@pytest.mark.asyncio
async def test_a_read_with_an_upstream_stays_in_the_main_graph(
    monkeypatch: pytest.MonkeyPatch, job_storage: InMemoryJobStorage
) -> None:
    """A read with another upstream runs in the main graph as a miss."""
    graph = _runtime_graph(_PARAM_READ)
    key, read = _node(graph, "Key"), _node(graph, "Read")
    assert key in graph.nodes[read].dependencies

    async def _never(*_: Any) -> tuple[str, list[str]]:
        raise AssertionError("a read with an upstream stays in the main graph")

    monkeypatch.setattr(FlowmeshRuntimeManager, "_submit_workflow", _never)
    reads = await FlowmeshRuntimeManager()._run_unversioned_reads(
        _reusing(graph), Schedule({"w": [key, read]})
    )
    assert reads == {}


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
    plan, _ = _planned(InMemoryJobStorage(), graph)
    order = [decision.node for decision in plan.decisions]
    assert order.index(mark) < order.index(join)
    assert order.index("b") < order.index("s")
    assert order.index("s") < order.index(join)
