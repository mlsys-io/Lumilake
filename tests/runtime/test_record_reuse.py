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
from lumilake_server.ops import LLMChatOp, OpMessage, as_output, input_placeholder
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
        node_task_map=task_ids
        or {_node(graph, name): f"task-{name}" for name in results},
        task_worker={},
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


def _read_sub_run(
    monkeypatch: pytest.MonkeyPatch, graph: Any, response: Any = None
) -> str:
    """Patch a read sub-run over ``graph``: one read task, done, answering it."""
    read = _node(graph, "Read")
    serve(monkeypatch, {"task-read": "PENDING" if response is None else "DONE"})
    answers = {
        "_submit_workflow": ("wf-sub", ["task-read"]),
        "_resolve_task_node_maps": ({"task-read": read}, {read: "task-read"}),
        "_read_task_result": response,
    }
    for name, answer in answers.items():
        monkeypatch.setattr(
            FlowmeshRuntimeManager, name, AsyncMock(return_value=answer)
        )
    monkeypatch.setattr(envs, "LUMILAKE_POLL_INTERVAL_SECONDS", 0.0)
    return read


def _decisions(plan: ReusePlan) -> dict[str, tuple[str, str | None]]:
    return {d.op: (d.decision, d.reason) for d in plan.decisions}


def _planner(storage: InMemoryJobStorage) -> ReusePlanner:
    return ReusePlanner(storage=storage, logger=_LOG)


def _planned(
    storage: InMemoryJobStorage, graph: Any, job_id: str = "req-2", **kw: Any
) -> tuple[ReusePlan, Any]:
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
    plan, dispatched = _planned(storage, graph)
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


_STEPS = "def prompt(c):\n    import json\n    rows = [json.loads(v) for v in c[0]]\n"
_RETURN = "    return [{'id': r['id'], 'q': '/' + r['text']} for r in rows]\n"
_CODE_PROMPT = _STEPS + _RETURN
_LIST_PROMPT = _CODE_PROMPT.replace("'/' + r['text']", "[r['text']]")
_ROW_LIST_PROMPT = _STEPS + "    return [[r['text']] for r in rows]\n"
_STRING_PROMPT = _CODE_PROMPT.replace("'/' + r['text']", "json.dumps({'a': r['id']})")
_KEY_CODE = "def key(c):\n    return [json.loads(v)['text'] for v in c[0]]\n"
_INPUT = [json.dumps({"id": "a", "text": "one"})]
_CODE_ROW = "def ids(c):\n    import json\n    return json.loads(c[0])\n"
_ROWS_KEY = {"node": "Rows", "path": "id"}
_ONE, _LISTED = [{"id": "a", "q": "/one"}], [{"id": "a", "q": ["one"]}]
_STRING_ONE = {"id": "a", "q": '{"a": "a"}'}
_CONFIG = {"model": "m", "temperature": 0}
_IDS_KEY = {"node": "Ids", "path": "id"}


def _lambda(op_id: str, src: list[str], fn: str, code: str) -> dict[str, Any]:
    """One row Lambda op as YAML names it."""
    step = {"op": "LambdaOp", "inputs": src, "fn_name": fn, "code": code}
    return {"id": op_id, **step}


def _ask(op: str, inputs: list[str], **fields: Any) -> dict[str, Any]:
    """One model op over ``inputs`` as YAML names it."""
    return {"op": op, "inputs": inputs, "config": _CONFIG, **fields}


def _model_workflow(
    path: str | None, code: str, key: Any = _ROWS_KEY
) -> dict[str, Any]:
    """A python prompt step keyed by ``key``, read by one model op at ``path``."""
    src = ["Ids"] if key is _IDS_KEY else ["Rows"]
    prompt = {**_lambda("Prompt", src, "prompt", code), "key": key}
    ids = {**_lambda("Ids", ["Rows"], "ids", _CODE_ROW), "key": _ROWS_KEY}
    ops = [ids, prompt] if key is _IDS_KEY else [prompt]
    ask = _ask("LLMCompletionOp", ["Prompt"], prompt="Prompt")
    if path is not None:
        ask = _ask(
            "LLMChatOp",
            ["Prompt"],
            messages=[{"role": "user", "content": "hi"}],
            rowwise_template="{v}",
            rowwise_columns=[{"label": "v", "node": "Prompt", "path": path}],
        )
    return {
        "name": "model",
        "inputs": {"Rows": _INPUT},
        "ops": [*ops, {"id": "Ask", **ask}],
        "outputs": [{"name": "answer", "ref": "Ask"}],
    }


_MODEL_CASES = [
    ("items.output.q", _CODE_PROMPT, _ONE, ["/one"], _ROWS_KEY),
    (None, _CODE_PROMPT, _ONE, _ONE, _ROWS_KEY),
    ("items.output.q", _LIST_PROMPT, _LISTED, [["one"]], _ROWS_KEY),
    ("items.output", _ROW_LIST_PROMPT, [["one"]], None, None),
    ("items.output", _ROW_LIST_PROMPT, ['["one"]'], None, _IDS_KEY),
    ("items.output.q.a", _STRING_PROMPT, [_STRING_ONE], None, _ROWS_KEY),
]


_SQL = {"type": "lumid", "mode": "sql", "template": "SELECT 1"}
_PARAM_SPEC = {**_SQL, "template": "SELECT 1 WHERE {k}", "params": [{"label": "k"}]}
_READ = {"id": "Read", "op": "DataRetrievalOp", "data_spec": _SQL}
_READ_ASK = {"id": "Ask"} | _ask(
    "LLMChatOp", ["Read"], messages=[{"role": "user", "content": "Read"}]
)
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
@pytest.mark.parametrize(("path", "code", "stored", "items", "key"), _MODEL_CASES)
async def test_a_served_prompt_feeds_a_model_op(
    path: str | None,
    code: str,
    stored: list[Any],
    items: list[Any] | None,
    key: dict[str, str] | None,
) -> None:
    """A served prompt fills the payload; a row Lambda returning a list stays out."""
    workflow = _model_workflow(path, code, key)
    rows = {"Prompt": stored, "Ask": [{"text": "A"}]}
    if key is _IDS_KEY:
        rows["Ids"] = [{"id": "a", "text": "one"}]
    await _run(storage := InMemoryJobStorage(), workflow, rows, "req-1")
    graph = _runtime_graph(workflow)
    plan, dispatched = _planned(storage, graph)
    node = _node(graph, "Ask")
    assert _decisions(plan)["Prompt"] == (
        ("miss", "read by a non-literal op") if items is None else ("hit", None)
    )
    spec = dispatched.nodes[node].data_spec
    if items is None:
        assert spec == graph.nodes[node].data_spec
        return
    if isinstance(spec, ListData):
        assert spec.items == items
    else:
        assert isinstance(spec, DataFrame)
        column = spec.columns[0]
        assert column.data is not None and isinstance(column.data, ListData)
        assert column.data.items == items


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
    assert _decisions(plan) == {
        "Read": ("miss", "unversioned upstream"),
        "Join": join,
        "Final": final,
    }


@pytest.mark.asyncio
async def test_a_cancelled_request_cancels_its_read_sub_run(monkeypatch: Any) -> None:
    """A request cancelled while its read sub-run polls stops as cancelled."""
    graph = _runtime_graph(_READ_CHAIN)
    read, manager = _read_sub_run(monkeypatch, graph), FlowmeshRuntimeManager()

    async def _cancel_request(self: FlowmeshRuntimeManager, request_id: str) -> None:
        async with self._task_status_lock:
            self._cancelled_requests.add(request_id)

    async def _cancelling_status(*_: Any, **__: Any) -> bool:
        await manager.cancel_request("req-2")
        return False

    monkeypatch.setattr(
        FlowmeshRuntimeManager, "update_task_status", _cancelling_status
    )
    monkeypatch.setattr(FlowmeshRuntimeManager, "cancel_request", _cancel_request)

    with pytest.raises(RequestCancelledError):
        await manager._run_unversioned_reads(_reusing(graph), Schedule({"w": [read]}))
    assert manager._batch_workflow_id[("req-2", "batch-1")] == "wf-sub"


@pytest.mark.asyncio
async def test_a_consumer_a_reusing_job_writes_reuses() -> None:
    """A reusing job keys its consumer on the read's value, so the next hits."""
    storage, graph = InMemoryJobStorage(), _runtime_graph(_READ_CHAIN)
    read = _node(graph, "Read")
    content = {read: NodeRecords(keyed=False, values=_ROWS["Mark"], keys=None)}
    plan, dispatched = _planned(storage, graph, content_values=content)
    assert _decisions(plan)["Join"] == ("miss", "no entry")
    await _run(
        storage,
        _READ_CHAIN,
        {"Join": _ROWS["Join"]},
        "req-2",
        graph=dispatched,
        served=writer_records(plan, content),
        content_nodes=frozenset(plan.content),
    )
    again, _ = _planned(storage, graph, "req-3", content_values=content)
    assert _decisions(again)["Join"] == ("hit", None)


@pytest.mark.asyncio
async def test_a_read_that_is_an_output_comes_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retrieval that is an output comes back from the sub-run's response."""
    workflow = {**_READ_CHAIN, "outputs": [{"name": "rows", "ref": "Read"}]}
    response = {"items": [{"run_id": "r", "table": json.dumps(_ROWS["Mark"])}]}
    graph = _runtime_graph(workflow)
    read = _read_sub_run(monkeypatch, graph, response)
    manager, request_info = FlowmeshRuntimeManager(), _reusing(graph)
    await manager._run_unversioned_reads(request_info, Schedule({"w": [read]}))
    request_info.runtime_graph = replace(
        graph, output_node_map={}, nodes={}, node_order=[], output_paths={}
    )
    outputs = await manager._aggregate_read_outputs(request_info)
    manager._reuse_read_outputs["req-2"] = outputs
    flat: dict[str, Any] = {}
    manager._add_reused_outputs(request_info, ReusePlan(), flat)
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
    pairs = next(node for node in graph.nodes if "pairs" in node)
    (key,) = graph.nodes[pairs].dependencies
    await _run(
        storage := InMemoryJobStorage(),
        _CLASSIFY,
        {"Key": ["one"]},
        "req-1",
        graph=graph,
        task_ids={key: "task-Key"},
    )
    plan, dispatched = _planned(storage, graph)
    assert key in plan.hits and pairs in dispatched.nodes
    pairs_spec = dispatched.nodes[pairs].data_spec
    assert isinstance(pairs_spec, PythonData)
    (stage,) = [entry for entry in pairs_spec.plan if entry.node == key]
    assert stage.path == "output" and stage.values == ["one"]


@pytest.mark.asyncio
async def test_a_history_edge_keeps_its_upstream() -> None:
    """A chat's history reads a prompt no stored value holds, so it runs."""
    config = GenerationConfig(model="m", temperature=0)
    stock = input_placeholder("Stock")
    first = LLMChatOp(
        [OpMessage(role="user", content=stock)], config=config, return_history=True
    )
    second = LLMChatOp([OpMessage(role="user", content=first)], config=config)
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
    assert read not in plan.content_served
    chat_spec = dispatched.nodes[chat].data_spec
    assert isinstance(chat_spec, GraphTemplateData)
    (column,) = chat_spec.template.columns
    assert column.node == read


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
