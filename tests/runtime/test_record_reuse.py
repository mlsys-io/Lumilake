"""Whole-op reuse: a reusing plan answers an op whose every record the store
already holds and dispatches the rest."""

import atexit
import json
import logging
import posixpath
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import yaml
from flowmesh.models.tasks import TaskStatusPage

from lumilake import envs
from lumilake_server.common import GenerationConfig
from lumilake_server.data_specs import (
    ColumnPlanEntry,
    DataFrame,
    GraphTemplateData,
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
    digest_of,
    is_marker,
)
from lumilake_server.runtime.record_writer import (
    NodeRecords,
    RecordWriter,
    read_entry,
    read_index,
    resolve_plans,
)
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
_PATIENT_OPS: list[dict[str, Any]] = [
    {
        "id": "Notes",
        "op": "LambdaOp",
        "inputs": ["Rows"],
        "mode": "row",
        "fn_name": "notes",
        "key": {"node": "Rows", "path": ["pid", "nctid"]},
        "code": "lambda c: [json.loads(v) for v in c[0]]",
    },
    {
        "id": "Ask",
        "op": "LambdaOp",
        "inputs": ["Notes"],
        "mode": "row",
        "fn_name": "ask",
        "key": {"node": "Notes", "path": "nctid"},
        "code": "lambda c: [r['note'] for r in c[0]]",
    },
]


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
        spec_digest=digest_of({"op": op}),
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
    fail: Callable[[str, Mapping[str, Any]], str | None] | None = None,
) -> None:
    """Write one job's finished tasks the way dispatch does; ``items`` are raw
    items, and ``fail`` names the records whose task failed, written as a marker
    carrying the reason it returns."""
    graph = _runtime_graph(workflow) if graph is None else graph
    names = _names(graph)
    raw = items or {}
    if fail is not None:
        raw = {
            n: [_markered(names[n], item, fail) for item in v] for n, v in raw.items()
        }
    tasks = {
        f"task-{n}": {"items": [{"output": r} for r in v]} for n, v in results.items()
    }
    tasks |= {f"task-{n}": {"items": v} for n, v in raw.items()}
    await RecordWriter(storage=storage, enabled=True, logger=_LOG).write(
        job_id=job_id,
        member_job_ids=[job_id],
        round_number=0,
        runtime_graph=graph,
        node_task_map=task_ids
        or {
            node: f"task-{name}"
            for node, name in names.items()
            if name in {**results, **raw}
        },
        results=tasks,
        served=served,
        content_nodes=content_nodes,
    )


def _markered(
    op: str,
    item: Mapping[str, Any],
    fail: Callable[[str, Mapping[str, Any]], str | None],
) -> Mapping[str, Any]:
    """One task item with its output replaced by a marker when ``fail`` names it."""
    reason = fail(op, item)
    if reason is None:
        return item
    return {**item, "output": {"_fail": {"op": op, "reason": reason}}}


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


class _Cycle:
    """One storage across the three runs a reuse test needs: a full run that
    stores, a changed run that plans, computes its dispatched steps as a worker
    does and stores, and an unchanged run that plans again."""

    def __init__(self, storage: InMemoryJobStorage) -> None:
        self.storage = storage
        self.run = 0
        self.items: dict[str, list[Any]] = {}

    async def full(
        self,
        workflow: dict[str, Any],
        fail: Callable[[str, Mapping[str, Any]], str | None] | None = None,
    ) -> Any:
        """Run 1: every python step of the workflow, run as its worker does over
        its upstreams, written the way dispatch writes it. ``fail(op, item)``
        names the records whose task failed, which the writer receives as a
        marker carrying the reason it returns."""
        self.run += 1
        self.items = self._steps(_runtime_graph(workflow))
        await self._write(_runtime_graph(workflow), workflow, fail=fail)
        return self.plan(workflow)

    def plan(self, workflow: dict[str, Any]) -> Any:
        """Plan ``workflow`` against what the runs so far stored."""
        return _planned(
            self.storage, _runtime_graph(workflow), job_id=f"req-{self.run + 1}"
        )

    def serves(self, workflow: dict[str, Any], op: str, whole: bool = False) -> None:
        """Assert the next plan answers ``op`` whole; ``whole`` adds that it
        dispatches nothing."""
        plan, dispatched = self.plan(workflow)
        node = _node(_runtime_graph(workflow), op)
        assert plan.hits.get(node), _decisions(plan)
        assert node not in dispatched.nodes
        if whole:
            assert dispatched.nodes == {}

    async def rerun(self, workflow: dict[str, Any]) -> Any:
        """Run 2: plan, run every step the plan left in the graph over its
        upstreams as its worker does, and store this run's own items under the
        live keys."""
        plan, dispatched = self.plan(workflow)
        self.run += 1
        items = self._steps(dispatched)
        self.items |= items
        await self._write(dispatched, workflow, writer_records(plan, {}))
        return plan, dispatched, items

    async def _write(
        self,
        graph: Any,
        workflow: dict[str, Any],
        served: dict[str, Any] | None = None,
        fail: Callable[[str, Mapping[str, Any]], str | None] | None = None,
    ) -> None:
        await _run(
            self.storage,
            workflow,
            {},
            f"req-{self.run}",
            graph=graph,
            served=served,
            items=self.items,
            task_ids={node: f"task-{node}" for node in self.items},
            fail=fail,
        )

    def _steps(self, graph: Any) -> dict[str, list[Any]]:
        """Every python step of ``graph``, run in dependency order as its worker
        does, by node id."""
        ran: dict[str, list[Any]] = {}
        for node in graph.topological_order():
            if node not in graph.nodes:
                continue
            op = graph.nodes[node]
            if not isinstance(op.data_spec, PythonData):
                continue
            upstreams = {dep: ran[dep] for dep in op.dependencies if dep in ran}
            ran[node] = _run_step(op, upstreams)
        return ran


def _names(graph: Any) -> dict[str, str]:
    """Each dispatched node's op name."""
    return {node: graph.op_names.get(node, node) for node in graph.nodes}


def _run_step(op: Any, inputs: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Run one dispatched op's generated step as its worker does, over upstreams,
    staging each upstream's items as a finished task reads them back."""
    plan = op.data_spec.plan
    blobs = [python_step.runtime_blob(), *python_step.literal_blobs(plan).values()]
    spec = python_step.flowmesh_spec(
        op.node_id, op.data_spec, {b.key: b for b in blobs}
    )
    for mounted in blobs:
        relative = mounted.mount_path.removeprefix(_BLOBS_ROOT).lstrip("/")
        path = _mount_root() / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(mounted.body)
    staged: dict[str, str] = {}
    for node, items in inputs.items():
        directory = Path(tempfile.mkdtemp(dir=_mount_root()))
        (directory / "results.json").write_text(
            json.dumps({"task_id": "t", "result": {"items": items}})
        )
        staged[node] = str(directory)
    namespace: dict[str, Any] = {}
    exec(spec["code"].replace(_BLOBS_ROOT, str(_mount_root())), namespace)
    return namespace["main"](staged)["items"]


_BLOBS_ROOT = posixpath.dirname(python_step.TASK_RUNTIME_MOUNT)
_MOUNT: dict[str, Path] = {}


def _mount_root() -> Path:
    """A mount root under the system temp dir for the blobs a generated step reads."""
    if "root" not in _MOUNT:
        root = Path(tempfile.mkdtemp(prefix="lumilake-step-mounts-"))
        atexit.register(shutil.rmtree, root, True)
        _MOUNT["root"] = root
    return _MOUNT["root"]


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
        "Final": ("miss", "spec changed"),
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
        (
            [{"id": "a", "q": "X"}, {"id": "b", "q": "two"}],
            None,
            ("miss", "partial records"),
        ),
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
    workflow = {
        "name": "patients",
        "inputs": {"Rows": [json.dumps(row) for row in rows]},
        "ops": _PATIENT_OPS,
        "outputs": [{"name": "asked", "ref": "Ask"}],
    }
    changed = {
        **workflow,
        "ops": [_PATIENT_OPS[0], {**_PATIENT_OPS[1], "code": "lambda c: c[1]"}],
    }
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
    assert _decisions(plan)["Parsed"] == ("miss", "spec changed"), _decisions(plan)
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
    assert _decisions(plan)["Parsed"] == ("miss", "spec changed"), _decisions(plan)
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


def _op(op_id: str, src: list[str], fn: str, code: str, mode: str) -> dict[str, Any]:
    """One Lambda op in the given mode as YAML names it."""
    return {**_lambda(op_id, src, fn, code), "mode": mode}


def _pipeline(
    name: str,
    ops: list[dict[str, Any]],
    output: str,
    rows: list[str] | None = None,
) -> dict[str, Any]:
    """A one-input workflow whose ops are ``ops`` and whose output is ``output``."""
    return {
        "name": name,
        "inputs": {"Rows": rows or ['{"id": "p1"}', '{"id": "p2"}']},
        "ops": ops,
        "outputs": [{"name": name, "ref": output}],
    }


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
    if items is None:
        assert dispatched.nodes[_node(graph, "Ask")].data_spec == (
            graph.nodes[_node(graph, "Ask")].data_spec
        )
        return
    step = next(
        node
        for node in dispatched.nodes
        if plan.pass_throughs.get(node) == _node(graph, "Prompt")
    )
    entry, positions = dispatched.nodes[step].data_spec.plan
    assert [json.loads(row["output"]) for row in entry.values] == stored
    assert positions.values == list(range(len(entry.values)))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("returned", "join", "final"),
    [
        (_ROWS["Mark"], ("hit", None), ("hit", None)),
        (
            [{**_ROWS["Mark"][0], "text": "X"}, _ROWS["Mark"][1]],
            ("miss", "input changed"),
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


def _keyed_read_graph() -> Any:
    """A keyed sql retrieval whose consumer reads one row per ``case_id``."""
    workflow = {
        "name": "keyed-read",
        "ops": [
            {
                "id": "Read",
                "op": "DataRetrievalOp",
                "data_spec": {
                    **_SQL,
                    "template": "SELECT case_id FROM cases",
                    "key": {"column": "case_id"},
                },
            },
            _READ_ASK,
        ],
        "outputs": [{"name": "answer", "ref": "Ask"}],
    }
    return _runtime_graph(workflow)


_KEYED_TAG: dict[str, Any] = {
    "name": "keyed-tag",
    "ops": [
        {
            **_READ,
            "data_spec": {
                **_SQL,
                "template": "SELECT case_id, t FROM cases",
                "key": {"column": "case_id"},
            },
        },
        {
            "id": "Tag",
            "op": "LambdaOp",
            "inputs": ["Read"],
            "mode": "row",
            "fn_name": "tag",
            "code": (
                "def tag(c):\n    import json\n"
                "    return json.loads(c[0][0])['t'].upper()\n"
            ),
            "key": {"node": "Read", "path": "case_id"},
        },
    ],
    "outputs": [{"name": "tag", "ref": "Tag"}],
}


def _keyed_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A keyed retrieval's response, one table column set per row key."""
    columns: dict[str, dict[str, str]] = {}
    for position, row in enumerate(rows):
        for name, value in row.items():
            columns.setdefault(name, {})[str(position)] = value
    return {"items": [{"table": {"df": json.dumps(columns)}}]}


def _keyed_content(
    read: str,
    rows: list[dict[str, Any]],
    *,
    per_row: bool = True,
    key: str = "case_id",
) -> dict[str, Any]:
    """A keyed read re-run, handing one item per record or one holding every one."""
    return {
        read: NodeRecords(
            keyed=True,
            values=rows,
            keys=[r[key] for r in rows],
            items=(
                [_keyed_table([row])["items"][0] for row in rows]
                if per_row
                else _keyed_table(rows)["items"]
            ),
            content=True,
        )
    }


@pytest.mark.asyncio
async def test_a_keyed_read_versions_its_consumer_per_row() -> None:
    """A keyed retrieval is versioned per row, so a changed row's consumer
    misses on that key alone while every other key's record still hits. A read
    that hands one item holding every record keys none of them, so its reader is
    computed whole rather than served from one row's identity."""
    rows = [{"case_id": "c1", "t": "x"}, {"case_id": "c2", "t": "y"}]
    changed = [rows[0], {"case_id": "c2", "t": "y!"}]
    graph = _runtime_graph(_KEYED_TAG)
    read = _node(graph, "Read")
    storage = InMemoryJobStorage()
    await _run(
        storage,
        _KEYED_TAG,
        {"Tag": ["X", "Y"]},
        "req-1",
        items={"Read": _keyed_table(rows)["items"]},
    )
    plan, _ = _planned(storage, graph, content_values=_keyed_content(read, rows))
    assert _decisions(plan)["Tag"] == ("hit", None), _decisions(plan)
    plan, _ = _planned(storage, graph, content_values=_keyed_content(read, changed))
    assert _decisions(plan)["Tag"] == ("miss", "partial records"), _decisions(plan)
    whole = _keyed_content(read, changed, per_row=False)
    plan, _ = _planned(storage, graph, content_values=whole)
    assert _decisions(plan)["Tag"] == ("miss", "unkeyed row"), _decisions(plan)


@pytest.mark.asyncio
async def test_a_reuse_sub_run_reads_a_keyed_retrieval_back_by_its_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A keyed retrieval re-ran in the sub-run reads back one record per key,
    so its keyed consumer decides on a key instead of an unkeyed row."""
    graph = _keyed_read_graph()
    read = _node(graph, "Read")
    response = {
        "items": [{"table": {"df": json.dumps({"case_id": {"0": "c1", "1": "c2"}})}}]
    }
    _read_sub_run(monkeypatch, graph, response)
    records = await FlowmeshRuntimeManager()._run_unversioned_reads(
        _reusing(graph), Schedule({"w": [read]})
    )
    assert records[read].keyed
    assert records[read].keys == ["c1", "c2"]


@pytest.mark.asyncio
async def test_a_reuse_sub_run_refuses_a_repeated_retrieval_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated key in a re-ran retrieval's rows fails its consumer's job the
    same way the full run's arrival check does."""
    graph = _keyed_read_graph()
    read = _node(graph, "Read")
    response = {
        "items": [{"table": {"df": json.dumps({"case_id": {"0": "c1", "1": "c1"}})}}]
    }
    _read_sub_run(monkeypatch, graph, response)
    with pytest.raises(RuntimeError, match="repeated at row 0 of item 0 and row 1"):
        await FlowmeshRuntimeManager()._run_unversioned_reads(
            _reusing(graph), Schedule({"w": [read]})
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


_TWO_READS: dict[str, Any] = {
    "name": "two-reads",
    "ops": [
        {
            **_READ,
            "data_spec": {
                **_SQL,
                "template": "SELECT tid, patient, nctid FROM trials",
                "key": {"column": "tid"},
            },
        },
        {
            "id": "Scores",
            "op": "DataRetrievalOp",
            "data_spec": {
                **_SQL,
                "template": "SELECT sid, patient, nctid, score FROM scores",
                "key": {"column": "sid"},
            },
        },
        {
            "id": "Merge",
            "op": "JoinOp",
            "inputs": ["Read", "Scores"],
            "by": ["patient", "nctid"],
        },
    ],
    "outputs": [{"name": "merged", "ref": "Merge"}],
}


def _rows_of(plan: ReusePlan, op: str) -> list[tuple[str | None, str, str | None]]:
    """Each record's key, decision and reason for one op's decision."""
    (decision,) = [d for d in plan.decisions if d.op == op]
    return [(row.record, row.decision, row.reason) for row in decision.outcomes]


@pytest.mark.asyncio
async def test_a_join_over_two_re_ran_reads_keys_its_rows_by_their_by_values() -> None:
    """Both inputs are re-read in this run and hold their rows in different
    orders, so each join row is keyed by its own ``by`` values rather than by a
    first-input row's key, and an unchanged rerun hits every one of them."""
    graph = _runtime_graph(_TWO_READS)
    read, scores = (_node(graph, name) for name in ("Read", "Scores"))
    trials = [
        {"tid": "t1", "patient": "p1", "nctid": "N1"},
        {"tid": "t2", "patient": "p1", "nctid": "N2"},
    ]
    scored = [
        {"sid": "s2", "patient": "p1", "nctid": "N2", "score": 2},
        {"sid": "s1", "patient": "p1", "nctid": "N1", "score": 1},
    ]
    content = {
        **_keyed_content(read, trials, key="tid", per_row=False),
        **_keyed_content(scores, scored, key="sid", per_row=False),
    }
    storage = InMemoryJobStorage()
    plan, dispatched = _planned(storage, graph, content_values=content)
    assert _decisions(plan)["Merge"] == ("miss", "no entry")
    merged = [{**row, "score": s["score"]} for row, s in zip(trials, scored[::-1])]
    await _run(
        storage,
        _TWO_READS,
        {"Merge": merged},
        "req-2",
        graph=dispatched,
        served=writer_records(plan, content),
        content_nodes=frozenset(content),
    )
    again, _ = _planned(storage, graph, "req-3", content_values=content)
    assert _rows_of(again, "Merge") == [
        ('["p1","N1"]', "hit", None),
        ('["p1","N2"]', "hit", None),
    ]


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


def _tag_workflow(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One keyed row step over a literal input, so its own step reads the rows."""
    tag = _op("Tag", ["Rows"], "tag", "lambda c: c[0]['t'].upper()", "row")
    tag["key"] = {"node": "Rows", "path": "id"}
    return _pipeline("tag", [tag], "Tag", [json.dumps(row) for row in rows])


@pytest.mark.asyncio
async def test_a_partial_runs_spec_digest_equals_a_full_runs() -> None:
    """A literal column's extra fields, and the served slot beside it, stay out of
    one rule, so a partial run's op digests the same op a full run would."""
    rows = [{"id": "a", "t": "x"}, {"id": "b", "t": "y"}]
    amended = [rows[0], {"id": "b", "t": "y!"}]
    storage = InMemoryJobStorage()
    stored = _with_tag_entry(_tag_workflow(rows))
    await _run(
        storage,
        _tag_workflow(rows),
        {"Tag": ["X", "Y"]},
        "req-1",
        graph=stored,
    )
    graph = _with_tag_entry(_tag_workflow(amended))
    tag = _node(graph, "Tag")
    plan, dispatched = _planned(storage, graph)
    assert _decisions(plan)["Tag"] == ("miss", "partial records"), _decisions(plan)
    full = resolve_plans(graph, graph.topological_order())[tag].spec_digest
    partial = resolve_plans(dispatched, dispatched.topological_order())[tag].spec_digest
    assert partial == full


def _with_tag_entry(workflow: dict[str, Any]) -> Any:
    """The tag graph with a labelled, flattening literal entry a slot could clone."""
    graph = _runtime_graph(workflow)
    tag = _node(graph, "Tag")
    spec = graph.nodes[tag].data_spec
    assert isinstance(spec, PythonData)
    entry = spec.plan[0].model_copy(update={"label": "v", "flatten": True})
    graph.nodes[tag] = replace(
        graph.nodes[tag], data_spec=spec.model_copy(update={"plan": [entry]})
    )
    return graph


_REDUCE_OPS: list[dict[str, Any]] = [
    {**_op("Tag", ["Rows"], "tag", "lambda row: row[0]", "row")}
    | {"key": {"node": "Rows", "path": "id"}},
    {
        **_op(
            "Rank", ["Tag"], "rank", "lambda rows: [r[0]['id'] for r in rows]", "reduce"
        ),
        "by": "patient",
    },
]
_REDUCE_ROWS = [
    {"id": "t1", "patient": "p1", "meta": {"p": 1}},
    {"id": "t2", "patient": "p2", "meta": {"p": 2}},
    {"id": "t3", "patient": "p3", "meta": {"p": 3}},
]


def _reduce_workflow(
    rows: list[dict[str, Any]], by: str = "patient", changed: bool = False
) -> dict[str, Any]:
    """The reduce fixture over ``rows``, grouped by ``by``, with a per-row
    literal list beside it so every column must stay at live width; ``changed``
    alters the last literal, which changes one group's inputs alone."""
    tag, rank = _REDUCE_OPS
    weights = ["w1", "w2", "w3!"] if changed else ["w1", "w2", "w3"]
    return _pipeline(
        "reduce-reuse",
        [
            tag,
            {"id": "Weights", "op": "DataOp", "inputs": [], "data": weights},
            {**rank, "by": by, "inputs": ["Tag", "Weights"]},
        ],
        "Rank",
        [json.dumps(row) for row in rows],
    )


@pytest.mark.parametrize("by", ["patient", "meta"])
@pytest.mark.asyncio
async def test_a_reduce_cycle_merges_one_changed_group(by: str) -> None:
    """A reduce over a served input whose last per-row literal changes: run 2
    recomputes that row's group only and emits every live group in group order,
    so its output equals a full run's and it stores that group, and run 3 serves
    the reduce whole. ``by`` spells a group key as a string and as an object,
    which take different spelling paths, and the literal stays at live width
    beside the served column. Covers findings 1, 4, 5 and 8."""
    rows = _REDUCE_ROWS
    cycle = _Cycle(InMemoryJobStorage())
    await cycle.full(_reduce_workflow(rows, by))
    changed = _reduce_workflow(rows, by, changed=True)
    plan, _, items = await cycle.rerun(changed)
    rank_id = _node(_runtime_graph(_reduce_workflow(rows, by)), "Rank")
    assert _decisions(plan)["Rank"] == ("miss", "partial group records"), _decisions(
        plan
    )
    slots = plan.partial[rank_id].groups
    assert slots is not None
    assert [slot.served is not None for slot in slots] == [True, True, False]
    full_graph = _runtime_graph(changed)
    tag_id = _node(full_graph, "Tag")
    full = _run_step(
        full_graph.nodes[rank_id],
        {tag_id: _run_step(full_graph.nodes[tag_id], {})},
    )
    assert [(item["key"], item["output"]) for item in items[rank_id]] == [
        (item["key"], item["output"]) for item in full
    ]
    cycle.serves(changed, "Rank")
    moved = [*rows[:2], {**rows[2], "id": "t3b"}]
    plan, _ = _planned(
        cycle.storage, _runtime_graph(_reduce_workflow(moved, by, changed=True))
    )
    assert _decisions(plan)["Rank"] == ("miss", "input dispatched"), _decisions(plan)


@pytest.mark.parametrize("variant", ["changed_parent", "empty_parent"])
@pytest.mark.asyncio
async def test_an_expand_cycle_decides_each_parent(variant: str) -> None:
    """Two expand cycle variants. A changed parent's row is partially served, so
    the expand misses whole as an upstream miss and a third run over the same
    rows answers it: the level-by-level limit beside the expand's hit path. A
    parent that expands to nothing is served on an unchanged rerun instead."""
    if variant == "changed_parent":
        rows = ['{"id": "p1"}', '{"id": "p2"}']
        changed = ['{"id": "p1"}', '{"id": "p2", "v": 2}']
        cycle = _Cycle(InMemoryJobStorage())
        await cycle.full(_expand_workflow(rows))
        plan, _, _ = await cycle.rerun(_expand_workflow(changed))
        assert _decisions(plan)["Split"] == ("miss", "upstream miss"), _decisions(plan)
        cycle.serves(_expand_workflow(changed), "Split")
        return
    workflow = _expand_workflow(
        children="[] if row[0]['id'] == 'p2' else row[0]['id'] + 'a'"
    )
    cycle = _Cycle(InMemoryJobStorage())
    await cycle.full(workflow)
    cycle.serves(workflow, "Split", whole=True)


@pytest.mark.asyncio
async def test_an_expand_a_chat_reads_serves_it_its_children() -> None:
    """A chat reads an expand's children one scalar per prompt row, so the expand
    is served whole and a pass-through step carries those stored children."""
    workflow = _expand_workflow()
    workflow["ops"].append(
        _ask(
            "LLMChatOp",
            ["Split"],
            messages=[_SPEAK],
            rowwise_template="{v}",
            rowwise_columns=[{"label": "v", "node": "Split", "path": "items.output"}],
        )
        | {"id": "Ask"}
    )
    workflow["outputs"] = [{"name": "split", "ref": "Ask"}]
    cycle = _Cycle(InMemoryJobStorage())
    await cycle.full(workflow)
    plan, dispatched = cycle.plan(workflow)
    assert _decisions(plan)["Split"] == ("hit", None), _decisions(plan)
    graph = _runtime_graph(workflow)
    assert _node(graph, "Split") not in dispatched.nodes
    step = next(
        node
        for node in dispatched.nodes
        if plan.pass_throughs.get(node) == _node(graph, "Split")
    )
    assert [
        row["output"] for row in dispatched.nodes[step].data_spec.plan[0].values
    ] == [
        "p1a",
        "p1b",
        "p2a",
        "p2b",
    ]


@pytest.mark.asyncio
async def test_an_expand_cycle_keys_by_its_parent_beside_a_served_list() -> None:
    """An expand reading a keyed parent and an unkeyed whole-op list keys its
    children under that parent while the list is served and the parent reruns,
    so its next run answers those children instead of missing an unkeyed parent."""
    workflow = _expand_workflow(fixed=True)
    cycle = _Cycle(InMemoryJobStorage())
    await cycle.full(workflow)
    changed = _expand_workflow(['{"id": "p1"}', '{"id": "p2", "v": 2}'], fixed=True)
    plan, _, _ = await cycle.rerun(changed)
    assert _decisions(plan)["Fill"] == ("miss", "partial records"), _decisions(plan)
    assert _decisions(plan)["Const"] == ("hit", None), _decisions(plan)
    cycle.serves(changed, "Split")


@pytest.mark.asyncio
async def test_a_row_cycle_stores_every_record_and_a_third_run_hits() -> None:
    """A row op whose two live keys miss in the middle: run 2 computes just those
    and writes every key, so run 3 serves the op whole. Covers finding 4."""
    rows = [
        {"id": "a", "t": "x"},
        {"id": "b", "t": "y"},
        {"id": "c", "t": "z"},
    ]
    amended = [rows[0], {"id": "b", "t": "y!"}, {"id": "c", "t": "z!"}]
    cycle, workflow = _Cycle(InMemoryJobStorage()), _tag_workflow(amended)
    await cycle.full(_tag_workflow(rows))
    plan, _, items = await cycle.rerun(workflow)
    assert _decisions(plan)["Tag"] == ("miss", "partial records"), _decisions(plan)
    stored = [_stored_entry(cycle.storage, "req-2", "Tag", r) for r in "abc"]
    tag = _node(_runtime_graph(workflow), "Tag")
    assert {item["key"] for item in items[tag]} == {"a", "b", "c"}
    full = _run_step(_runtime_graph(workflow).nodes[tag], {})
    by_key = {str(entry.address.record): entry.value for entry in stored}
    assert [by_key[record] for record in sorted(by_key)] == [
        item["value"] for item in full
    ]
    cycle.serves(workflow, "Tag", whole=True)


def _parse_chain_workflow(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A keyed row step's text through a parse step and a scoring row step, so
    the chain below a partially served row step is keyed only by inheritance."""
    text = "def text(c):\n    return c[0]['payload']\n"
    score = "def score(c):\n    return str(c[0]['n'] * 10)\n"
    ops = [
        {**_lambda("Texts", ["Rows"], "text", text), "key": _ROWS_KEY},
        {"id": "Parsed", "op": "ParseOp", "inputs": ["Texts"]},
        {**_lambda("Score", ["Parsed"], "score", score), "mode": "row"},
    ]
    return _pipeline(
        "parse-chain",
        ops,
        "Score",
        [json.dumps({"id": row["id"], "payload": row["payload"]}) for row in rows],
    )


@pytest.mark.asyncio
async def test_a_chain_below_a_partially_served_row_step_serves_on_a_third_run() -> (
    None
):
    """One row fails in run 1 and is fixed in run 2, so run 2's row step is
    partially served; the parse and row steps below it store their records
    under the keys they inherit, exactly as a full run writes them, so run 3
    serves the whole chain."""
    rows = [
        {"id": "s1", "payload": '{"n": 1}'},
        {"id": "s2", "payload": '{"n": 2}'},
        {"id": "s3", "payload": "not-json"},
        {"id": "s4", "payload": '{"n": 4}'},
    ]
    fixed = [*rows[:2], {"id": "s3", "payload": '{"n": 3}'}, rows[3]]
    cycle = _Cycle(InMemoryJobStorage())
    await cycle.full(_parse_chain_workflow(rows))
    plan, _, _ = await cycle.rerun(_parse_chain_workflow(fixed))
    assert _decisions(plan)["Texts"] == ("miss", "partial records"), _decisions(plan)
    for op in ("Parsed", "Score"):
        index = read_index(cycle.storage, "req-2")
        assert index is not None
        assert {e.record for e in index.entries if e.op == op} == {
            "s1",
            "s2",
            "s3",
            "s4",
        }
    again, _ = cycle.plan(_parse_chain_workflow(fixed))
    assert _decisions(again) == {
        "Texts": ("hit", None),
        "Parsed": ("hit", None),
        "Score": ("hit", None),
    }, _decisions(again)


@pytest.mark.parametrize("make", ["reduce", "expand", "empty_expand", "unkeyed_expand"])
@pytest.mark.asyncio
async def test_a_served_reduce_or_expand_output_is_one_whole_run_value(
    make: str, monkeypatch: pytest.MonkeyPatch, job_storage: InMemoryJobStorage
) -> None:
    """A reduce or expand that is also the workflow output stores one record per
    group, child or parent, yet a whole run archives it as one value over all of
    them. A run the store answers completely rebuilds exactly that one value, so
    the output merge finds one value where a full run produces one, whether the
    records are groups, children, an unkeyed parent's children or a parent that
    expanded to nothing."""
    workflow = (
        _expand_workflow(children="[] if row[0]['id'] == 'p2' else row[0]['id'] + 'a'")
        if make == "empty_expand"
        else _reduce_workflow(_REDUCE_ROWS) if make == "reduce" else _expand_workflow()
    )
    if make == "unkeyed_expand":
        del workflow["ops"][0]["key"]
    graph = _runtime_graph(workflow)
    cycle = _Cycle(job_storage)
    await cycle.full(workflow)
    (output,) = graph.output_node_map

    async def _never(*_: Any) -> tuple[str, list[str]]:
        raise AssertionError("an all-hit request must submit nothing")

    monkeypatch.setattr(FlowmeshRuntimeManager, "_submit_workflow", _never)
    answers = await FlowmeshRuntimeManager().process_request(
        _reusing(graph), Schedule(worker_assignment={"w": list(graph.nodes)}), ["w"]
    )
    whole = [item["output"] for item in cycle.items[output]]
    assert answers["flat_outputs"] == {output: [json.dumps(whole)]}


@pytest.mark.parametrize("make", ["reduce", "expand"])
@pytest.mark.asyncio
async def test_an_items_output_of_a_whole_run_op_runs(make: str) -> None:
    """An output that names the whole item archives every item as a whole run
    emits it, envelope and all, which no stored value holds, so the op runs."""
    workflow = (
        _reduce_workflow(_REDUCE_ROWS) if make == "reduce" else _expand_workflow()
    )
    workflow["outputs"][0]["path"] = "items"
    cycle = _Cycle(InMemoryJobStorage())
    await cycle.full(workflow)
    plan, dispatched = cycle.plan(workflow)
    output = _node(_runtime_graph(workflow), "Rank" if make == "reduce" else "Split")
    assert output in dispatched.nodes
    assert _decisions(plan)[_names(_runtime_graph(workflow))[output]] == (
        "miss",
        "value not servable",
    )


def _collected_reduce_workflow(changed: bool = False) -> dict[str, Any]:
    """The reduce fixture with a list-mode collector above it as the output, so
    a live reader reads the partially served reduce's every group."""
    workflow = _reduce_workflow(_REDUCE_ROWS, changed=changed)
    collect = _lambda(
        "Collect",
        ["Rank"],
        "collect",
        "def collect(c):\n    return ['<'.join(map(str, c[0]))]\n",
    ) | {"mode": "list"}
    return {
        **workflow,
        "ops": [*workflow["ops"], collect],
        "outputs": [{"name": "collected", "ref": "Collect"}],
    }


@pytest.mark.asyncio
async def test_a_collector_above_a_partially_served_reduce_serves_a_third_run() -> None:
    """A list-mode collector reads every group of a reduce its run only partly
    serves; the served groups' stored entries answer the collector's column at
    their positions, so it computes, and the next run serves the collector
    instead of truncating its input and reporting a partial reduce again."""
    cycle = _Cycle(InMemoryJobStorage())
    await cycle.full(_collected_reduce_workflow())
    plan, dispatched, items = await cycle.rerun(_collected_reduce_workflow(True))
    assert _decisions(plan)["Rank"] == ("miss", "partial group records"), _decisions(
        plan
    )
    graph = _runtime_graph(_collected_reduce_workflow(True))
    (whole,) = [item["output"] for item in items[_node(graph, "Collect")]]
    assert whole.count("<") == 2 and whole.startswith("['t1']")
    stored = _stored_entry(cycle.storage, "req-2", "Collect", None)
    assert stored.value == [whole]
    again, served = cycle.plan(_collected_reduce_workflow(True))
    assert _decisions(again)["Collect"] == ("hit", None), _decisions(again)
    assert served.nodes == {}


@pytest.mark.parametrize(
    ("make", "served"),
    [("expand_reduce", ["Split", "Fold"]), ("unkeyed", ["Rank"])],
)
@pytest.mark.asyncio
async def test_an_unchanged_reduce_over_a_mixed_input_serves(
    make: str, served: list[str]
) -> None:
    """An unchanged replay serves a reduce that reads a mixed input: an expand the
    store answers whole is its stored source, so it decides its groups rather than
    missing as ``input dispatched``, and a reduce whose group reads an unkeyed
    value beside its keyed rows serves too because the planner names the group's
    inputs the way the writer does. Covers findings 2 and 6."""
    workflow = (
        _expand_reduce_workflow()
        if make == "expand_reduce"
        else _reduce_unkeyed_workflow()
    )
    cycle = _Cycle(InMemoryJobStorage())
    await cycle.full(workflow)
    for op in served:
        cycle.serves(workflow, op)


def _expand_reduce_workflow() -> dict[str, Any]:
    """A keyed row step, an expand over it, and a reduce that groups the children
    by each child's own ``parent`` field."""
    fill, split = _expand_workflow()["ops"]
    split = {
        **split,
        "code": "lambda row: [{'parent': row[0]['id'], 'x': s} for s in 'ab']",
    }
    fold = _lambda("Fold", ["Split"], "fold", "lambda rows: [r[0] for r in rows]")
    fold |= {"mode": "reduce", "by": "parent"}
    return _pipeline("expand-reduce", [fill, split, fold], "Fold")


def _reduce_unkeyed_workflow() -> dict[str, Any]:
    """A keyed row step whose two rows share a group, a whole-op list step, and a
    reduce that groups the keyed input and reads the whole-op value as an
    unkeyed input."""
    tag, rank = _REDUCE_OPS
    globals_op = _op("Globals", ["Rows"], "globs", "lambda c: [len(c[0])]", "list")
    rank = {**rank, "inputs": ["Tag", "Globals"]}
    rows = [{"id": "t1", "patient": "p1"}, {"id": "t2", "patient": "p1"}]
    return _pipeline(
        "reduce-unkeyed", [globals_op, tag, rank], "Rank", [json.dumps(r) for r in rows]
    )


_SLOTS = [{"children", "parent"}, {"parent"}]


def _fails_p2_group(name: str, item: Mapping[str, Any]) -> str | None:
    """Fail the reduce group keyed ``p2``."""
    return "boom" if (name, item.get("key")) == ("Rank", "p2") else None


def _fails_p2_child(name: str, item: Mapping[str, Any]) -> str | None:
    """Fail the expand's second child under the ``p2`` parent."""
    return (
        "boom"
        if (name, item.get("parent"), item.get("child")) == ("Split", 1, 1)
        else None
    )


@pytest.mark.parametrize(
    ("op", "record", "reason", "fail", "slots"),
    [
        ("Rank", "p2", "partial group records", _fails_p2_group, None),
        ("Split", "p2\x1f1", "partial children", _fails_p2_child, _SLOTS),
    ],
)
@pytest.mark.asyncio
async def test_a_failed_record_is_recomputed_and_then_served(
    op: str,
    record: str,
    reason: str,
    fail: Callable[[str, Mapping[str, Any]], str | None],
    slots: list[set[str]] | None,
) -> None:
    """A reduce group and an expand child whose task fails in run 1 are stored as
    markers, so run 2 recomputes each; the computed value lands under the success
    key, so run 3 serves the op whole."""
    workflow = _reduce_workflow(_REDUCE_ROWS) if slots is None else _expand_workflow()
    cycle = _Cycle(InMemoryJobStorage())
    await cycle.full(workflow, fail=fail)
    assert is_marker(_stored_entry(cycle.storage, "req-1", op, record).value)
    plan, dispatched, _ = await cycle.rerun(workflow)
    assert _decisions(plan)[op] == ("miss", reason), _decisions(plan)
    spec = dispatched.nodes[_node(_runtime_graph(workflow), op)].data_spec
    assert isinstance(spec, PythonData)
    if slots is not None:
        assert [set(slot) for slot in spec.plan[-1].values or []] == slots
    stored = _stored_entry(cycle.storage, "req-2", op, record)
    assert not is_marker(stored.value), stored.value
    cycle.serves(workflow, op)


def _stored_entry(
    storage: InMemoryJobStorage, job_id: str, op: str, record: str | None
) -> RecordEntry:
    """One job's stored entry for one record."""
    index = read_index(storage, job_id)
    assert index is not None
    found = index.find(op=op, record=record, round_number=0)
    assert found is not None
    entry = read_entry(storage, found.digest)
    assert entry is not None
    return entry


def _expand_workflow(
    rows: list[str] | None = None,
    fixed: bool = False,
    children: str = "row[0]['id'] + 'b'",
) -> dict[str, Any]:
    """A keyed row step read by an expand that splits every row into two children;
    ``fixed`` adds an unkeyed whole-op list beside the parent, and ``children``
    builds each child."""
    fill = _op("Fill", ["Rows"], "fill", "lambda row: row[0]", "row")
    fill["key"] = {"node": "Rows", "path": "id"}
    split = _op(
        "Split",
        ["Fill"],
        "split",
        f"lambda row: [row[0]['id'] + 'a', {children}]",
        "expand",
    )
    if not fixed:
        return _pipeline("expand-reuse", [fill, split], "Split", rows)
    const = _op("Const", ["Fixed"], "const", "lambda c: [len(c)]", "list")
    split |= {"inputs": ["Fill", "Const"], "fn_name": "split2"}
    workflow = _pipeline("expand-reuse", [const, fill, split], "Split", rows)
    workflow["inputs"]["Fixed"] = ["1", "2"]
    return workflow


def _read_cycle_workflow(rows: list[str]) -> Any:
    """A keyed unversioned read whose one item holds a whole column table,
    feeding an expand of the table's rows, then a row step."""
    read = {
        "id": "Read",
        "op": "DataRetrievalOp",
        "data_spec": {**_SQL, "key": {"column": "id"}},
    }
    split = _op(
        "Split", ["Read"], "split", "def split(value):\n    return value[0]\n", "expand"
    )
    tag = _op("Tag", ["Split"], "tag", "lambda c: c[0]['t'].upper()", "row")
    return _pipeline("read-cycle", [read, split, tag], "Tag", rows)


def _read_items(rows: list[dict[str, Any]], per_row: bool) -> list[dict[str, Any]]:
    """A retrieval's task items: one per record, or one holding the whole table."""
    if not per_row:
        return _keyed_table(rows)["items"]
    return [_keyed_table([row])["items"][0] for row in rows]


@pytest.mark.parametrize("per_row", [False, True])
@pytest.mark.asyncio
async def test_a_read_a_reusing_job_reruns_feeds_an_expand(
    monkeypatch: pytest.MonkeyPatch, job_storage: InMemoryJobStorage, per_row: bool
) -> None:
    """An unversioned read a reusing job re-runs hands its expand the items a full
    run gives it. Its children are served only when it hands one item per record:
    then an unchanged run serves them and a changed row re-expands that parent.
    One item holding every record leaves the children unserved, so a run that
    changes any row re-expands the whole table instead of serving a stale one,
    and each recomputed child reaches the row step as its new value."""
    graph = _runtime_graph(_read_cycle_workflow([]))
    read, split, tag = (_node(graph, name) for name in ("Read", "Split", "Tag"))
    state: dict[str, Any] = {"rows": [], "column": None, "done": {}, "graph": graph}

    async def _submit(_self: Any, task_yaml: str, _id: str) -> tuple[str, list[str]]:
        nodes = [n["name"] for n in yaml.safe_load(task_yaml)["spec"]["graph"]["nodes"]]
        for node in nodes:
            op = state["graph"].nodes[node]
            steps = getattr(op.data_spec, "plan", [])
            stages = [e.node for e in steps if e.kind == "stage" and not e.values]
            items = (
                _read_items(state["rows"], per_row)
                if node == read
                else _run_step(op, {n: state["done"][n] for n in stages})
            )
            state["done"][node] = items
            state[f"task-{node}"] = {"items": items}
        return f"wf-{len(nodes)}", [f"task-{n}" for n in nodes]

    async def _retrieve(task_id: str) -> Any:
        return state[task_id]

    async def _done(_self: Any, status: dict[str, str], _id: str) -> bool:
        status.update(dict.fromkeys(status, "DONE"))
        return True

    async def _statuses(_self: Any, workflow_id: str, **_: Any) -> Any:
        rows = [
            {"task_id": t, "status": "DONE", "graph_node_name": t[5:]}
            for t in state
            if t.startswith("task-")
        ]
        return TaskStatusPage(rows=rows, next_cursor="0")

    fm = SimpleNamespace(results=SimpleNamespace(retrieve=_retrieve))
    build = FlowmeshRuntimeManager._build_task_spec

    async def _build(self: Any, info: Any, planned: Any, schedule: Any = None):
        if split in planned.nodes:
            state["column"] = planned.nodes[split].data_spec.plan[0].values
            state["graph"] = planned
        return await build(self, info, planned, schedule)

    for name, value in (
        ("_build_task_spec", _build),
        ("_submit_workflow", _submit),
        ("update_task_status", _done),
        ("fetch_workflow_task_statuses", _statuses),
        ("fm", property(lambda _self: fm)),
    ):
        monkeypatch.setattr(FlowmeshRuntimeManager, name, value)
    monkeypatch.setattr(envs, "LUMILAKE_POLL_INTERVAL_SECONDS", 0.0)

    async def once(rows: list[dict[str, Any]], job: str, reuse: bool):
        state["rows"], state["done"], state["graph"] = rows, {}, graph
        info = _reusing(graph)
        info.request_id, info.member_reuse = job, {job: reuse}
        info.reuse_records = reuse
        info.runtime_graph = graph
        out = await FlowmeshRuntimeManager().process_request(
            info, Schedule({"w": list(graph.nodes)}), ["w"]
        )
        return {d.op: (d.decision, d.reason) for d in out["reuse_decisions"]}

    def computed(op: str) -> list[Any]:
        return [
            i["output"]["t"]
            for i in state[f"task-{op}"]["items"]
            if not i.get("served")
        ]

    rows = [{"id": "a1", "t": "x"}, {"id": "a2", "t": "x"}]
    await once(rows, "req-1", False)
    assert {i["parent"] for i in state[f"task-{split}"]["items"]} == (
        {0, 1} if per_row else {0}
    )
    index = read_index(job_storage, "req-1")
    assert index is not None
    stored = {e.record for e in index.entries if e.op == "Split"}
    if per_row:
        assert stored == {"a1\x1f0", "a1\x1fcount", "a2\x1f0", "a2\x1fcount"}
    else:
        assert stored == set()
    assert (await once(rows, "req-2", True))["Split"] == (
        ("hit", None) if per_row else ("miss", "unkeyed row")
    )
    changed = [rows[0], {"id": "a2", "t": "z"}]
    plan = await once(changed, "req-3", True)
    assert state["column"] == _read_items(changed, per_row)
    assert plan["Split"] == (
        ("miss", "partial children") if per_row else ("miss", "unkeyed row")
    )
    assert computed(split) == (["z"] if per_row else ["x", "z"])
    assert sorted(i["output"] for i in state[f"task-{tag}"]["items"]) == ["X", "Z"]


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
    assert _decisions(plan)["Join"] == ("miss", "spec changed"), _decisions(plan)
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
