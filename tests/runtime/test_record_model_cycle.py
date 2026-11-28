"""The three-run cycle of a per-record model op, through the runtime manager:
run 1 stores everything, run 2 recomputes the changed rows and merges them
back, run 3 hits every op and dispatches nothing."""

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from flowmesh.models.tasks import TaskStatusPage

from lumilake import envs
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.record_writer import read_index
from lumilake_server.runtime.request import RequestInfo
from lumilake_server.runtime.runtime_manager import FlowmeshRuntimeManager
from lumilake_server.utils.job_storage import InMemoryJobStorage
from tests.runtime.runtime_manager.conftest import job_storage as job_storage
from tests.runtime.test_record_reuse import (  # noqa: F401
    _lambda,
    _run_step,
    _runtime_graph,
)

_TAG = {
    "id": "Tag",
    "op": "LambdaOp",
    "inputs": ["Rows"],
    "mode": "row",
    "fn_name": "tag",
    "code": "def tag(c):\n    return {'id': c[0]['id'], 't': c[0]['t'].upper()}\n",
    "key": {"node": "Rows", "path": "id"},
}
_COMPLETION = {
    "id": "Ask",
    "op": "LLMCompletionOp",
    "inputs": ["Tag"],
    "config": {"model": "m", "temperature": 0},
    "prompt": "Tag",
    "key": {"node": "Tag", "path": "id"},
}
_CHAT = {
    "id": "Ask",
    "op": "LLMChatOp",
    "inputs": ["Tag"],
    "config": {"model": "m", "temperature": 0},
    "messages": [{"role": "user", "content": "x"}],
    "rowwise_template": "{v}",
    "rowwise_columns": [{"label": "v", "node": "Tag", "path": "items.output.t"}],
    "key": {"node": "Tag", "path": "id"},
}
_HOSTED = {
    **_CHAT,
    "config": {
        "model": "m",
        "temperature": 0,
        "api": {"url": "http://llm", "authorization": "tok"},
    },
}
_AFTER = {
    "id": "After",
    "op": "LambdaOp",
    "inputs": ["Ask"],
    "mode": "row",
    "fn_name": "after",
    "code": "def after(c):\n    return str(c[0]) + '!'\n",
}
_ROWS = [{"id": "a", "t": "x"}, {"id": "b", "t": "y"}]
_RETRIEVED = [
    {"table": {"df": json.dumps({n: {"0": v} for n, v in r.items()})}} for r in _ROWS
]
_AMENDED = [_ROWS[0], {"id": "b", "t": "z"}]
_READ_OP = {
    "id": "Read",
    "op": "DataRetrievalOp",
    "data_spec": {
        "type": "lumid",
        "mode": "sql",
        "template": "SELECT id, t FROM rows",
        "key": {"column": "id"},
    },
}
_OVER_READ = {
    **_CHAT,
    "inputs": ["Tag", "Read"],
    "rowwise_template": "{v}{t}",
    "rowwise_columns": [
        {"label": "v", "node": "Tag", "path": "items.output.t"},
        {"label": "t", "node": "Read", "path": "table.df.t"},
    ],
}
_WHOLE = {
    **_CHAT,
    "id": "First",
    "inputs": ["Ask"],
    "rowwise_columns": [
        {"label": "v", "node": "Ask", "path": "items.metadata"},
    ],
    "rowwise_template": "{v}",
    "key": {"node": "Ask", "path": "id"},
}


class _Worker:
    """A fake model worker: a python step runs by its generated task code, a model op
    answers one text per live row, naming that row's own values, so a partial
    run stores the values a full run would."""

    def __init__(self) -> None:
        self.graph: Any = None
        self.submissions: list[list[str]] = []
        self.responses: dict[str, Any] = {}
        self.specs: list[dict[str, Any]] = []
        self.fail_first = False
        self.cancelled: list[str] = []

    def submit(self, graph: Any, task_yaml: str) -> tuple[str, list[str]]:
        self.graph = graph
        spec = yaml.safe_load(task_yaml)
        self.specs.append(spec)
        nodes = [node["name"] for node in spec["spec"]["graph"]["nodes"]]
        self.submissions.append(nodes)
        done: dict[str, list[Any]] = {}
        for node in nodes:
            op = graph.nodes[node]
            if op.task_type == "data_retrieval":
                response = {"items": _RETRIEVED}
            elif op.task_type == "python":
                response = _step(op, done)
            else:
                response = _answer(op, done)
            self.responses[f"task-{node}"] = response
            done[node] = response["items"]
        return f"wf-{len(self.submissions)}", [f"task-{node}" for node in nodes]


def _step(op: Any, done: dict[str, list[Any]]) -> dict[str, Any]:
    """One python step's result, run by its generated task code over its stages."""
    stages = [
        entry.node
        for entry in op.data_spec.plan
        if entry.kind == "stage" and entry.values is None
    ]
    missing = [stage for stage in stages if stage not in done]
    if missing:
        raise AssertionError(f"{op.node_id} dispatched without {missing}")
    return {"items": _run_step(op, {stage: done[stage] for stage in stages})}


def _walk(item: Any, path: str) -> Any:
    """One upstream item's value at a node path."""
    for part in path.removeprefix("value.").removeprefix("items.").split("."):
        if part and isinstance(item, dict):
            item = item.get(part, item)
    return item


def _answer(op: Any, done: Mapping[str, list[Any]]) -> dict[str, Any]:
    """One model op's response, one text per live row of its own spec naming that
    row's values; a one-value column broadcasts over the rows."""
    spec = op.data_spec
    reads = (
        [(column.node, column.path or "output") for column in spec.columns]
        if spec.type == "dataframe"
        else [(spec.node, "output")]
    )
    width = max(len(done[node]) for node, _ in reads)
    texts = [
        f"{op.node_id}:"
        + "|".join(
            str(_walk(done[node][min(at, len(done[node]) - 1)], path))
            for node, path in reads
        )
        for at in range(width)
    ]
    if op.task_type == "api":
        return {"items": [{"rows": [{"text": text} for text in texts]}]}
    if spec.type == "dataframe":
        return {"items": [{"output": text} for text in texts]}
    prompt = [{"role": "user", "content": "x"}]
    return {"items": [{"output": t, "metadata": {"prompt": prompt}} for t in texts]}


@pytest.fixture
def worker() -> _Worker:
    return _Worker()


@pytest.fixture
def run(
    monkeypatch: pytest.MonkeyPatch, job_storage: InMemoryJobStorage, worker: _Worker
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Run one job through the runtime manager against the fake worker."""
    build = FlowmeshRuntimeManager._build_task_spec

    async def _build(self: Any, info: RequestInfo, graph: Any, schedule: Any = None):
        worker.graph = graph
        return await build(self, info, graph, schedule)

    async def _submit(_self: Any, task_yaml: str, _id: str) -> tuple[str, list[str]]:
        return worker.submit(worker.graph, task_yaml)

    async def _retrieve(task_id: str) -> Any:
        return worker.responses[task_id]

    async def _done(_self: Any, status: dict[str, str], _id: str) -> bool:
        status.update(dict.fromkeys(status, "DONE"))
        if worker.fail_first:
            status[next(iter(status))] = "FAILED"
        return True

    async def _statuses(_self: Any, workflow_id: str, **_: Any) -> Any:
        rows = [
            {"task_id": task, "status": "DONE", "graph_node_name": task[5:]}
            for task in worker.responses
        ]
        return TaskStatusPage(rows=rows, next_cursor="0")

    async def _cancel(_self: Any, workflow_id: str) -> None:
        worker.cancelled.append(workflow_id)

    fm = SimpleNamespace(results=SimpleNamespace(retrieve=_retrieve))
    for name, value in (
        ("_build_task_spec", _build),
        ("_submit_workflow", _submit),
        ("update_task_status", _done),
        ("fetch_workflow_task_statuses", _statuses),
        ("_abort_workflow", _cancel),
        ("fm", property(lambda _self: fm)),
    ):
        monkeypatch.setattr(FlowmeshRuntimeManager, name, value)
    monkeypatch.setattr(envs, "LUMILAKE_POLL_INTERVAL_SECONDS", 0.0)

    async def _run(
        graph: Any,
        request_id: str,
        workers: dict[str, list[str]] | None = None,
        gpu_workers: set[str] | None = None,
        reuse: bool = True,
    ) -> dict[str, Any]:
        info = RequestInfo(
            request_id=request_id,
            runtime_graphs={"g": graph},
            data_profile_graphs={},
            store_records=True,
            reuse_records=reuse,
            member_reuse={request_id: reuse},
            gpu_worker_ids=set() if gpu_workers is None else gpu_workers,
        )
        info.runtime_graph = graph
        info.batch_id = "batch-1"
        info.output_node_map = {
            node: ("g", name) for node, name in graph.output_node_map.items()
        }
        schedule = Schedule({"w": list(graph.nodes)} if workers is None else workers)
        result = await FlowmeshRuntimeManager().process_request(
            info, schedule, list(schedule.worker_assignment)
        )
        result["output_node_map"] = dict(info.output_node_map)
        return result

    return _run


def _workflow(
    ops: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    out: str,
    extra: Sequence[dict[str, str]] = (),
    **inputs: Any,
) -> Any:
    return _runtime_graph(
        {
            "name": "cycle",
            "inputs": {"Rows": [json.dumps(row) for row in rows], **inputs},
            "ops": ops,
            "outputs": [{"name": "out", "ref": out}, *extra],
        }
    )


def _reasons(outputs: dict[str, Any]) -> dict[str, tuple[str, str | None]]:
    return {d.op: (d.decision, d.reason) for d in outputs["reuse_decisions"]}


def _node(graph: Any, name: str) -> str:
    """The runtime node an op name lowered to."""
    return next(node for node, op in graph.op_names.items() if op == name)


def _runs(worker: _Worker, before: int, nodes: Sequence[str]) -> dict[str, int]:
    """How many sub-runs since ``before`` held each node."""
    return {
        node: sum(1 for sub in worker.submissions[before:] if node in sub)
        for node in nodes
    }


def _keys(storage: InMemoryJobStorage, job_id: str) -> dict[tuple[str, Any], str]:
    """Every record one job stored, addressed by op name and record key."""
    index = read_index(storage, job_id)
    assert index is not None
    return {(entry.op, entry.record): entry.digest for entry in index.entries}


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [_CHAT, _HOSTED, _COMPLETION])
async def test_a_partial_model_cycle_keeps_every_live_record(
    run: Callable[..., Awaitable[dict[str, Any]]],
    job_storage: InMemoryJobStorage,
    worker: _Worker,
    model: dict[str, Any],
) -> None:
    """A keyed chat, hosted chat or completion serves its unchanged rows: run 2
    computes only the changed row and merges it back, run 3 hits every op and
    dispatches nothing, and the reader's output equals a full run's."""
    ops = [_TAG, model, _AFTER]

    async def once(rows: list[dict[str, Any]], request_id: str, reuse: bool = True):
        return await run(_workflow(ops, rows, "After"), request_id, reuse=reuse)

    await once(_ROWS, "req-1")
    first = _keys(job_storage, "req-1")
    outputs = await once(_AMENDED, "req-2")
    assert _reasons(outputs)["Ask"] == ("miss", "partial records"), _reasons(outputs)
    stored = _keys(job_storage, "req-2")
    assert stored["Ask", "a"] == first["Ask", "a"]
    assert stored["Ask", "b"] != first["Ask", "b"]
    assert {record for op, record in stored if op == "After"} == {"a", "b"}
    assert (
        outputs["flat_outputs"]
        == (await once(_AMENDED, "req-full", False))["flat_outputs"]
    )
    submitted = len(worker.submissions)
    again = await once(_AMENDED, "req-3")
    assert [d.decision for d in again["reuse_decisions"]] == ["hit"] * 3
    assert worker.submissions[submitted:] == []


@pytest.mark.asyncio
async def test_a_trimmed_phase_reads_the_rows_the_store_holds(
    run: Callable[..., Awaitable[dict[str, Any]]], worker: _Worker
) -> None:
    """A stored step above a changed one leaves the phase, and the changed step
    reads that step's stored rows."""
    rewrite: dict[str, Any] = {
        **_TAG,
        "id": "Rewrite",
        "inputs": ["Tag"],
        "fn_name": "rewrite",
        "code": (
            "def rewrite(c):\n" "    return {'id': c[0]['id'], 't': c[0]['t'] + '!'}\n"
        ),
    }
    chat = {
        **_CHAT,
        "inputs": ["Rewrite"],
        "rowwise_columns": [
            {"label": "v", "node": "Rewrite", "path": "items.output.t"}
        ],
        "key": {"node": "Rewrite", "path": "id"},
    }
    await run(_workflow([_TAG, rewrite, chat, _AFTER], _ROWS, "After"), "req-1")
    changed = _workflow(
        [
            _TAG,
            {**rewrite, "code": rewrite["code"].replace("+ '!'", "+ '?'")},
            chat,
            _AFTER,
        ],
        _ROWS,
        "After",
    )
    before = len(worker.submissions)
    outputs = await run(changed, "req-2")
    node = _node(changed, "Rewrite")
    ran = [nodes for nodes in worker.submissions[before:] if node in nodes]
    assert ran == [[node]], worker.submissions[before:]
    assert [item["output"] for item in worker.responses[f"task-{node}"]["items"]] == [
        '{"id": "a", "t": "X?"}',
        '{"id": "b", "t": "Y?"}',
    ]
    assert [d.decision for d in outputs["reuse_decisions"]] == [
        "hit",
        "hit",
        "miss",
        "miss",
    ]
    again = await run(changed, "req-3")
    assert [d.decision for d in again["reuse_decisions"]] == ["hit"] * 4


@pytest.mark.parametrize("kind", ["aligned", "expand"])
@pytest.mark.asyncio
async def test_a_changed_chat_reads_a_served_row_as_its_dict(
    run: Callable[..., Awaitable[dict[str, Any]]], worker: _Worker, kind: str
) -> None:
    """An aligned or expand step emits its row under ``output`` as it stands, so
    the pass-through that replays its stored rows to a chat whose config changed
    hands them over as those dicts, not as their JSON text."""
    producer = _reader_chain(kind)
    ops = [
        *producer,
        {
            **_CHAT,
            "inputs": ["Provider"],
            "rowwise_columns": [
                {"label": "v", "node": "Provider", "path": "items.output.s"}
            ],
            "key": {"node": "Provider", "path": "id"},
        },
    ]
    await run(_workflow(ops, _ROWS, "Ask"), "req-1")
    changed = {**ops[-1], "config": {"model": "m", "temperature": 0.5}}
    await run(_workflow([*ops[:-1], changed], _ROWS, "Ask"), "req-2")
    (step,) = [n for n in worker.submissions[-1] if "/pass/" in n]
    replay = worker.responses[f"task-{step}"]["items"]
    assert [item["output"]["s"] for item in replay] == ["x*", "x*"], replay


def _reader_chain(kind: str) -> list[dict[str, Any]]:
    """A keyed aligned or expand producer whose rows carry ``s``."""
    if kind == "expand":
        code = "lambda row: [dict(row[0], s='x*')]"
        return [_TAG, {**_lambda("Provider", ["Tag"], "provider", code), "mode": kind}]
    code = "lambda c: [dict(r, s='x*') for r in c[0]]"
    producer = _lambda("Provider", ["Rows"], "provider", code)
    producer |= {"mode": kind, "key": {"node": "Rows", "path": "id"}}
    return [producer]


@pytest.mark.asyncio
async def test_a_generated_step_lands_on_a_cpu_worker(
    run: Callable[..., Awaitable[dict[str, Any]]], worker: _Worker
) -> None:
    """A partial chat's generated steps land on the CPU worker whose nodes the
    store answered, while the chat keeps its GPU worker."""
    await run(_workflow([_TAG, _CHAT], _ROWS, "Ask"), "req-1")
    graph = _workflow([_TAG, _CHAT], _AMENDED, "Ask")
    ask = _node(graph, "Ask")
    split = {"gpu": [ask], "cpu": [node for node in graph.nodes if node != ask]}
    await run(graph, "req-2", split, {"gpu"})
    selected = worker.specs[-1]["metadata"]["annotations"]["schedule_hint"][
        "selected_worker"
    ]["selected"]
    generated = [
        node for node in selected if "/pass/" in node or node.endswith("/merge")
    ]
    assert generated, selected
    assert all(selected[node] == ["cpu"] for node in generated), selected
    assert selected[ask] == ["gpu"], selected[ask]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ops", "reason"),
    [
        ([_TAG, _COMPLETION], "value not servable"),
        ([_TAG, {**_TAG, "id": "Ask"}], "partial records"),
        (
            [_TAG, _CHAT, {**_COMPLETION, "id": "Ask2", "inputs": ["Ask"]}],
            "value not servable",
        ),
    ],
)
async def test_a_partly_served_output_runs_under_its_own_id(
    run: Callable[..., Awaitable[dict[str, Any]]],
    worker: _Worker,
    ops: list[dict[str, Any]],
    reason: str,
) -> None:
    """A python output step keeps its own id and partial run; a completion output
    runs in full with its chat history."""
    out = ops[-1]["id"]
    await run(_workflow(ops, _ROWS, out), "req-1")
    before = len(worker.submissions)
    outputs = await run(_workflow(ops, _AMENDED, out), "req-2")
    assert _reasons(outputs)[out] == ("miss", reason), _reasons(outputs)
    ((node, (_, name)),) = outputs["output_node_map"].items()
    assert name == "out" and not node.endswith("/merge"), node
    assert sum(1 for sub in worker.submissions[before:] if node in sub) == 1
    if ops[-1]["op"] != "LambdaOp":
        assert outputs["chat_histories"][node] == [
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": t}]
            for t in outputs["flat_outputs"][node]
        ]


_CFGSTEP = {
    "id": "Cfg",
    "op": "LambdaOp",
    "inputs": ["Config"],
    "mode": "row",
    "fn_name": "cfg",
    "code": "def cfg(c):\n    return {'id': 'shared', 'v': c[0]}\n",
}
_CFG: dict[str, tuple[dict[str, Any], list[str], str, str]] = {
    "broadcast": (_CFGSTEP, ['"shared"'], "items.output.v", '"shared"'),
    "whole_op": (
        {
            **_CFGSTEP,
            "mode": "list",
            "code": (
                "def cfg(c):\n    import json\n"
                "    return [json.loads(v) for v in c[0]]\n"
            ),
        },
        ['"x"', '"y"'],
        "items.output",
        "y",
    ),
}
"""The ``Cfg`` step beside the chat's row column: its input, its ``items`` path and
the value the changed row reads. A rowwise step keyed on one input value
stores one entry that broadcasts; a list-mode step stores its whole row list."""


@pytest.mark.asyncio
@pytest.mark.parametrize("column", ["broadcast", "whole_op"])
async def test_a_partial_chat_reads_its_served_columns(
    run: Callable[..., Awaitable[dict[str, Any]]],
    job_storage: InMemoryJobStorage,
    worker: _Worker,
    column: str,
) -> None:
    """A chat that feeds another op recomputes only the changed rows, and each one
    reads its own row of every served column: a stored value that broadcasts reads
    the same for every row, and a whole-op producer's row list gives each row its
    own entry."""
    step, config, path, value = _CFG[column]
    chat = {
        **_CHAT,
        "inputs": ["Tag", "Cfg"],
        "rowwise_template": "{v}{shared}",
        "rowwise_columns": [
            {"label": "v", "node": "Tag", "path": "items.output.t"},
            {"label": "shared", "node": "Cfg", "path": path},
        ],
    }
    ops: list[dict[str, Any]] = [_TAG, step, chat, _AFTER]
    await run(_workflow(ops, _ROWS, "After", Config=config), "req-1")
    first = _keys(job_storage, "req-1")
    graph = _workflow(ops, _AMENDED, "After", Config=config)
    ask = _node(graph, "Ask")
    outputs = await run(graph, "req-2")
    assert _reasons(outputs)["Ask"] == ("miss", "partial records"), _reasons(outputs)
    assert worker.responses[f"task-{ask}"]["items"] == [{"output": f"{ask}:Z|{value}"}]
    stored = _keys(job_storage, "req-2")
    assert stored["Ask", "a"] == first["Ask", "a"]
    assert stored["Ask", "b"] != first["Ask", "b"]
    assert {record for op, record in stored if op == "After"} == {"a", "b"}


_ASK_Y = {**_CHAT, "id": "AskY", "rowwise_template": "{v}-y"}
_BOTH = {**_AFTER, "id": "Both", "inputs": ["AskX", "AskY"]}
_SPLIT = {
    "id": "Split",
    "op": "LambdaOp",
    "inputs": ["Tag"],
    "mode": "expand",
    "fn_name": "split",
    "code": (
        "def split(c):\n"
        "    return [{'id': c[0]['id'] + str(i), 'n': i} for i in range(2)]\n"
    ),
}


@pytest.mark.asyncio
async def test_a_re_ran_read_that_is_an_output_survives_a_level(
    run: Callable[..., Awaitable[dict[str, Any]]], worker: _Worker
) -> None:
    """A re-ran retrieval that is a workflow output keeps its rows across a level,
    and a chat that reads it stays in the main run with it."""
    out = [{"name": "rows", "ref": "Read"}]
    ops: list[dict[str, Any]] = [_READ_OP, _TAG, _OVER_READ, _AFTER]
    await run(_workflow(ops, _ROWS, "After", out), "req-1")
    changed = [*ops[:-2], {**_OVER_READ, "rowwise_template": "{v}|{t}"}, _AFTER]
    graph = _workflow(changed, _ROWS, "After", out)
    read_node, ask = _node(graph, "Read"), _node(graph, "Ask")
    before = len(worker.submissions)
    again = await run(graph, "req-2")
    ask_runs = [s for s in worker.submissions[before:] if ask in s]
    assert ask_runs and all(
        read_node in s and _node(graph, "After") in s for s in ask_runs
    ), worker.submissions[before:]
    fresh = await run(graph, "req-full", reuse=False)
    assert again["flat_outputs"][read_node] == fresh["flat_outputs"][read_node]


@pytest.mark.asyncio
async def test_two_partly_served_chats_run_in_one_level_and_dispatch_once(
    run: Callable[..., Awaitable[dict[str, Any]]], worker: _Worker
) -> None:
    """Two chats over the same phase rows share one level and run once each."""
    ops: list[dict[str, Any]] = [_TAG, {**_CHAT, "id": "AskX"}, _ASK_Y, _BOTH]
    await run(_workflow(ops, _ROWS, "Both"), "req-1")
    changed = _workflow(ops, _AMENDED, "Both")
    ask_x, ask_y = (_node(changed, name) for name in ("AskX", "AskY"))
    before = len(worker.submissions)
    second = await run(changed, "req-2")
    reasons = _reasons(second)
    assert reasons["AskX"] == reasons["AskY"] == ("miss", "partial records"), reasons
    together = [s for s in worker.submissions[before:] if {ask_x, ask_y} <= set(s)]
    assert len(together) == 1, worker.submissions[before:]
    assert _runs(worker, before, [ask_x, ask_y]) == {ask_x: 1, ask_y: 1}


_PARSE_BETWEEN = {
    **_AFTER,
    "id": "Parse",
    "inputs": ["Ask"],
    "code": "def after(c):\n    return c[0] + '!'\n",
    "key": {"node": "Ask", "path": "id"},
}
_EXPAND_BETWEEN = {
    **_SPLIT,
    "id": "Between",
    "inputs": ["Parse"],
    "code": "lambda c: [{'id': c[0] + str(i), 'v': c[0]} for i in (0, 1)]",
}
_TAG_BETWEEN = {
    **_TAG,
    "id": "Uppercase",
    "inputs": ["Between"],
    "code": "def tag(c):\n    return {'id': c[0]['id'], 't': c[0]['v'].upper()}\n",
    "key": {"node": "Between", "path": "id"},
}
_OVER_BETWEEN = {
    **_CHAT,
    "id": "Ask2",
    "inputs": ["Uppercase"],
    "rowwise_columns": [{"label": "v", "node": "Uppercase", "path": "items.output.t"}],
    "key": {"node": "Uppercase", "path": "id"},
}


@pytest.mark.asyncio
async def test_two_steps_between_two_chats_land_in_the_first_level(
    run: Callable[..., Awaitable[dict[str, Any]]], worker: _Worker
) -> None:
    """A row step, an expand and a step over it between two chats join the
    first level whole, so the chat over the expand still decides per record."""
    tail = {**_AFTER, "id": "Tail", "inputs": ["Ask2"]}
    ops: list[dict[str, Any]] = [
        _TAG,
        _CHAT,
        _PARSE_BETWEEN,
        _EXPAND_BETWEEN,
        _TAG_BETWEEN,
        _OVER_BETWEEN,
        tail,
    ]
    await run(_workflow(ops, _ROWS, "Tail"), "req-1")
    changed = _workflow(ops, _AMENDED, "Tail")
    ask, ask2 = (_node(changed, name) for name in ("Ask", "Ask2"))
    before = len(worker.submissions)
    second = await run(changed, "req-2")
    reasons = _reasons(second)
    assert reasons["Ask"] == reasons["Ask2"] == ("miss", "partial records"), reasons
    assert reasons["Between"] == ("miss", "partial children"), reasons
    assert _runs(worker, before, [ask, ask2]) == {ask: 1, ask2: 1}


_PACK = {**_TAG, "id": "Pack", "code": "def tag(c):\n    return [c[0]['t']]\n"}
_OVER_LIST = {
    **_CHAT,
    "inputs": ["Pack"],
    "rowwise_columns": [{"label": "v", "node": "Pack", "path": "items.output"}],
    "key": {"node": "Pack", "path": "id"},
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ops", "out", "refused"),
    [
        ([_TAG, _CHAT, _WHOLE], "First", "Ask"),
        ([_TAG, _PACK, _OVER_LIST, _AFTER], "After", "Pack"),
    ],
)
async def test_an_op_a_later_reader_refuses_is_left_to_the_main_run(
    run: Callable[..., Awaitable[dict[str, Any]]],
    worker: _Worker,
    ops: list[dict[str, Any]],
    out: str,
    refused: str,
) -> None:
    """An op whose stored rows a reader cannot take stays out of a level, so a
    rerun dispatches it once in the phase or once in the main run."""
    await run(_workflow(ops, _ROWS, out), "req-1")
    graph = _workflow(ops, _AMENDED, out)
    node = _node(graph, refused)
    before = len(worker.submissions)
    outputs = await run(graph, "req-2")
    assert _reasons(outputs)[refused] == ("miss", "read by a non-literal op")
    assert _runs(worker, before, [node]) == {node: 1 if refused == "Ask" else 2}


@pytest.mark.asyncio
async def test_a_failed_sub_run_cancels_its_workflow(
    run: Callable[..., Awaitable[dict[str, Any]]], worker: _Worker
) -> None:
    """A sub-run whose task fails leaves its other tasks running, so the
    request aborts the workflow it submitted rather than leaving workers held."""
    worker.fail_first = True
    with pytest.raises(RuntimeError):
        await run(_workflow([_TAG, _CHAT], _ROWS, "Ask"), "req-fail")
    assert worker.cancelled, "the sub-run's workflow was not cancelled"


_ALIGNED = {
    "id": "Prompts",
    "op": "LambdaOp",
    "inputs": ["Split"],
    "mode": "aligned",
    "fn_name": "prompts",
    "code": "lambda c: [{'id': r['id'], 't': r['n']} for r in c[0]]",
    "key": {"node": "Split", "path": "id"},
}
_ALIGNED_CHAT = {
    **_CHAT,
    "inputs": ["Prompts"],
    "rowwise_columns": [{"label": "v", "node": "Prompts", "path": "items.output.t"}],
    "key": {"node": "Prompts", "path": "id"},
}


@pytest.mark.asyncio
async def test_an_aligned_step_declaring_another_nodes_key_serves_per_row(
    run: Callable[..., Awaitable[dict[str, Any]]], worker: _Worker
) -> None:
    """An aligned step that declared the expand's key is stored under that key,
    so a changed row recomputes only its own chat, over the children not counts."""
    ops: list[dict[str, Any]] = [_TAG, _SPLIT, _ALIGNED, _ALIGNED_CHAT, _AFTER]
    await run(_workflow(ops, _ROWS, "After"), "req-1")
    assert set(
        _reasons(await run(_workflow(ops, _ROWS, "After"), "req-2")).values()
    ) == {("hit", None)}
    changed = _workflow(ops, _AMENDED, "After")
    ask = _node(changed, "Ask")
    before = len(worker.submissions)
    assert _reasons(await run(changed, "req-3"))["Ask"] == ("miss", "partial records")
    assert len(worker.responses[f"task-{ask}"]["items"]) == 2
    assert _runs(worker, before, [ask]) == {ask: 1}
