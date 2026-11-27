"""The three-run cycle of a per-record model op, through the runtime manager:
run 1 stores everything, run 2 recomputes the changed rows and merges them
back, run 3 hits every op and dispatches nothing."""

import json
from collections.abc import Awaitable, Callable, Mapping
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
_AMENDED = [_ROWS[0], {"id": "b", "t": "z"}]


class _Worker:
    """A fake model worker: a python step runs by its generated task code, a model op
    answers one text per live row, naming that row's own values, so a partial
    run stores the values a full run would."""

    def __init__(self) -> None:
        self.graph: Any = None
        self.submissions: list[list[str]] = []
        self.responses: dict[str, Any] = {}
        self.specs: list[dict[str, Any]] = []

    def submit(self, graph: Any, task_yaml: str) -> tuple[str, list[str]]:
        self.graph = graph
        spec = yaml.safe_load(task_yaml)
        self.specs.append(spec)
        nodes = [node["name"] for node in spec["spec"]["graph"]["nodes"]]
        self.submissions.append(nodes)
        done: dict[str, list[Any]] = {}
        for node in nodes:
            op = graph.nodes[node]
            response = (
                _answer(op, done) if op.task_type != "python" else _step(op, done)
            )
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
        return True

    async def _statuses(_self: Any, workflow_id: str, **_: Any) -> Any:
        rows = [
            {"task_id": task, "status": "DONE", "graph_node_name": task[5:]}
            for task in worker.responses
        ]
        return TaskStatusPage(rows=rows, next_cursor="0")

    results = SimpleNamespace(retrieve=_retrieve)
    for name, value in (
        ("_build_task_spec", _build),
        ("_submit_workflow", _submit),
        ("update_task_status", _done),
        ("fetch_workflow_task_statuses", _statuses),
        ("fm", property(lambda _self: SimpleNamespace(results=results))),
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
    ops: list[dict[str, Any]], rows: list[dict[str, Any]], out: str, **inputs: Any
) -> Any:
    return _runtime_graph(
        {
            "name": "cycle",
            "inputs": {"Rows": [json.dumps(row) for row in rows], **inputs},
            "ops": ops,
            "outputs": [{"name": "out", "ref": out}],
        }
    )


def _reasons(outputs: dict[str, Any]) -> dict[str, tuple[str, str | None]]:
    return {d.op: (d.decision, d.reason) for d in outputs["reuse_decisions"]}


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
    """A reusing run whose stored step sits above a changed one drops the stored step
    from its phase, and the changed step reads that step's stored rows."""
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
    node = next(
        node for node in changed.nodes if changed.op_names.get(node) == "Rewrite"
    )
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


@pytest.mark.asyncio
async def test_a_generated_step_lands_on_a_cpu_worker(
    run: Callable[..., Awaitable[dict[str, Any]]], worker: _Worker
) -> None:
    """A partial chat runs its generated steps on the CPU worker that served its
    producer, whose every node the store answered, while the chat keeps the GPU
    worker its plan chose."""
    await run(_workflow([_TAG, _CHAT], _ROWS, "Ask"), "req-1")
    graph = _workflow([_TAG, _CHAT], _AMENDED, "Ask")
    ask = next(node for node in graph.nodes if graph.op_names.get(node) == "Ask")
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
    ("out", "reason"),
    [
        (_COMPLETION, "value not servable"),
        ({**_TAG, "id": "Ask"}, "partial records"),
    ],
)
async def test_a_partly_served_output_runs_under_its_own_id(
    run: Callable[..., Awaitable[dict[str, Any]]], out: dict[str, Any], reason: str
) -> None:
    """A python row step that is the workflow output keeps its own id and partial
    run; a completion that is one runs in full, since only a full run keeps the
    output's chat history, and its history comes back with every row."""
    await run(_workflow([_TAG, out], _ROWS, out["id"]), "req-1")
    outputs = await run(_workflow([_TAG, out], _AMENDED, out["id"]), "req-2")
    assert _reasons(outputs)[out["id"]] == ("miss", reason), _reasons(outputs)
    ((node, (_, name)),) = outputs["output_node_map"].items()
    assert name == "out" and not node.endswith("/merge"), node
    histories = outputs["chat_histories"].get(node, [])
    if out is _COMPLETION:
        assert histories == [
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
    ask = next(node for node in graph.nodes if graph.op_names.get(node) == "Ask")
    outputs = await run(graph, "req-2")
    assert _reasons(outputs)["Ask"] == ("miss", "partial records"), _reasons(outputs)
    assert worker.responses[f"task-{ask}"]["items"] == [{"output": f"{ask}:Z|{value}"}]
    stored = _keys(job_storage, "req-2")
    assert stored["Ask", "a"] == first["Ask", "a"]
    assert stored["Ask", "b"] != first["Ask", "b"]
    assert {record for op, record in stored if op == "After"} == {"a", "b"}


_SECOND_CHAT = {
    **_CHAT,
    "id": "Ask2",
    "inputs": ["Ask"],
    "rowwise_columns": [{"label": "v", "node": "Ask", "path": "items.output"}],
    "key": {"node": "Ask", "path": "id"},
}


@pytest.mark.asyncio
async def test_a_second_model_level_decides_per_record(
    run: Callable[..., Awaitable[dict[str, Any]]], worker: _Worker
) -> None:
    """A second chat reads the first, so a changed row recomputes only its own
    records at both levels and an unchanged rerun serves both."""
    tail: dict[str, Any] = {**_AFTER, "id": "Tail", "inputs": ["Ask2"]}
    ops: list[dict[str, Any]] = [_TAG, _CHAT, _SECOND_CHAT, tail]
    await run(_workflow(ops, _ROWS, "Tail"), "req-1")
    changed = _workflow(ops, _AMENDED, "Tail")
    names = {node: changed.op_names.get(node) for node in changed.nodes}
    ask, ask2 = (node for node, name in names.items() if name in ("Ask", "Ask2"))
    before = len(worker.submissions)
    second = await run(changed, "req-2")
    reasons = _reasons(second)
    assert reasons["Ask"] == ("miss", "partial records"), reasons
    assert reasons["Ask2"] == ("miss", "partial records"), reasons
    ran = {
        node: sum(1 for nodes in worker.submissions[before:] if node in nodes)
        for node in (ask, ask2)
    }
    assert ran == {ask: 1, ask2: 1}, worker.submissions[before:]
    again = await run(changed, "req-3")
    assert [d.decision for d in again["reuse_decisions"]] == ["hit"] * 4
