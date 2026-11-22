"""Whole-op reuse: a reusing plan answers an op whose every record the store
already holds, dispatches the rest, and keys what it writes exactly as a later
reuse reads it back."""

import json
import logging
from dataclasses import replace
from typing import Any

import pytest

from lumilake_server.runtime import python_step
from lumilake_server.runtime.lambda_task_helpers import _column
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.record_store import RecordEntry
from lumilake_server.runtime.record_writer import RecordWriter
from lumilake_server.runtime.request import RequestInfo
from lumilake_server.runtime.reuse_planner import (
    ReusePlan,
    ReusePlanner,
    rebuilt_output,
    served_records,
)
from lumilake_server.runtime.runtime_manager import FlowmeshRuntimeManager
from lumilake_server.utils.job_storage import InMemoryJobStorage
from tests.runtime.runtime_manager.conftest import (
    job_storage as job_storage,
)  # noqa: F401
from tests.runtime.test_record_store import (
    _WORKFLOW,
    _runtime_graph,
)

_THIRD_CODE = (
    "def third(c):\n"
    "    import json\n"
    "    rows = [json.loads(v) for v in c[0]]\n"
    "    return [{'id': r['id'], 'text': r['text'] + '?'} for r in rows]\n"
)

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


def _answers(values: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        f"task-{n}": {"items": [{"output": r} for r in v]} for n, v in values.items()
    }


def _node(graph: Any, name: str) -> str:
    return next(node for node in graph.nodes if graph.op_names.get(node) == name)


def _with_code(workflow: dict[str, Any], op_id: str, code: str) -> dict[str, Any]:
    return {
        **workflow,
        "ops": [
            {**o, "code": code} if o["id"] == op_id else o for o in workflow["ops"]
        ],
    }


async def _run(
    storage: InMemoryJobStorage,
    workflow: dict[str, Any],
    *,
    results: dict[str, Any],
    job_id: str,
    served: dict[str, Any] | None = None,
) -> Any:
    """Write one job's finished tasks the way dispatch does."""
    graph = _runtime_graph(workflow)
    await _writer(storage).write(
        job_id=job_id,
        member_job_ids=[job_id],
        round_number=0,
        runtime_graph=graph,
        node_task_map={_node(graph, name): f"task-{name}" for name in results},
        task_worker={},
        results=_answers(results),
        served=served,
    )
    return graph


def _decisions(plan: ReusePlan) -> dict[str, tuple[str, str | None]]:
    return {d.op: (d.decision, d.reason) for d in plan.decisions}


def _planner(storage: InMemoryJobStorage) -> ReusePlanner:
    return ReusePlanner(storage=storage, logger=logging.getLogger("test.reuse"))


def _writer(storage: InMemoryJobStorage) -> RecordWriter:
    return RecordWriter(storage=storage, enabled=True, logger=logging.getLogger("t"))


@pytest.mark.asyncio
async def test_all_hit_serves_every_op_and_writes_nothing() -> None:
    """A resubmitted job whose ops and inputs are unchanged hits every op, so
    nothing is dispatched, and the served values equal the original run's."""
    storage, graph = InMemoryJobStorage(), _runtime_graph(_FOUR_OPS)
    await _run(storage, _FOUR_OPS, results=_ROWS, job_id="req-1")
    plan, dispatched = await _planner(storage).plan(
        runtime_graph=graph, round_number=0, job_id="req-2"
    )
    assert _decisions(plan) == dict.fromkeys(("Mark", "Join", "Final"), ("hit", None))
    assert dispatched.nodes == {}
    out = rebuilt_output(plan, _node(graph, "Final"))
    assert [json.loads(v) for v in out] == _ROWS["Final"]


@pytest.mark.asyncio
async def test_no_hit_dispatches_the_whole_graph() -> None:
    """A job whose spec the store never saw misses every op and dispatches the
    whole graph."""
    graph = _runtime_graph(_FOUR_OPS)
    plan, dispatched = await _planner(InMemoryJobStorage()).plan(
        runtime_graph=graph, round_number=0, job_id="req-1"
    )

    assert {d.decision for d in plan.decisions} == {"miss"}
    assert plan.decisions[0].reason == "no entry"
    assert set(dispatched.nodes) == set(graph.nodes)


@pytest.mark.asyncio
async def test_one_changed_op_reruns_only_it_and_its_descendants() -> None:
    """Changing the last op's code misses that op alone: its upstream hits, and
    the partial run rewrites its key so a resubmission of the same workflow
    afterwards hits everywhere."""
    storage = InMemoryJobStorage()
    await _run(storage, _FOUR_OPS, results=_ROWS, job_id="req-1")
    changed = _with_code(_FOUR_OPS, "Final", _THIRD_CODE.replace("+ '?'", "+ '!!'"))
    graph = _runtime_graph(changed)
    planner = _planner(storage)
    plan, dispatched = await planner.plan(
        runtime_graph=graph, round_number=0, job_id="req-2"
    )

    assert _decisions(plan) == {
        "Mark": ("hit", None),
        "Join": ("hit", None),
        "Final": ("miss", "no entry"),
    }
    assert list(dispatched.nodes) == [_node(graph, "Final")]
    assert dispatched.record_keys[_node(graph, "Final")]["path"] == "id"

    await _run(
        storage,
        changed,
        results={"Final": _ROWS["Final"]},
        job_id="req-2",
        served=served_records(plan),
    )
    again, _ = await planner.plan(runtime_graph=graph, round_number=0, job_id="req-3")
    assert {d.decision for d in again.decisions} == {"hit"}


def test_a_rebuilt_output_keeps_a_stored_string() -> None:
    """A hit output whose stored value is text is emitted as that text, not
    re-encoded as JSON."""
    entry = RecordEntry.model_validate(
        {
            "digest": "d",
            "address": {"op": "Text", "record": "a"},
            "value": '{"id": "a"}',
            "value_digest": "v",
            "source": {"job": "j", "task_id": "t"},
            "written_at": 0.0,
        }
    )
    assert rebuilt_output(ReusePlan(hits={"n": [entry]}), "n") == ['{"id": "a"}']


@pytest.mark.asyncio
async def test_a_failed_entry_is_never_served() -> None:
    """A stored failure answers no request: its op and everything downstream of
    it miss."""
    storage = InMemoryJobStorage()
    failed = {**_ROWS, "Mark": [{"_fail": {"op": "Mark"}}, *_ROWS["Mark"][1:]]}
    await _run(storage, _FOUR_OPS, results=failed, job_id="req-1")

    plan, _ = await _planner(storage).plan(
        runtime_graph=_runtime_graph(_FOUR_OPS), round_number=0, job_id="req-2"
    )
    assert plan.decisions[0].decision == "miss"
    assert {d.decision for d in plan.decisions} == {"miss"}


@pytest.mark.asyncio
async def test_an_unseeded_sampling_op_always_misses() -> None:
    """An op that draws tokens with no seed has no reproducible key, so it
    misses however often its stored draw is asked for."""
    storage = InMemoryJobStorage()
    await _run(storage, _FOUR_OPS, results=_ROWS, job_id="req-1")
    graph = _runtime_graph(_FOUR_OPS)
    final = _node(graph, "Final")
    graph.nodes[final] = replace(
        graph.nodes[final], task_type="inference", inference_spec={"temperature": 0.7}
    )

    plan, _ = await _planner(storage).plan(
        runtime_graph=graph, round_number=0, job_id="req-2"
    )

    assert _decisions(plan)["Mark"] == ("hit", None)
    assert _decisions(plan)["Join"][0] == "miss"


@pytest.mark.asyncio
async def test_an_all_hit_request_dispatches_no_task(
    monkeypatch: pytest.MonkeyPatch, job_storage: InMemoryJobStorage
) -> None:
    """A reusing request whose graph the store already answers submits no
    FlowMesh workflow: every op hits, every output is rebuilt from its stored
    entries, and the schedule names no node."""
    graph = _runtime_graph(_WORKFLOW)
    await _run(
        job_storage,
        _WORKFLOW,
        results={k: _ROWS[k] for k in ("Mark", "Join")},
        job_id="req-1",
    )
    request_info = RequestInfo(
        request_id="req-2",
        runtime_graphs={"g": graph},
        data_profile_graphs={},
        store_records=True,
        reuse_records=True,
        member_reuse={"req-2": True},
    )
    request_info.batch_id = "batch-1"
    request_info.runtime_graph = graph
    request_info.output_node_map = dict.fromkeys(graph.output_node_map, ("g", "x"))
    schedule = Schedule(worker_assignment={"w": list(graph.nodes)})

    async def _never(*_: Any) -> tuple[str, list[str]]:
        raise AssertionError("an all-hit request must submit nothing")

    monkeypatch.setattr(FlowmeshRuntimeManager, "_submit_workflow", _never)

    outputs = await FlowmeshRuntimeManager().process_request(
        request_info, schedule, worker_ids=["w"]
    )

    assert schedule.worker_assignment == {}
    assert outputs["flat_outputs"] == {
        _node(graph, "Join"): [json.dumps(row) for row in _ROWS["Join"]]
    }
    assert [(d["op"], d["decision"]) for d in outputs["reuse_decisions"]] == [
        ("Mark", "hit"),
        ("Join", "hit"),
    ]


def test_a_served_entry_is_data_not_a_stage_input() -> None:
    """A plan entry carrying stored values is read as that column and adds no
    stage to the task spec, so a partial run never waits on the served node."""
    entry: dict[str, Any] = {"kind": "stage", "node": "Mark", "values": [{"id": "a"}]}
    blob = python_step.runtime_blob()
    spec = python_step.flowmesh_spec(
        "Join",
        {
            "code": "def join(c):\n    return c[0]\n",
            "fn_name": "join",
            "plan": [entry],
            "timeout_s": 60,
        },
        {blob.key: blob},
    )
    assert "inputs" not in spec
    assert _column(entry, {}, str) == [str({"id": "a"})]
