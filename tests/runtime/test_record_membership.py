"""Membership, deletion, re-route and version checks through the store path.

Every case runs the writer over a finished batch, then plans a second job over
the same archive, so each assertion reads what a real reuse run would decide.
"""

import asyncio
import json
import logging
from typing import Any

import pytest

from lumilake import envs
from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.record_writer import RecordWriter
from lumilake_server.runtime.reuse_planner import ReusePlanner
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.utils.job_storage import InMemoryJobStorage

_LOG = logging.getLogger("test.membership")

_PAIRS = [
    {"pair": "p1t1", "patient": "p1", "trial": "t1"},
    {"pair": "p1t2", "patient": "p1", "trial": "t2"},
    {"pair": "p2t1", "patient": "p2", "trial": "t1"},
]
_RANKED = [
    {"output": ["t1", "t2"], "key": "p1", "members": [0, 1]},
    {"output": ["t1"], "key": "p2", "members": [2]},
]

_GROUPED = """\
name: groups
inputs: {K: []}
ops:
  - {id: Pairs, op: LambdaOp, inputs: [K], mode: row, fn_name: p,
     key: {node: K, path: pair}, code: "lambda row: json.loads(row[0])"}
  - {id: Rank, op: LambdaOp, inputs: [Pairs], mode: reduce, by: patient,
     fn_name: r, code: "lambda rows: rows"}
outputs: [{name: ranked, ref: Rank}]
"""


def _graph(
    text: str, inputs: dict[str, list[str]] | None = None
) -> tuple[Any, dict[str, str]]:
    """The runtime graph and each op name's runtime node id."""
    spec = next(iter(parse_yaml_payload(text).values()))
    values = spec["inputs"] if inputs is None else inputs
    runtime = RuntimeGraphBuilder().build(
        Graph.from_json(spec["graph"]).compile(**values), store_records=True
    )
    return runtime, {name: node for node, name in runtime.op_names.items()}


def _items(*rows: Any) -> dict[str, Any]:
    return {"items": [{"output": row} for row in rows]}


def _pairs_items(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The task items a PythonData step emits: ``output``, ``key``, ``members``."""
    return {"items": rows}


async def _run(
    storage: InMemoryJobStorage,
    built: tuple[Any, dict[str, str]],
    pairs: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    job_id: str,
) -> None:
    """Write one job's finished reduce over ``pairs``."""
    graph, node_of = built
    await RecordWriter(storage=storage, enabled=True, logger=_LOG).write(
        job_id=job_id,
        member_job_ids=[job_id],
        round_number=0,
        runtime_graph=graph,
        node_task_map={node_of["Pairs"]: "task-p", node_of["Rank"]: "task-r"},
        task_worker={},
        results={
            "task-p": {"items": [{"output": json.dumps(row)} for row in pairs]},
            "task-r": _pairs_items(ranked),
        },
    )


def _decisions(storage: InMemoryJobStorage, built: tuple[Any, dict[str, str]]) -> Any:
    plan, _ = ReusePlanner(storage=storage, logger=_LOG).plan(
        runtime_graph=built[0], round_number=0, job_id="req-2"
    )
    return {decision.op: decision for decision in plan.decisions}


def test_a_deleted_member_misses_only_its_group() -> None:
    """Dropping one pair's row misses its group and leaves the other group served."""
    storage = InMemoryJobStorage()
    built = _graph(_GROUPED)
    asyncio.run(_run(storage, built, _PAIRS, _RANKED, "req-1"))
    shorter = [
        {"output": ["t1"], "key": "p1", "members": [0]},
        {"output": ["t1"], "key": "p2", "members": [1]},
    ]
    asyncio.run(_run(storage, built, [_PAIRS[0], _PAIRS[2]], shorter, "req-2"))
    reasons = _decisions(storage, built)
    assert reasons["Rank"].decision == "miss"
    assert reasons["Rank"].outcomes[0].record == "p1"


_JOIN = """\
name: joined
inputs: {Docs: []}
ops:
  - {id: D, op: LambdaOp, inputs: [Docs], mode: row, fn_name: r,
     key: {node: Docs, path: id}, code: "lambda row: json.loads(row[0])"}
  - {id: X, op: DataOp, data: ['{"id": "a", "v": 1}', '{"id": "b", "v": 1}']}
  - {id: M, op: JoinOp, inputs: [D, X], by: id}
outputs: [{name: merged, ref: M}]
"""


def _joined(docs: list[dict[str, Any]]) -> tuple[Any, dict[str, str]]:
    return _graph(_JOIN, {"Docs": [json.dumps(row) for row in docs]})


async def _write_join(
    storage: InMemoryJobStorage, docs: list[dict[str, Any]], job_id: str
) -> None:
    """Write one job's finished join over ``docs``."""
    graph, node_of = _joined(docs)
    await RecordWriter(storage=storage, enabled=True, logger=_LOG).write(
        job_id=job_id,
        member_job_ids=[job_id],
        round_number=0,
        runtime_graph=graph,
        node_task_map={node_of["D"]: "task-d", node_of["M"]: "task-m"},
        task_worker={},
        results={
            "task-d": {"items": [{"output": json.dumps({**d, "v": 1})} for d in docs]},
            "task-m": {"items": [{"output": {**d, "v": 1}} for d in docs]},
        },
    )


def _join_decision(storage: InMemoryJobStorage, docs: list[dict[str, Any]]) -> Any:
    graph, _ = _joined(docs)
    plan, _ = ReusePlanner(storage=storage, logger=_LOG).plan(
        runtime_graph=graph, round_number=0, job_id="next"
    )
    return next(decision for decision in plan.decisions if decision.op == "M")


def test_an_added_row_leaves_every_other_record_served() -> None:
    """A key this run has and the previous run lacked misses, and a row the
    previous run already had keeps its own hit."""
    storage = InMemoryJobStorage()
    asyncio.run(_write_join(storage, [{"id": "a"}], "req-1"))
    graph, _ = _joined([{"id": "a"}, {"id": "b"}])
    plan, _ = ReusePlanner(storage=storage, logger=_LOG).plan(
        runtime_graph=graph, round_number=0, job_id="next"
    )
    row = next(decision for decision in plan.decisions if decision.op == "D")
    assert [(o.record, o.decision) for o in row.outcomes] == [
        ("a", "hit"),
        ("b", "miss"),
    ]


def test_a_rerouted_row_misses_only_its_own_record() -> None:
    """A row that moved to another branch misses that record alone."""
    storage = InMemoryJobStorage()
    asyncio.run(_write_join(storage, [{"id": "a"}, {"id": "b"}], "req-1"))
    decision = _join_decision(storage, [{"id": "a"}])
    outcomes = {outcome.record: outcome.decision for outcome in decision.outcomes}
    assert outcomes == {"a": "hit"}
    assert decision.decision == "hit"


def test_a_changed_image_digest_misses_every_dependent_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different runtime image misses every record, saying why."""
    storage = InMemoryJobStorage()
    monkeypatch.setattr(envs, "LUMILAKE_IMAGE_TAG", "img-one")
    asyncio.run(_write_join(storage, [{"id": "a"}], "req-1"))
    monkeypatch.setattr(envs, "LUMILAKE_IMAGE_TAG", "img-two")
    graph, _ = _joined([{"id": "a"}])
    plan, _ = ReusePlanner(storage=storage, logger=_LOG).plan(
        runtime_graph=graph, round_number=0, job_id="next"
    )
    row = next(decision for decision in plan.decisions if decision.op == "D")
    assert [outcome.decision for outcome in row.outcomes] == ["miss"]
    assert row.reason == "version changed"


def test_the_report_lists_each_record_decision_and_its_reason() -> None:
    """The report gives one decision per record, a hit or a miss with why."""
    storage = InMemoryJobStorage()
    asyncio.run(_write_join(storage, [{"id": "a"}], "req-1"))
    graph, _ = _joined([{"id": "a"}, {"id": "b"}])
    plan, _ = ReusePlanner(storage=storage, logger=_LOG).plan(
        runtime_graph=graph, round_number=0, job_id="next"
    )
    row = next(decision for decision in plan.decisions if decision.op == "D")
    assert row.records == 2
    assert [
        (outcome.record, outcome.decision, outcome.reason) for outcome in row.outcomes
    ] == [("a", "hit", None), ("b", "miss", "no entry")]
