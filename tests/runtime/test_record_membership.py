"""Membership, deletion, re-route and version checks through the store path.

Every case runs the writer over a finished batch, then plans a second job over
the same archive, so each assertion reads what a real reuse run would decide.
"""

import asyncio
import json
import logging
from typing import Any

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
    tasks = {node_of["Pairs"]: "task-p", node_of["Rank"]: "task-r"}
    results: dict[str, Any] = {
        "task-p": {"items": [{"output": json.dumps(row)} for row in pairs]},
        "task-r": _pairs_items(ranked),
    }
    if "Config" in node_of:
        tasks[node_of["Config"]] = "task-c"
        results["task-c"] = {"items": [{"output": {"bias": 1}}]}
    await RecordWriter(storage=storage, enabled=True, logger=_LOG).write(
        job_id=job_id,
        member_job_ids=[job_id],
        round_number=0,
        runtime_graph=graph,
        node_task_map=tasks,
        task_worker={},
        results=results,
    )


_OTHER_GROUPED = """\
name: groups
inputs: {K: []}
ops:
  - {id: Pairs, op: LambdaOp, inputs: [K], mode: row, fn_name: p,
     key: {node: K, path: pair}, code: "lambda row: json.loads(row[0])"}
  - {id: Weights, op: DataOp, data: ['{"patient": "p1", "w": 1}']}
  - {id: Rank, op: LambdaOp, inputs: [Pairs, Weights], mode: reduce, by: patient,
     fn_name: r, code: "lambda rows: rows"}
outputs: [{name: ranked, ref: Rank}]
"""


def test_an_address_scope_separates_two_workflows_naming_one_op() -> None:
    """Two workflows that both name an op ``Rank`` read their own addresses, so
    the second one's records miss as new ones and never cite the first's entry."""
    storage = InMemoryJobStorage()
    rows = [json.dumps(row) for row in _PAIRS]
    asyncio.run(_run(storage, _graph(_GROUPED, {"K": rows}), _PAIRS, _RANKED, "req-1"))
    other = _graph(_OTHER_GROUPED, {"K": rows})
    ranked = _decisions(storage, other)["Rank"]
    decisions = [
        (outcome.record, outcome.decision, outcome.reason)
        for outcome in ranked.outcomes
    ]
    assert decisions == [
        ("p1", "miss", "no entry"),
        ("p2", "miss", "no entry"),
    ]


def _decisions(storage: InMemoryJobStorage, built: tuple[Any, dict[str, str]]) -> Any:
    plan, _ = ReusePlanner(storage=storage, logger=_LOG).plan(
        runtime_graph=built[0], round_number=0, job_id="req-2"
    )
    return {decision.op: decision for decision in plan.decisions}


def test_a_deleted_member_misses_only_its_group() -> None:
    """Dropping one pair's row misses its group and leaves the other group served."""
    storage = InMemoryJobStorage()
    stored = _graph(_GROUPED, {"K": [json.dumps(row) for row in _PAIRS]})
    asyncio.run(_run(storage, stored, _PAIRS, _RANKED, "req-1"))
    kept = [row for row in _PAIRS if row["pair"] != "p1t2"]
    shorter = _graph(_GROUPED, {"K": [json.dumps(row) for row in kept]})
    ranked = _decisions(storage, shorter)["Rank"]
    assert [
        (outcome.record, outcome.decision, outcome.reason)
        for outcome in ranked.outcomes
    ] == [("p1", "miss", "membership changed"), ("p2", "hit", None)]


_ADDED = [*_PAIRS, {"pair": "p3t1", "patient": "p3", "trial": "t1"}]
_ADDED_RANKED = [*_RANKED, {"output": ["t1"], "key": "p3", "members": [3]}]


def test_an_added_group_is_computed_beside_the_served_ones() -> None:
    """A group this run has and the previous run lacked is a record here: the
    rows the previous run had stay served, and the added group is computed."""
    storage = InMemoryJobStorage()
    asyncio.run(
        _run(
            storage,
            _graph(_GROUPED, {"K": [json.dumps(row) for row in _PAIRS]}),
            _PAIRS,
            _RANKED,
            "req-1",
        )
    )
    added = _graph(_GROUPED, {"K": [json.dumps(row) for row in _ADDED]})
    outcomes = _decisions(storage, added)
    assert [
        (outcome.record, outcome.decision, outcome.reason)
        for outcome in outcomes["Pairs"].outcomes
    ] == [
        ("p1t1", "hit", None),
        ("p1t2", "hit", None),
        ("p2t1", "hit", None),
        ("p3t1", "miss", "no entry"),
    ]
    assert outcomes["Rank"].decision == "miss"
    assert all(
        outcome.reason == "upstream miss: no entry"
        for outcome in outcomes["Rank"].outcomes
    )


_WEIGHTED = """\
name: weighted
inputs: {K: []}
ops:
  - {id: Pairs, op: LambdaOp, inputs: [K], mode: row, fn_name: p,
     key: {node: K, path: pair}, code: "lambda row: json.loads(row[0])"}
  - {id: Config, op: LambdaOp, inputs: [K], mode: list, fn_name: c,
     code: "lambda rows: {'bias': 1}"}
  - {id: Rank, op: LambdaOp, inputs: [Pairs, Config], mode: reduce, by: patient,
     fn_name: r, code: "lambda rows: rows"}
outputs: [{name: ranked, ref: Rank}]
"""


def test_a_reduce_reading_a_whole_input_hits_on_the_next_run() -> None:
    """A reduce with an unkeyed second input reads it once, after its members,
    the way its writer signed it, so a rerun hits every group."""
    storage = InMemoryJobStorage()
    stored = _graph(_WEIGHTED, {"K": [json.dumps(row) for row in _PAIRS]})
    asyncio.run(_run(storage, stored, _PAIRS, _RANKED, "req-1"))
    again = _decisions(storage, stored)["Rank"]
    assert [
        (outcome.record, outcome.decision, outcome.reason) for outcome in again.outcomes
    ] == [("p1", "hit", None), ("p2", "hit", None)]
    """A record whose input missed reports that input's own reason, not a bare
    upstream miss, so a reader sees why the whole chain ran."""
    storage = InMemoryJobStorage()
    asyncio.run(_write_join(storage, [{"id": "a"}], "req-1"))
    graph, _ = _joined([{"id": "a"}, {"id": "b"}])
    plan, _ = ReusePlanner(storage=storage, logger=_LOG).plan(
        runtime_graph=graph, round_number=0, job_id="next"
    )
    joined = next(decision for decision in plan.decisions if decision.op == "M")
    assert joined.decision == "miss"
    assert joined.reason == "upstream miss"
    assert [outcome.reason for outcome in joined.outcomes] == [
        "upstream miss: no entry"
    ]


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


_REROUTED = """\
name: rerouted
inputs: {Docs: []}
ops:
  - {id: D, op: LambdaOp, inputs: [Docs], mode: row, fn_name: r,
     key: {node: Docs, path: id}, code: "lambda row: json.loads(row[0])"}
  - {id: X, op: DataOp, data: %s}
  - {id: Y, op: DataOp, data: %s}
  - {id: M, op: JoinOp, inputs: [D, X, Y], by: id}
outputs: [{name: merged, ref: M}]
"""

_DOCS = [{"id": "a"}, {"id": "b"}]


_ATTACHED = """\
name: attached
inputs: {Docs: []}
ops:
  - {id: D, op: LambdaOp, inputs: [Docs], mode: row, fn_name: r,
     key: {node: Docs, path: id}, code: "lambda row: json.loads(row[0])"}
  - {id: Judge, op: LambdaOp, inputs: [D], mode: row, fn_name: j,
     code: "lambda row: 'ok'"}
  - {id: M, op: JoinOp, inputs: [D, Judge], by: id, attach: {verdict: Judge}}
outputs: [{name: merged, ref: M}]
"""


async def _write_attached(storage: InMemoryJobStorage, job_id: str) -> None:
    """Write one job's finished attach join over ``_DOCS``."""
    graph, node_of = _graph(_ATTACHED, {"Docs": [json.dumps(row) for row in _DOCS]})
    await RecordWriter(storage=storage, enabled=True, logger=_LOG).write(
        job_id=job_id,
        member_job_ids=[job_id],
        round_number=0,
        runtime_graph=graph,
        node_task_map={
            node_of["D"]: "task-d",
            node_of["Judge"]: "task-j",
            node_of["M"]: "task-m",
        },
        task_worker={},
        results={
            "task-d": {"items": [{"output": json.dumps(row)} for row in _DOCS]},
            "task-j": {"items": [{"output": "ok"} for _ in _DOCS]},
            "task-m": {
                "items": [{"output": {**row, "verdict": "ok"}} for row in _DOCS]
            },
        },
    )


def test_an_attach_join_hits_on_the_next_run() -> None:
    """A join that attaches a rowwise op's output reads it at each row's record,
    the way its writer signed it, so a rerun hits every row."""
    storage = InMemoryJobStorage()
    asyncio.run(_write_attached(storage, "req-1"))
    graph, _ = _graph(_ATTACHED, {"Docs": [json.dumps(row) for row in _DOCS]})
    plan, _ = ReusePlanner(storage=storage, logger=_LOG).plan(
        runtime_graph=graph, round_number=0, job_id="next"
    )
    joined = next(decision for decision in plan.decisions if decision.op == "M")
    assert [
        (outcome.record, outcome.decision, outcome.reason)
        for outcome in joined.outcomes
    ] == [("a", "hit", None), ("b", "hit", None)]


def _rerouted(left: list[str], right: list[str]) -> tuple[Any, dict[str, str]]:
    """The graph over ``_DOCS``, with each further input holding its own rows."""
    return _graph(
        _REROUTED % (json.dumps(left), json.dumps(right)),
        {"Docs": [json.dumps(row) for row in _DOCS]},
    )


def _write_reroute(
    storage: InMemoryJobStorage, left: list[str], right: list[str], job_id: str
) -> None:
    """Write one job's finished join over the two further inputs."""
    graph, node_of = _rerouted(left, right)
    asyncio.run(
        RecordWriter(storage=storage, enabled=True, logger=_LOG).write(
            job_id=job_id,
            member_job_ids=[job_id],
            round_number=0,
            runtime_graph=graph,
            node_task_map={node_of["D"]: "task-d", node_of["M"]: "task-m"},
            task_worker={},
            results={
                "task-d": {
                    "items": [{"output": json.dumps({**d, "v": 1})} for d in _DOCS]
                },
                "task-m": {"items": [{"output": {**d, "v": 1}} for d in _DOCS]},
            },
        )
    )


def _reuse_join(storage: InMemoryJobStorage, left: list[str], right: list[str]) -> Any:
    graph, _ = _rerouted(left, right)
    plan, _ = ReusePlanner(storage=storage, logger=_LOG).plan(
        runtime_graph=graph, round_number=0, job_id="next"
    )
    return next(decision for decision in plan.decisions if decision.op == "M")


def test_a_rerouted_row_misses_only_its_own_record() -> None:
    """A row that moved into another join column misses that record alone."""
    storage = InMemoryJobStorage()
    an = '{"id": "a", "v": 1}'
    bn = '{"id": "b", "v": 1}'
    _write_reroute(storage, [bn], [], "req-1")
    decision = _reuse_join(storage, [bn], [an])
    assert [
        (outcome.record, outcome.decision, outcome.reason)
        for outcome in decision.outcomes
    ] == [("a", "miss", "input changed"), ("b", "hit", None)]


_PYTHON_IMAGE = """\
name: imaged
inputs: {K: []}
ops:
  - {id: Pairs, op: LambdaOp, inputs: [K], mode: row, fn_name: p,
     key: {node: K, path: pair}, code: "lambda row: json.loads(row[0])", image: %s}
outputs: [{name: pairs, ref: Pairs}]
"""


async def _write_pairs(
    storage: InMemoryJobStorage,
    built: tuple[Any, dict[str, str]],
    pairs: list[dict[str, Any]],
    job_id: str,
) -> None:
    """Write one job's finished row step over ``pairs``."""
    graph, node_of = built
    await RecordWriter(storage=storage, enabled=True, logger=_LOG).write(
        job_id=job_id,
        member_job_ids=[job_id],
        round_number=0,
        runtime_graph=graph,
        node_task_map={node_of["Pairs"]: "task-p"},
        task_worker={},
        results={
            "task-p": {"items": [{"output": json.dumps(row)} for row in pairs]},
        },
    )


def test_a_changed_op_image_misses_its_records() -> None:
    """A standalone task's declared image is the version its records ran with,
    so a different image misses those records with reason ``version changed``."""
    storage = InMemoryJobStorage()
    rows = {"K": [json.dumps(row) for row in _PAIRS]}
    first = _graph(_PYTHON_IMAGE % "ghcr.io/x/py:one", rows)
    asyncio.run(_write_pairs(storage, first, _PAIRS, "req-1"))
    other = _graph(_PYTHON_IMAGE % "ghcr.io/x/py:two", rows)
    pairs = _decisions(storage, other)["Pairs"]
    assert [
        (outcome.record, outcome.decision, outcome.reason) for outcome in pairs.outcomes
    ] == [
        ("p1t1", "miss", "version changed"),
        ("p1t2", "miss", "version changed"),
        ("p2t1", "miss", "version changed"),
    ]


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
