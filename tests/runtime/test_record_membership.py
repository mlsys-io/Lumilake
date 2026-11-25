"""Membership, deletion, re-route and version checks through the store path.

Every case runs the writer over a finished batch, then plans a second job over
the same archive, so each assertion reads what a real reuse run would decide.
"""

import asyncio
import json
import logging
from typing import Any

import pytest

from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.record_store import address_object_name
from lumilake_server.runtime.record_writer import RecordWriter
from lumilake_server.runtime.reuse_planner import ReusePlanner, read_address_entry
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
_PAIRS_OP = """\
  - {id: Pairs, op: LambdaOp, inputs: [K], mode: row, fn_name: p,
     key: {node: K, path: pair}, code: "lambda row: json.loads(row[0])"}"""

_GROUPED = f"""\
name: groups
inputs: {{K: []}}
ops:
{_PAIRS_OP}
  - {{id: Rank, op: LambdaOp, inputs: [Pairs], mode: reduce, by: patient,
     fn_name: r, code: "lambda rows: rows"}}
outputs: [{{name: ranked, ref: Rank}}]
"""

_OTHER_GROUPED = f"""\
name: groups
inputs: {{K: []}}
ops:
{_PAIRS_OP}
  - {{id: Weights, op: DataOp, data: ['{{"patient": "p1", "w": 1}}']}}
  - {{id: Rank, op: LambdaOp, inputs: [Pairs, Weights], mode: reduce, by: patient,
     fn_name: r, code: "lambda rows: rows"}}
outputs: [{{name: ranked, ref: Rank}}]
"""

_WEIGHTED = f"""\
name: weighted
inputs: {{K: []}}
ops:
{_PAIRS_OP}
  - {{id: Config, op: LambdaOp, inputs: [K], mode: list, fn_name: c,
     code: "lambda rows: {{'bias': 1}}"}}
  - {{id: Rank, op: LambdaOp, inputs: [Pairs, Config], mode: reduce, by: patient,
     fn_name: r, code: "lambda rows: rows"}}
outputs: [{{name: ranked, ref: Rank}}]
"""

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

_REROUTED = _JOIN.replace(
    '  - {id: X, op: DataOp, data: [\'{"id": "a", "v": 1}\','
    ' \'{"id": "b", "v": 1}\']}',
    "  - {id: X, op: DataOp, data: %s}\n  - {id: Y, op: DataOp, data: %s}",
).replace("inputs: [D, X], by: id", "inputs: [D, X, Y], by: id")

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

_PYTHON_IMAGE = """\
name: imaged
inputs: {K: []}
ops:
  - {id: Pairs, op: LambdaOp, inputs: [K], mode: row, fn_name: p,
     key: {node: K, path: pair}, code: "lambda row: json.loads(row[0])",
     image: %s}
outputs: [{name: pairs, ref: Pairs}]
"""

_DOCS = [{"id": "a"}, {"id": "b"}]


def _graph(text: str, **rows: list[dict[str, Any]]) -> tuple[Any, dict[str, str]]:
    """The runtime graph over each input's rows and each op name's node id."""
    spec = next(iter(parse_yaml_payload(text).values()))
    values = {name: [json.dumps(r) for r in items] for name, items in rows.items()}
    runtime = RuntimeGraphBuilder().build(
        Graph.from_json(spec["graph"]).compile(**values), store_records=True
    )
    return runtime, {name: node for node, name in runtime.op_names.items()}


async def _write(
    storage: InMemoryJobStorage,
    built: tuple[Any, dict[str, str]],
    items: dict[str, Any],
) -> None:
    """Write one job's finished nodes, each op's already-shaped task items."""
    graph, node_of = built
    await RecordWriter(storage=storage, enabled=True, logger=_LOG).write(
        job_id="req-1",
        member_job_ids=["req-1"],
        round_number=0,
        runtime_graph=graph,
        node_task_map={node_of[name]: f"task-{name}" for name in items},
        task_worker={},
        results={f"task-{name}": {"items": rows} for name, rows in items.items()},
    )


def _plan(storage: InMemoryJobStorage, built: tuple[Any, dict[str, str]]) -> Any:
    """Each op's decision on a fresh job over ``built``."""
    plan, _ = ReusePlanner(storage=storage, logger=_LOG).plan(
        runtime_graph=built[0], round_number=0, job_id="req-2"
    )
    return {decision.op: decision for decision in plan.decisions}


def _rows(decision: Any) -> list[tuple[str | None, str, str | None]]:
    """Each record's key, decision and reason, as the report lists them."""
    return [(row.record, row.decision, row.reason) for row in decision.outcomes]


def _pairs(rows: list[dict[str, Any]]) -> list[str]:
    return [json.dumps(row) for row in rows]


def _outputs(values: list[Any]) -> list[dict[str, Any]]:
    """Each value as a task item carrying it under ``output``."""
    return [{"output": value} for value in values]


def test_an_address_scope_separates_two_workflows_naming_one_op() -> None:
    """Two workflows that both name an op ``Rank`` read their own addresses, so
    the second one's records miss as new ones and never cite the first's entry."""
    storage = InMemoryJobStorage()
    asyncio.run(
        _write(
            storage,
            _graph(_GROUPED, K=_PAIRS),
            {"Pairs": _outputs(_pairs(_PAIRS)), "Rank": _RANKED},
        )
    )
    ranked = _plan(storage, _graph(_OTHER_GROUPED, K=_PAIRS))["Rank"]
    assert _rows(ranked) == [("p1", "miss", "no entry"), ("p2", "miss", "no entry")]


def test_a_deleted_member_misses_only_its_group() -> None:
    """Dropping one pair's row misses its group and leaves the other group served."""
    storage = InMemoryJobStorage()
    asyncio.run(
        _write(
            storage,
            _graph(_GROUPED, K=_PAIRS),
            {"Pairs": _outputs(_pairs(_PAIRS)), "Rank": _RANKED},
        )
    )
    kept = [row for row in _PAIRS if row["pair"] != "p1t2"]
    ranked = _plan(storage, _graph(_GROUPED, K=kept))["Rank"]
    assert _rows(ranked) == [("p1", "miss", "membership changed"), ("p2", "hit", None)]


def test_an_added_group_is_computed_beside_the_served_ones() -> None:
    """A group this run has and the previous run lacked is a record here: the
    rows the previous run had stay served, and the added group is computed."""
    storage = InMemoryJobStorage()
    asyncio.run(
        _write(
            storage,
            _graph(_GROUPED, K=_PAIRS),
            {"Pairs": _outputs(_pairs(_PAIRS)), "Rank": _RANKED},
        )
    )
    added = [*_PAIRS, {"pair": "p3t1", "patient": "p3", "trial": "t1"}]
    outcomes = _plan(storage, _graph(_GROUPED, K=added))
    assert _rows(outcomes["Pairs"]) == [
        ("p1t1", "hit", None),
        ("p1t2", "hit", None),
        ("p2t1", "hit", None),
        ("p3t1", "miss", "no entry"),
    ]
    assert {row.reason for row in outcomes["Rank"].outcomes} == {
        "upstream miss: no entry"
    }


def test_a_reduce_reading_a_whole_input_hits_on_the_next_run() -> None:
    """A reduce with an unkeyed second input reads it once, after its members,
    the way its writer signed it, so a rerun hits every group."""
    storage = InMemoryJobStorage()
    built = _graph(_WEIGHTED, K=_PAIRS)
    asyncio.run(
        _write(
            storage,
            built,
            {
                "Pairs": _outputs(_pairs(_PAIRS)),
                "Config": [{"output": {"bias": 1}}],
                "Rank": _RANKED,
            },
        )
    )
    assert _rows(_plan(storage, built)["Rank"]) == [
        ("p1", "hit", None),
        ("p2", "hit", None),
    ]


def test_an_added_row_misses_alone_and_a_dependent_names_its_cause() -> None:
    """A row this run added misses alone; the join above it names the cause it
    missed for, so a reader sees why the chain ran."""
    storage = InMemoryJobStorage()
    asyncio.run(
        _write(
            storage,
            _graph(_JOIN, Docs=[{"id": "a"}]),
            {
                "D": _outputs([json.dumps({"id": "a", "v": 1})]),
                "M": _outputs([{"id": "a", "v": 1}]),
            },
        )
    )
    outcomes = _plan(storage, _graph(_JOIN, Docs=_DOCS))
    assert _rows(outcomes["D"]) == [("a", "hit", None), ("b", "miss", "no entry")]
    assert _rows(outcomes["M"]) == [(None, "miss", "upstream miss: no entry")]
    assert outcomes["D"].records == 2


def test_an_attach_join_hits_on_the_next_run() -> None:
    """A join that attaches a rowwise op's output reads it at each row's record,
    the way its writer signed it, so a rerun hits every row."""
    storage = InMemoryJobStorage()
    built = _graph(_ATTACHED, Docs=_DOCS)
    asyncio.run(
        _write(
            storage,
            built,
            {
                "D": _outputs(_pairs(_DOCS)),
                "Judge": _outputs(["ok", "ok"]),
                "M": _outputs([{**row, "verdict": "ok"} for row in _DOCS]),
            },
        )
    )
    assert _rows(_plan(storage, built)["M"]) == [("a", "hit", None), ("b", "hit", None)]


def test_a_rerouted_row_misses_only_its_own_record() -> None:
    """A row that moved into another join column misses that record alone."""
    storage = InMemoryJobStorage()
    an, bn = '{"id": "a", "v": 1}', '{"id": "b", "v": 1}'
    asyncio.run(
        _write(
            storage,
            _graph(_REROUTED % (json.dumps([bn]), json.dumps([])), Docs=_DOCS),
            {
                "D": _outputs([json.dumps({**d, "v": 1}) for d in _DOCS]),
                "M": _outputs([{**d, "v": 1} for d in _DOCS]),
            },
        )
    )
    decision = _plan(
        storage, _graph(_REROUTED % (json.dumps([bn]), json.dumps([an])), Docs=_DOCS)
    )["M"]
    assert _rows(decision) == [("a", "miss", "input changed"), ("b", "hit", None)]


def test_a_changed_op_image_misses_its_records() -> None:
    """A standalone task's declared image is the version its records ran with,
    so a different image misses those records with reason ``version changed``."""
    storage = InMemoryJobStorage()
    asyncio.run(
        _write(
            storage,
            _graph(_PYTHON_IMAGE % "ghcr.io/x/py:one", K=_PAIRS),
            {"Pairs": _outputs(_pairs(_PAIRS))},
        )
    )
    pairs = _plan(storage, _graph(_PYTHON_IMAGE % "ghcr.io/x/py:two", K=_PAIRS))[
        "Pairs"
    ]
    assert _rows(pairs) == [
        ("p1t1", "miss", "version changed"),
        ("p1t2", "miss", "version changed"),
        ("p2t1", "miss", "version changed"),
    ]


def test_a_corrupt_address_object_raises_instead_of_reading_as_absent() -> None:
    """An address object that does not parse is a fault, not a missing pointer,
    so a probe raises rather than reporting the record as never written."""
    storage = InMemoryJobStorage()
    storage.save_blob(
        address_object_name("scope", "p1", 0), b"not json", "application/json"
    )
    with pytest.raises(ValueError):
        read_address_entry(storage, "scope", "p1", 0)
