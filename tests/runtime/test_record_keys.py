"""Keyed records end to end: retrieval, join, reduce and expand keys."""

import json
import logging
from typing import Any

import pytest

from lumilake import envs
from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.record_store import RecordEntry
from lumilake_server.runtime.record_writer import RecordWriter, read_entry, read_index
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.utils.job_storage import InMemoryJobStorage

_JOINED = """\
name: keys
ops:
  - {id: Trials, op: DataRetrievalOp, data_spec: {type: lumid, mode: sql,
     template: "SELECT nctid, title FROM trials", key: {column: nctid}}}
  - {id: Extra, op: DataOp, data: ['{"nctid": "N1"}']}
  - {id: Merge, op: JoinOp, inputs: [Trials, Extra], by: nctid}
outputs:
  - {name: merged, ref: Merge}
"""

_COMPOSITE = """\
name: composite
ops:
  - {id: T, op: DataRetrievalOp, data_spec: {type: lumid, mode: sql, key: {column: tid},
     template: "SELECT tid, nctid, arm FROM trials"}}
  - {id: E, op: DataRetrievalOp, data_spec: {type: lumid, mode: sql, key: {column: rid},
     template: "SELECT rid, nctid, arm, score FROM scores"}}
  - {id: L, op: DataOp, data: %s}
  - {id: M, op: JoinOp, inputs: [T, E, L], by: [nctid, arm]}
outputs:
  - {name: merged, ref: M}
"""

_BROADCAST = """\
name: broadcast
inputs: {Docs: %s, Extra: %s}
ops:
  - {id: D, op: LambdaOp, inputs: [Docs], mode: row, fn_name: read,
     code: "lambda c: json.loads(c[0])"}
  - {id: E, op: LambdaOp, inputs: [Extra], mode: row, fn_name: read,
     code: "lambda c: json.loads(c[0])"}
  - {id: M, op: JoinOp, inputs: [D, E], by: doc, row_key: [doc, id]}
outputs: [{name: merged, ref: M}]
"""

_UNKEYED = """\
name: unkeyed
ops:
  - id: Trials
    op: DataRetrievalOp
    data_spec: {type: lumid, mode: sql, template: SELECT nctid FROM trials}
outputs:
  - {name: rows, ref: Trials}
"""

_GROUPED = """\
name: groups
inputs: {K: ["a"]}
ops:
  - {id: Pairs, op: LambdaOp, inputs: [K], mode: list, fn_name: p, code: "lambda c: c"}
  - {id: Rank, op: LambdaOp, inputs: [Pairs], mode: reduce, by: patient,
     fn_name: r, code: "lambda rows: rows"}
  - {id: Count, op: LambdaOp, inputs: [Rank], mode: row, fn_name: c,
     code: "lambda row: row"}
outputs: [{name: counted, ref: Count}]
"""

_EXPANDED = """\
name: chain
inputs: {K: ['{"id": "p1"}', '{"id": "p2"}']}
ops:
  - {id: Fill, op: LambdaOp, inputs: [K], mode: row, fn_name: f,
     key: {node: K, path: "id"}, code: "lambda row: json.loads(row[0])"}
  - {id: Split, op: LambdaOp, inputs: [Fill], mode: expand, fn_name: s,
     code: "lambda row: [row[0]['id'] + 'a', row[0]['id'] + 'b']"}
  - {id: Tag, op: LambdaOp, inputs: [Split], mode: row, fn_name: t,
     code: "lambda row: row[0]"}
  - {id: Rank, op: LambdaOp, inputs: [Tag], mode: reduce, by: id, fn_name: r,
     code: "lambda rows: rows"}
outputs: [{name: ranked, ref: Rank}]
"""

_LITERAL_GROUPS = """\
name: literal
inputs: {K: %s}
ops:
  - {id: Rank, op: LambdaOp, inputs: [K], mode: reduce, by: id, fn_name: rank,
     code: "def rank(rows): return [r['id'] for r in rows]"}
outputs:
  - {name: ranked, ref: Rank}
"""

_SPLIT_LITERAL = """\
name: split
inputs: {K: [a, b]}
ops:
  - {id: Split, op: LambdaOp, inputs: [K], mode: expand, fn_name: f,
     code: "lambda row: [row[0]]"}
outputs: [{name: split, ref: Split}]
"""

_ROWS = [
    {"nctid": "N1", "title": "aspirin"},
    {"nctid": "N2", "title": "statins"},
]

_PAIRS = [
    {"patient": "p1", "trial": "t1"},
    {"patient": "p1", "trial": "t2"},
    {"patient": "p2", "trial": "t1"},
]

_GROUPS = [
    {"output": ["t1", "t2"], "key": "p1", "members": [0, 1]},
    {"output": ["t1"], "key": "p2", "members": [2]},
]

_SCORED = [{"id": "p1", "s": 5}, {"id": "p2", "s": 9}]


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "t")


def _graph(text: str) -> tuple[Any, dict[str, str]]:
    """The runtime graph and each declared op's runtime node id."""
    spec = next(iter(parse_yaml_payload(text).values()))
    runtime = RuntimeGraphBuilder().build(
        Graph.from_json(spec["graph"]).compile(**spec["inputs"]), store_records=True
    )
    return runtime, {name: node_id for node_id, name in runtime.op_names.items()}


def _table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    columns: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        for name, value in row.items():
            columns.setdefault(name, {})[str(index)] = value
    return {"table": {"df": json.dumps(columns)}}


def _items(*rows: Any) -> dict[str, Any]:
    """A task's items, one output per row."""
    return {"items": [{"output": row} for row in rows]}


def _rows_text(rows: list[dict[str, Any]]) -> str:
    """Rows as the JSON text a workflow-input literal carries."""
    return repr([json.dumps(row) for row in rows])


async def _write(
    built: tuple[Any, dict[str, str]], tasks: dict[str, Any], run: str = "req-1"
) -> InMemoryJobStorage:
    graph, node_of = built
    storage = InMemoryJobStorage()
    await RecordWriter(
        storage=storage, enabled=True, logger=logging.getLogger("test.keys")
    ).write(
        job_id=run,
        member_job_ids=[run],
        round_number=0,
        runtime_graph=graph,
        node_task_map={node_of[op]: f"task-{node_of[op]}-{run}" for op in tasks},
        task_worker={},
        results={f"task-{node_of[op]}-{run}": value for op, value in tasks.items()},
    )
    return storage


def _entries(storage: InMemoryJobStorage, run: str = "req-1") -> list[RecordEntry]:
    index = read_index(storage, run)
    assert index is not None
    found = [read_entry(storage, item.digest) for item in index.entries]
    return [entry for entry in found if entry is not None]


def _records(entries: list[RecordEntry], op: str) -> dict[str, RecordEntry]:
    """One op's keyed entries by key."""
    return {
        entry.address.record: entry
        for entry in entries
        if entry.address.op == op and entry.address.record is not None
    }


async def _records_of(
    built: tuple[Any, dict[str, str]],
    tasks: dict[str, Any],
    op: str,
    run: str = "req-1",
) -> dict[str, RecordEntry]:
    """One op's entries from a fresh write of ``tasks``."""
    return _records(_entries(await _write(built, tasks, run), run), op)


@pytest.mark.asyncio
async def test_an_unkeyed_retrieval_is_versioned_by_its_value() -> None:
    """An unkeyed retrieval is one whole-op record versioned by its value."""
    entries = _entries(
        await _write(_graph(_UNKEYED), {"Trials": {"items": [_table(_ROWS)]}})
    )
    (entry,) = [e for e in entries if e.address.op == "Trials"]
    assert entry.address.record is None
    assert entry.versions.value_version == entry.value_digest


@pytest.mark.asyncio
async def test_a_keyed_retrieval_writes_one_entry_per_key() -> None:
    """A keyed retrieval writes one entry per row key, stable across runs."""
    graph = _graph(_JOINED)
    first = await _records_of(graph, {"Trials": {"items": [_table(_ROWS)]}}, "Trials")
    again = await _records_of(
        graph, {"Trials": {"items": [_table(_ROWS)]}}, "Trials", "req-2"
    )
    assert sorted(first) == ["N1", "N2"]
    assert all(entry.versions.unversioned for entry in first.values())
    assert {k: e.digest for k, e in first.items()} == {
        k: e.digest for k, e in again.items()
    }
    amended = await _records_of(
        graph,
        {"Trials": {"items": [_table([_ROWS[0], {"nctid": "N2", "title": "s!"}])]}},
        "Trials",
    )
    assert amended["N1"].digest == first["N1"].digest
    assert amended["N2"].digest != first["N2"].digest


@pytest.mark.asyncio
async def test_a_join_reads_each_further_input_at_its_own_join_key() -> None:
    """Each further input is read at its own row for the output's ``by`` values."""
    trials = [
        {"tid": "t1", "nctid": "N1", "arm": "a"},
        {"tid": "t2", "nctid": "N1", "arm": "b"},
        {"tid": "t3", "nctid": "N2", "arm": "a"},
    ]
    scores = [
        {"rid": "r9", "nctid": "N1", "arm": "b", "score": 4},
        {"rid": "r8", "nctid": "N1", "arm": "a", "score": 3},
        {"rid": "r7", "nctid": "N2", "arm": "a", "score": 7},
    ]
    notes = [{"nctid": "N1", "arm": "a", "note": "x"}, {"nctid": "N2", "arm": "a"}]
    merged = [{**row, "score": score} for row, score in zip(trials, (3, 4, 7))]

    async def stored(
        extra: list[dict[str, Any]], note_rows: list[dict[str, Any]], run: str
    ) -> dict[str, RecordEntry]:
        return await _records_of(
            _graph(_COMPOSITE % _rows_text(note_rows)),
            {
                "T": {"items": [_table(trials)]},
                "E": {"items": [_table(extra)]},
                "M": _items(*merged),
            },
            "M",
            run,
        )

    base = await stored(scores, notes, "req-1")
    assert sorted(base) == ['["N1","a"]', '["N1","b"]', '["N2","a"]']
    amended = [{**scores[0], "score": 9}, *scores[1:]]
    after = await stored(amended, notes, "req-2")
    assert [key for key in base if after[key].digest != base[key].digest] == [
        '["N1","b"]'
    ]
    noted = await stored(scores, [{**notes[0], "note": "z"}, notes[1]], "req-3")
    assert [key for key in base if noted[key].digest != base[key].digest] == [
        '["N1","a"]'
    ]
    reversed_ = await stored(scores, notes[::-1], "req-4")
    assert all(reversed_[key].digest == base[key].digest for key in base)
    further = await _records_of(
        _graph(_COMPOSITE % _rows_text(notes)),
        {"E": {"items": [_table(amended)]}},
        "E",
        "req-2",
    )
    assert after['["N1","b"]'].input_digests[1] == further["r9"].digest


@pytest.mark.asyncio
async def test_a_broadcast_join_keys_each_row_by_its_row_key() -> None:
    """Rows of one ``by`` value stay apart: each is one record at its row_key."""
    docs = [{"id": 1, "doc": "a"}, {"id": 2, "doc": "a"}]
    merged = [{"id": 1, "doc": "a", "n": 1}, {"id": 2, "doc": "a", "n": 1}]
    entries = await _records_of(
        _graph(_BROADCAST % (_rows_text(docs), _rows_text([{"doc": "a", "n": 1}]))),
        {"D": _items(*docs), "M": _items(*merged)},
        "M",
    )
    assert sorted(entries) == ['["a",1]', '["a",2]']
    assert len({entry.digest for entry in entries.values()}) == 2


@pytest.mark.asyncio
async def test_a_reduce_group_names_its_members_and_feeds_a_lambda() -> None:
    """A reduce group is its ``by`` value and names the members it read."""
    graph = _graph(_GROUPED)
    tasks = {
        "Pairs": _items(*_PAIRS),
        "Rank": {"items": _GROUPS},
        "Count": _items({"n": 2}, {"n": 1}),
    }
    stored = await _records_of(graph, tasks, "Rank")
    assert [(key, len(set(entry.member_keys))) for key, entry in stored.items()] == [
        ("p1", 2),
        ("p2", 1),
    ]
    assert sorted(await _records_of(graph, tasks, "Count")) == ["p1", "p2"]


@pytest.mark.asyncio
async def test_a_reduce_over_a_workflow_input_reads_the_rows_it_grouped() -> None:
    """A group's signature names the rows it grouped, so one changed row moves
    only its group's digest, and a key ``1`` and a key ``"1"`` collide."""
    grouped = [
        {"output": ["p1", "p1"], "key": "p1", "members": [0, 2]},
        {"output": ["p2"], "key": "p2", "members": [1]},
    ]
    tasks = {"Rank": {"items": grouped}}
    first = await _records_of(
        _graph(_LITERAL_GROUPS % _rows_text([*_SCORED, {"id": "p1", "s": 7}])),
        tasks,
        "Rank",
    )
    changed = await _records_of(
        _graph(_LITERAL_GROUPS % _rows_text([*_SCORED, {"id": "p1", "s": 8}])),
        tasks,
        "Rank",
    )
    assert changed["p1"].digest != first["p1"].digest
    assert changed["p2"].digest == first["p2"].digest
    with pytest.raises(ValueError, match="two records under key '1'"):
        await _records_of(
            _graph(_LITERAL_GROUPS % _rows_text([{"id": 1}, {"id": "1"}])),
            {
                "Rank": {
                    "items": [
                        {"output": [1], "key": 1, "members": [0]},
                        {"output": [2], "key": "1", "members": [1]},
                    ]
                }
            },
            "Rank",
        )


@pytest.mark.asyncio
async def test_an_expand_chain_keys_its_lambda_and_names_its_reduce_members() -> None:
    """An expand child's key survives a Lambda and a reduce, and an expand with
    no keyed input stores one whole-op record."""
    graph = _graph(_EXPANDED)
    children = [
        {"output": f"p{n}{s}", "parent": n - 1, "child": c}
        for n, s, c in ((1, "a", 0), (1, "b", 1), (2, "a", 0))
    ]
    grouped = [
        {"output": [2], "key": child["output"], "members": [i]}
        for i, child in enumerate(children)
    ]
    one = await _write(
        graph,
        {
            "Fill": _items({"id": "p1"}, {"id": "p2"}),
            "Split": {"items": children},
            "Tag": _items(*(child["output"] for child in children)),
            "Rank": {"items": grouped},
        },
    )
    entries = _entries(one)
    assert sorted(_records(entries, "Tag")) == ["p1\x1f0", "p1\x1f1", "p2\x1f0"]
    ranked = _records(entries, "Rank")
    assert sorted(ranked) == ["p1a", "p1b", "p2a"]
    assert ranked["p1a"].member_keys == ["p1\x1f0"]
    literal = _entries(
        await _write(_graph(_SPLIT_LITERAL), {"Split": _items("a", "b")})
    )
    assert [e.address.record for e in literal if e.address.op == "Split"] == [None]
    with pytest.raises(ValueError, match="op 'Split' keys its own records"):
        _graph(_SPLIT_LITERAL.replace("code:", "key: {node: K, path: id}, code:"))
