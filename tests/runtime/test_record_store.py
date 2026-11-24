"""Per-record store: one entry per logical record, and a lineage read that walks
a record's inputs from the archive alone."""

import asyncio
import json
import logging
import time
from dataclasses import replace
from typing import Any

import pytest
import yaml

from lumilake_server.graphs import Graph
from lumilake_server.ops import RecordKey
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.optimizer.halo import HaloOptimizer
from lumilake_server.runtime.record_store import (
    RecordEntry,
    digest_of,
    entry_object_name,
    op_spec_digest,
)
from lumilake_server.runtime.record_writer import (
    RecordWriter,
    lineage,
    read_entry,
    read_index,
)
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.runtime.sensitive import REDACTED_TOKEN_PLACEHOLDER
from lumilake_server.utils.job_storage import InMemoryJobStorage


def _code(fn: str, body: str) -> str:
    head = f"def {fn}(c):\n    import json\n    return ["
    return f"{head}{body} for r in map(json.loads, c[0])]\n"


_CODE_KEYED = _code("keyed", "{'id': r['id'], 'text': r['text'].upper()}")
_CODE_CONSTANT = _code("constant", "{'id': r['id'], 'text': 'SAME'}")
_CODE_SECOND = _code("second", "{'id': r['id'], 'text': r['text'] + '!'}")

_CODE_SEEDS = "def s(c):\n    return list(range(len(c[0])))\n"
_CODE_LIST = "def keep(c):\n    return c[0]\n"
_ROWS = ['{"id": "a", "text": "one"}', '{"id": "b", "text": "two"}']
_LOG = logging.getLogger("test.record_store")


def _op(op_id: str, op: str = "LambdaOp", **fields: Any) -> dict[str, Any]:
    mode = {} if op != "LambdaOp" else {"mode": "row"}
    return {"id": op_id, "op": op, **mode, **fields}


def _keyed_op(op_id: str, source: str, fn: str, code: str) -> dict[str, Any]:
    key = {"node": source, "path": "id"}
    return _op(op_id, inputs=[source], fn_name=fn, code=code, key=key)


_WORKFLOW: dict[str, Any] = {
    "name": "keyed",
    "inputs": {"Rows": list(_ROWS)},
    "ops": [
        _keyed_op("Mark", "Rows", "keyed", _CODE_KEYED),
        _keyed_op("Join", "Mark", "second", _CODE_SECOND),
    ],
    "outputs": [{"name": "joined", "ref": "Join"}],
}

_PATHS_WORKFLOW: dict[str, Any] = {
    "name": "paths",
    "inputs": {"Rows": ['{"id": "a", "x": "1", "y": "2"}']},
    "ops": [
        _op(
            "Mark",
            inputs=["Rows"],
            fn_name="mark",
            code="def mark(c):\n    import json\n    return json.loads(c[0][0])\n",
        ),
        *[
            _op(
                f"Ask{f}",
                "LLMChatOp",
                inputs=["Mark"],
                config={"model": "m"},
                messages=[{"role": "user", "content": "hi"}],
                rowwise_template="{v}",
                rowwise_columns=[
                    {"label": "v", "node": "Mark", "path": f"items.output.{f}"}
                ],
            )
            for f in ("X", "Y")
        ],
    ],
    "outputs": [{"name": "x", "ref": "AskX"}, {"name": "y", "ref": "AskY"}],
}


def _payload(inputs: dict[str, list[str]] | None = None) -> dict[str, Any]:
    workflow = yaml.safe_load(yaml.safe_dump(_WORKFLOW))
    return workflow if inputs is None else {**workflow, "inputs": inputs}


def _runtime_graph(
    payload: dict[str, Any], node_prefix: str | None = None, *, store: bool = True
) -> Any:
    spec = next(iter(parse_yaml_payload(payload).values()))
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    return RuntimeGraphBuilder().build(
        compiled, node_prefix=node_prefix, store_records=store
    )


def _node_id(graph: Any, op_suffix: str) -> str:
    return next(i[0] for o, i in graph.dsl_to_runtime.items() if f"_{op_suffix}_" in o)


async def _write(
    graph: Any,
    results: dict[str, Any],
    *,
    storage: InMemoryJobStorage | None = None,
    round_number: int = 0,
    job_id: str = "req-1",
    enabled: bool = True,
    raw: dict[str, Any] | None = None,
) -> tuple[InMemoryJobStorage, int]:
    """Store one batch; ``raw`` supplies an op's real response shape."""
    storage = storage or InMemoryJobStorage()
    writer = RecordWriter(storage=storage, enabled=enabled, logger=_LOG)
    task_results = {
        f"task-{n}": {"items": [{"output": v, "value": v} for v in vs]}
        for n, vs in results.items()
    }
    task_results |= {f"task-{n}": v for n, v in (raw or {}).items()}
    return storage, await writer.write(
        job_id=job_id,
        member_job_ids=[job_id],
        round_number=round_number,
        runtime_graph=graph,
        task_worker={},
        results=task_results,
        node_task_map={_node_id(graph, n): f"task-{n}" for n in results},
    )


def _entries(storage: InMemoryJobStorage, job_id: str = "req-1") -> list[RecordEntry]:
    index = read_index(storage, job_id)
    assert index is not None
    return [e for e in (read_entry(storage, i.digest) for i in index.entries) if e]


def _keys(entries: list[RecordEntry]) -> dict[tuple[str, str | None], str]:
    return {(e.address.op, e.address.record): e.digest for e in entries}


async def _store(
    graph: Any, results: dict[str, Any], **kwargs: Any
) -> list[RecordEntry]:
    storage, _ = await _write(graph, results, **kwargs)
    return _entries(storage, kwargs.get("job_id", "req-1"))


async def _write_batch(
    graph: Any,
    nodes: dict[str, str],
    *,
    members: list[str] | None = None,
    rounds: dict[str, int] | None = None,
    store: dict[str, bool] | None = None,
    owners: dict[str, set[str]] | None = None,
) -> InMemoryJobStorage:
    storage = InMemoryJobStorage()
    await RecordWriter(storage=storage, enabled=True, logger=_LOG).write(
        job_id="req-one",
        member_job_ids=["req-one"] if members is None else members,
        round_number=0,
        runtime_graph=graph,
        node_task_map=nodes,
        task_worker={},
        results={
            t: {"items": [{"output": r} for r in _TWO_ROWS]} for t in nodes.values()
        },
        member_store=store,
        member_rounds=rounds,
        node_owners=owners,
    )
    return storage


async def _deduped_batch(
    request_ids: list[str], store: dict[str, bool]
) -> InMemoryJobStorage:
    graphs = {
        f"batch_{i}::{r}": _runtime_graph(
            _payload(), node_prefix=f"batch_{i}", store=store.get(r, False)
        )
        for i, r in enumerate(request_ids)
    }
    graph, _, dedup = HaloOptimizer().optimize_graphs(graphs)
    owners: dict[str, set[str]] = {}
    for group_key, member in graphs.items():
        member_id = group_key.split("::")[1]
        for ids in member.dsl_to_runtime.values():
            for node in ids:
                owners.setdefault(dedup.get(node, node), set()).add(member_id)
    tasks = {node: f"task-{i}" for i, node in enumerate(graph.node_order)}
    return await _write_batch(
        graph, tasks, members=request_ids, store=store, owners=owners
    )


def _indexed(storage: InMemoryJobStorage, job_id: str) -> list[Any]:
    return sorted(_keys(_entries(storage, job_id)))


def _declared_name(graph: Any, name: str) -> str:
    return graph.op_names[_node_id(graph, name)]


_TWO_ROWS = [{"id": "a", "text": "ONE"}, {"id": "b", "text": "TWO"}]

_INDEXED_TWO_ROWS = [("Join", "a"), ("Join", "b"), ("Mark", "a"), ("Mark", "b")]


def _joined(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "text": row["text"] + "!"}


def _rows(text: str = "ONE") -> dict[str, list[Any]]:
    row = {"id": "a", "text": text}
    return {"Mark": [row], "Join": [{**row, "text": text + "!"}]}


_BROADCAST_WORKFLOW: dict[str, Any] = {
    **_WORKFLOW,
    "inputs": {"Rows": list(_ROWS), "Cfg": ["shared"]},
    "ops": [_WORKFLOW["ops"][0], {**_WORKFLOW["ops"][1], "inputs": ["Mark", "Cfg"]}],
}


@pytest.fixture
def graph() -> Any:
    return _runtime_graph(_payload())


@pytest.mark.asyncio
async def test_writer_stores_one_entry_per_keyed_row(graph: Any) -> None:
    """Each keyed row is one entry, and an object parses back to itself."""
    results = {"Mark": _TWO_ROWS, "Join": [_joined(r) for r in _TWO_ROWS]}
    storage, written = await _write(graph, results)
    entries = _entries(storage)
    assert written == 4 == len(entries)
    join, mark = _declared_name(graph, "Join"), _declared_name(graph, "Mark")
    assert sorted(_keys(entries)) == _INDEXED_TWO_ROWS
    body, _ = storage.load_blob(entry_object_name(entries[0].digest))
    assert RecordEntry.from_body(body).digest == entries[0].digest
    index = read_index(storage, "req-1")
    assert index is not None and index.rounds() == [0]
    chain = lineage(storage, index=index, op=join, record="a", round_number=0)
    named = [(e.address.op, e.source.task_id) for e in chain]
    assert named == [(join, "task-Join"), (mark, "task-Mark")]


@pytest.mark.asyncio
async def test_slices_of_one_job_store_equal_digests() -> None:
    """One job run whole and sliced stores the same records under one key."""
    rows = [{"id": json.loads(row)["id"], "text": "X"} for row in _ROWS]
    entries = await _store(_runtime_graph(_payload()), {"Mark": rows, "Join": rows})
    whole = _keys(entries)
    for index, row in enumerate(rows):
        graph = _runtime_graph(
            _payload({"Rows": [_ROWS[index]]}), node_prefix=f"slice-{index}"
        )
        stored = _keys(await _store(graph, {"Mark": [row], "Join": [row]}))
        assert stored == {key: whole[key] for key in stored}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "constant", [False, True], ids=["value-changed", "input-changed"]
)
async def test_signature_follows_the_inputs(constant: bool) -> None:
    """A consumer's key moves with its input record's key or value."""
    stored = []
    for round_index in ("first", "second"):
        literal = "changed" if constant and round_index == "second" else "one"
        payload = _payload({"Rows": [json.dumps({"id": "a", "text": literal})]})
        if constant:
            payload["ops"][0].update(fn_name="constant", code=_CODE_CONSTANT)
        graph = _runtime_graph(payload)
        text = "SAME" if constant else round_index.upper()
        stored.append(_keys(await _store(graph, _rows(text))))
    mark, join = _declared_name(graph, "Mark"), _declared_name(graph, "Join")
    first, second = stored
    assert (first[mark, "a"] == second[mark, "a"]) is not constant
    assert first[join, "a"] != second[join, "a"]
    rounds = [
        {e.digest for e in await _store(graph, _rows(), round_number=n)} for n in (1, 2)
    ]
    assert rounds[0].isdisjoint(rounds[1])


@pytest.mark.asyncio
async def test_failed_record_keys_on_its_own_inputs(graph: Any) -> None:
    """A failed record keeps its address and keys on the inputs it read."""
    join = _declared_name(graph, "Join")
    fail = {"_fail": {"reason": "boom"}}
    cases = [("one", fail), ("two", json.dumps(fail)), ("one", {"id": "a"})]
    entries = []
    for literal, value in cases:
        rows = {"Mark": [{"id": "a", "text": "SAME"}], "Join": [value]}
        graph = _runtime_graph(
            _payload({"Rows": [json.dumps({"id": "a", "text": literal})]})
        )
        entries.append(
            next(e for e in await _store(graph, rows) if e.address.op == join)
        )
    first, second, ok = entries
    assert (first.status, first.address.record) == ("failed", "a")
    assert first.input_digests != second.input_digests
    assert len({first.digest, second.digest, ok.digest}) == 3


@pytest.mark.asyncio
async def test_a_parse_step_keys_each_row_from_its_keyed_input() -> None:
    """A parse step is 1:1, so its rows keep their record keys."""
    payload = _payload()
    payload["ops"].append(_op("P", "ParseOp", inputs=["Mark"], format="json"))
    parsed = {"kind": "x"}
    failed = {"_fail": {"op": "P", "reason": "boom"}}
    rows = {"Mark": _TWO_ROWS, "P": [parsed, failed]}
    found = await _store(_runtime_graph(payload), rows)
    stored = [(e.address.record, e.status) for e in found if e.address.op == "P"]
    assert stored == [("a", "ok"), ("b", "failed")]


def test_an_unlisted_workflow_input_keys_from_any_position() -> None:
    """A workflow input the keyed op does not list still keys it, read last."""
    payload = _payload()
    payload["ops"][0] = {**payload["ops"][0], "key": {"node": "Cases", "path": "id"}}
    payload["inputs"]["Cases"] = ["c1"]
    graph = next(iter(parse_yaml_payload(payload).values()))["graph"]
    last = next(k for k, v in graph.items() if v.get("_declared_id") == "Cases")
    reordered = {k: v for k, v in graph.items() if k != last} | {last: graph[last]}
    built = Graph.from_json(reordered).as_dict()
    mark = next(op for op in built.values() if op.declared_id == "Mark")
    assert mark.key is not None and mark.key.node == last


def test_a_key_on_an_op_it_does_not_read_is_refused() -> None:
    """A key reads its declared op's rows, so naming an unread op is refused."""
    payload = _payload()
    payload["ops"][0] = {**payload["ops"][0], "key": {"node": "Join", "path": "id"}}
    with pytest.raises(
        ValueError, match="op 'Mark' key reads 'Join', which is not one of its inputs"
    ):
        _runtime_graph(payload)


def _list_graph(key: dict[str, str] | None = None) -> Any:
    payload = _payload()
    keep = _op("Keep", inputs=["Mark"], fn_name="keep", mode="list", code=_CODE_LIST)
    payload["ops"].append({**keep, **({} if key is None else {"key": key})})
    payload["outputs"].append({"name": "kept", "ref": "Keep"})
    return _runtime_graph(payload)


@pytest.mark.asyncio
async def test_whole_op_record_names_every_input_row() -> None:
    """A list-mode Lambda's one record names every input row."""
    keys = []
    for second in ("TWO", "CHANGED"):
        graph = _list_graph()
        rows = [*_TWO_ROWS[:1], {**_TWO_ROWS[1], "text": second}]
        keys.append(_keys(await _store(graph, {"Mark": rows, "Keep": [rows]})))
    mark, keep = _declared_name(graph, "Mark"), _declared_name(graph, "Keep")
    assert keys[0][mark, "a"] == keys[1][mark, "a"]
    assert keys[0][keep, None] != keys[1][keep, None]
    broadcast = _runtime_graph(_BROADCAST_WORKFLOW)
    rows = [_joined(r) for r in _TWO_ROWS]
    entries = await _store(broadcast, {"Mark": rows, "Join": rows})
    name = _declared_name(broadcast, "Join")
    assert len({e.digest for e in entries if e.address.op == name}) == 2
    kept = _list_graph({"node": "Mark", "path": "id"})
    storage, _ = await _write(kept, {"Mark": _TWO_ROWS, "Keep": [_TWO_ROWS[1]]})
    name = _declared_name(kept, "Keep")
    assert [k for k in _indexed(storage, "req-1") if k[0] == name] == [(name, "b")]


def _paths_graph(**changes: Any) -> Any:
    graph = _runtime_graph(_PATHS_WORKFLOW)
    for field in ("AskX", "AskY"):
        node = _node_id(graph, field)
        graph.nodes[node] = replace(graph.nodes[node], **changes)
    return graph


def _hosted(seed: int) -> Any:
    body = {"json": {"model": "m", "messages": "{{prompt}}", "seed": seed}}
    return _paths_graph(
        task_type="api", model_spec={}, inference_spec={}, api_spec=body
    )


_HOSTED_ROWS = [{"json": {"choices": [{"message": {"content": t}}]}} for t in "LR"]
_HOSTED = {"task_type": "api", "items": [{"rows": _HOSTED_ROWS}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["op", "input", "literal"])
async def test_a_keyed_chat_writes_one_record_per_row(source: str) -> None:
    """A keyed chat stores one record per row, keyed from a node, an input or a
    literal."""
    if source == "literal":
        rows_in = {"Mark": _TWO_ROWS, "Join": _TWO_ROWS}
        stored = []
        for graph in map(_literal_rows_graph, ("second", "CHANGED")):
            join = _declared_name(graph, "Join")
            entries = await _store(graph, rows_in)
            stored.append(
                {e.address.record: e.digest for e in entries if e.address.op == join}
            )
        assert stored[0]["a"] == stored[1]["a"] and stored[0]["b"] != stored[1]["b"]
        return
    if source == "op":
        graph = _hosted(0)
        mark = _node_id(graph, "Mark")
        graph.record_keys[_node_id(graph, "AskX")] = RecordKey(node=mark, path="id")
        marks = {"Mark": [{"id": "a"}, {"id": "b"}], "AskX": ["LEFT", "RIGHT"]}
        storage, _ = await _write(graph, marks, raw={"AskX": _HOSTED})
        chat = _declared_name(graph, "AskX")
    else:
        spec = _chat({"v": {"data": {"type": "list", "items": list(_ROWS)}}})
        graph = _runtime_graph(
            {
                "name": "chat",
                "inputs": {"Cases": list(_ROWS)},
                "ops": [spec],
                "outputs": [{"name": "o", "ref": "Ask"}],
            }
        )
        storage, _ = await _write(graph, {"Ask": ["LEFT", "RIGHT"]})
        chat = _declared_name(graph, "Ask")
    keys = [k for k in _indexed(storage, "req-1") if k[0] == chat]
    assert keys == [(chat, "a"), (chat, "b")]


async def _asked(graph: Any, job_id: str = "req-1") -> RecordEntry:
    entries = await _store(graph, {"AskX": ["hi"]}, job_id=job_id)
    return next(e for e in entries if e.address.record is None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spec", [{"temperature": 0.7}, {"do_sample": False, "temperature": 0.7}]
)
async def test_a_draw_no_seed_reproduces_gets_its_own_key(spec: dict[str, Any]) -> None:
    """A draw no seed pins names its production."""
    unseeded = _paths_graph(task_type="inference", inference_spec=spec)
    runs = [await _asked(unseeded, job) for job in ("req-one", "req-two")]
    draws = [f"{job}/task-AskX" for job in ("req-one", "req-two")]
    assert [e.realization.production for e in runs] == draws
    by_seed = [await _asked(_hosted(seed)) for seed in (7, 8)]
    assert [e.realization.seed for e in by_seed] == [7, 8]
    seeded = {e.digest for e in by_seed}
    assert len(seeded) == 2 and seeded.isdisjoint(e.digest for e in runs)
    same = [await _asked(_hosted(7), job) for job in ("req-one", "req-two")]
    realized = [(e.realization.seed, e.realization.production) for e in same]
    assert realized == [(7, draw) for draw in draws]


@pytest.mark.asyncio
async def test_a_batch_writes_and_indexes_per_member_job(graph: Any) -> None:
    """Only a storing member gets an index, at its own round."""
    mark, join = _node_id(graph, "Mark"), _node_id(graph, "Join")
    tasks = {mark: "task-Mark", join: "task-Join"}
    members = ["req-one", "req-two", "req-plain"]
    store = {"req-one": True, "req-two": True, "req-plain": False}
    storage = await _write_batch(
        graph,
        tasks,
        members=members,
        rounds={"req-one": 1, "req-two": 2, "req-plain": 1},
        store=store,
        owners={mark: {"req-one"}, join: {"req-two", "req-plain"}},
    )
    indices = [read_index(storage, job) for job in members]
    assert [index.rounds() if index else None for index in indices] == [[1], [2], None]
    rounds = [{e.address.round for e in _entries(storage, j)} for j in members[:2]]
    assert rounds == [{1}, {2}]
    with pytest.raises(ValueError, match="has no signature yet"):
        await _write_batch(
            graph, tasks, owners={mark: {"req-two"}, join: {"req-one"}}, store=store
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("second", [True, False], ids=["two-storing", "mixed"])
async def test_a_coalesced_batch_keeps_its_keys_through_dedupe(second: bool) -> None:
    """Merged nodes keep the key and name the writer reads."""
    store = {"req-one": True, "req-two": second}
    storage = await _deduped_batch(["req-one", "req-two"], store)
    assert _indexed(storage, "req-one") == _INDEXED_TWO_ROWS
    other = _indexed(storage, "req-two") if second else read_index(storage, "req-two")
    assert other == (_INDEXED_TWO_ROWS if second else None)


def test_two_chat_ops_reading_different_prompts_differ() -> None:
    """A different prompt path or schema element type gives a different spec."""
    graph = _runtime_graph(_PATHS_WORKFLOW)
    names = {"X": _declared_name(graph, "AskX"), "Y": _declared_name(graph, "AskY")}

    def asked(field: str, element: str = "none") -> str:
        node = graph.nodes[_node_id(graph, f"Ask{field}")]
        spec = {"json": {"schema": {"items": {"type": element}}}}
        return op_spec_digest(replace(node, task_type="api", api_spec=spec), names)

    assert asked("X") != asked("Y")
    assert asked("X", "string") != asked("X", "integer")


def _chat(sources: dict[str, Any]) -> dict[str, Any]:
    return _op(
        "Ask",
        "LLMChatOp",
        inputs=["Cases"],
        config={"model": "m"},
        messages=[{"role": "user", "content": "hi"}],
        rowwise_template="{v}",
        rowwise_columns=[{"label": label, **data} for label, data in sources.items()],
        key={"node": "Cases", "path": "id"},
    )


def _literal_rows_graph(second: str) -> Any:
    payload = _payload()
    rows = [_ROWS[0], json.dumps({"id": "b", "text": second})]
    payload["ops"][1] = {
        **_chat({"v": {"data": {"type": "list", "items": rows}}}),
        "id": "Join",
        "inputs": ["Mark"],
        "key": {"node": "Rows", "path": "id"},
    }
    return _runtime_graph(payload)


@pytest.mark.asyncio
async def test_a_node_bound_seed_reaches_its_records() -> None:
    """A per-row seed bound to a node's output path lands on each record."""
    payload = _payload()
    payload["ops"].insert(
        1, _op("Seeds", inputs=["Mark"], fn_name="s", code=_CODE_SEEDS)
    )
    payload["ops"][2] = {
        **_chat({"v": {"node": "Mark", "path": "items.output.text"}}),
        "id": "Join",
        "inputs": ["Mark", "Seeds"],
        "rowwise_seed": {"node": "Seeds", "path": "items.output"},
        "key": {"node": "Mark", "path": "id"},
    }
    graph = _runtime_graph(payload)
    rows = {"Mark": _TWO_ROWS, "Seeds": [0, 1], "Join": ["LEFT", "RIGHT"]}
    stored, chat = await _store(graph, rows), _declared_name(graph, "Join")
    seeded = [
        (e.address.record, e.realization.seed) for e in stored if e.address.op == chat
    ]
    assert seeded == [("a", 0), ("b", 1)]


@pytest.mark.asyncio
async def test_a_record_is_written_once(graph: Any) -> None:
    """A second job keeps the first job's object; an unversioned record, a
    store that is off and a redacted value behave as documented."""
    storage, digests = InMemoryJobStorage(), []
    for job in ("req-one", "req-two"):
        await _write(
            graph, {"Mark": [{"id": "a", "text": "ONE"}]}, storage=storage, job_id=job
        )
        digests.append(_entries(storage, job)[0].digest)
    assert digests[0] == digests[1]
    stored = read_entry(storage, digests[0])
    assert stored is not None and stored.source.job == "req-one"
    off, written = await _write(graph, {"Mark": [{"id": "a"}]}, enabled=False)
    assert (written, read_index(off, "req-1")) == (0, None)
    mark = _node_id(graph, "Mark")
    retrieval = {"task_type": "data_retrieval", "model_spec": {}, "inference_spec": {}}
    graph.nodes[mark] = replace(graph.nodes[mark], **retrieval)
    entries = await _store(graph, {"Mark": [{"id": "a", "text": "ONE"}]})
    assert all(e.versions.unversioned for e in entries)
    raw = {"id": "a", "body": {"Authorization": "Bearer sk-live"}}
    (entry,) = await _store(graph, {"Mark": [raw]})
    assert REDACTED_TOKEN_PLACEHOLDER in json.dumps(entry.value)
    assert entry.value_digest == digest_of(raw)


@pytest.mark.asyncio
async def test_two_batches_of_one_job_keep_each_other_s_index_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two batches of one storing job both add to the job's index."""
    save = InMemoryJobStorage.save_blob

    def slow(self: InMemoryJobStorage, k: str, data: bytes, c: str) -> str:
        time.sleep(0.05)
        return save(self, k, data, c)

    monkeypatch.setattr(InMemoryJobStorage, "save_blob", slow)
    graph, storage = _runtime_graph(_payload()), InMemoryJobStorage()
    rows = [{"Mark": [row], "Join": [row]} for row in _TWO_ROWS]
    await asyncio.gather(*(_write(graph, row, storage=storage) for row in rows))
    assert _indexed(storage, "req-1") == _INDEXED_TWO_ROWS


@pytest.mark.parametrize(
    ("key", "message"),
    [("id", "field 'key' Input should"), ({"node": "nope"}, "unknown id")],
)
def test_a_key_that_names_no_node_is_refused(key: Any, message: str) -> None:
    """A native graph's key is a model: a bare string or an unknown node id."""
    payload = _payload()
    payload["ops"][0]["key"] = key
    with pytest.raises(ValueError, match=message):
        _runtime_graph(payload)
