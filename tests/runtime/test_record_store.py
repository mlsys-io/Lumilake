"""Per-record store: the writer stores one entry per logical record, keys are
derived downstream, and the lineage read walks a record's inputs from the
archive alone."""

import json
import logging
from dataclasses import replace
from typing import Any

import pytest
import yaml

from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload
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

_CODE_KEYED = (
    "def keyed(c):\n"
    "    import json\n"
    "    rows = [json.loads(v) for v in c[0]]\n"
    "    return [{'id': r['id'], 'text': r['text'].upper()} for r in rows]\n"
)
_CODE_CONSTANT = (
    "def constant(c):\n"
    "    import json\n"
    "    rows = [json.loads(v) for v in c[0]]\n"
    "    return [{'id': r['id'], 'text': 'SAME'} for r in rows]\n"
)
_CODE_SECOND = (
    "def second(c):\n"
    "    import json\n"
    "    rows = [json.loads(v) for v in c[0]]\n"
    "    return [{'id': r['id'], 'text': r['text'] + '!'} for r in rows]\n"
)

_ROWS = [
    json.dumps({"id": "a", "text": "one"}),
    json.dumps({"id": "b", "text": "two"}),
]

_WORKFLOW: dict[str, Any] = {
    "name": "keyed",
    "inputs": {"Rows": list(_ROWS)},
    "ops": [
        {
            "id": "Mark",
            "op": "LambdaOp",
            "inputs": ["Rows"],
            "fn_name": "keyed",
            "mode": "row",
            "code": _CODE_KEYED,
            "key": {"node": "Rows", "path": "id"},
        },
        {
            "id": "Join",
            "op": "LambdaOp",
            "inputs": ["Mark"],
            "fn_name": "second",
            "mode": "row",
            "code": _CODE_SECOND,
            "key": {"node": "Mark", "path": "id"},
        },
    ],
    "outputs": [{"name": "joined", "ref": "Join"}],
}

_PATHS_WORKFLOW: dict[str, Any] = {
    "name": "paths",
    "inputs": {"Rows": ['{"id": "a", "x": "1", "y": "2"}']},
    "ops": [
        {
            "id": "Mark",
            "op": "LambdaOp",
            "inputs": ["Rows"],
            "fn_name": "mark",
            "mode": "row",
            "code": "def mark(c):\n    import json\n    return json.loads(c[0][0])\n",
        },
        *[
            {
                "id": f"Ask{field}",
                "op": "LLMChatOp",
                "inputs": ["Mark"],
                "config": {"model": "m"},
                "messages": [{"role": "user", "content": "hi"}],
                "rowwise_template": "{v}",
                "rowwise_columns": [
                    {"label": "v", "node": "Mark", "path": f"items.output.{field}"}
                ],
            }
            for field in ("X", "Y")
        ],
    ],
    "outputs": [{"name": "x", "ref": "AskX"}, {"name": "y", "ref": "AskY"}],
}


def _payload(inputs: dict[str, list[str]] | None = None) -> dict[str, Any]:
    workflow = yaml.safe_load(yaml.safe_dump(_WORKFLOW))
    if inputs is not None:
        workflow["inputs"] = inputs
    return workflow


def _runtime_graph(payload: dict[str, Any], node_prefix: str | None = None) -> Any:
    spec = next(iter(parse_yaml_payload(payload).values()))
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    return RuntimeGraphBuilder().build(compiled, node_prefix=node_prefix)


def _node_id(graph: Any, op_suffix: str) -> str:
    """The one runtime node the DSL op ``op_suffix`` lowered to."""
    return next(
        node_ids[0]
        for op_id, node_ids in graph.dsl_to_runtime.items()
        if f"_{op_suffix}_" in op_id
    )


async def _write(
    graph: Any,
    results: dict[str, Any],
    *,
    storage: InMemoryJobStorage | None = None,
    round_number: int = 0,
    job_id: str = "req-1",
    enabled: bool = True,
) -> tuple[InMemoryJobStorage, int]:
    storage = storage or InMemoryJobStorage()
    writer = RecordWriter(
        storage=storage, enabled=enabled, logger=logging.getLogger("test.record_store")
    )
    written = await writer.write(
        job_id=job_id,
        member_job_ids=[job_id],
        round_number=round_number,
        runtime_graph=graph,
        node_task_map={_node_id(graph, name): f"task-{name}" for name in results},
        task_worker={"task-Mark": "worker-1", "task-Join": "worker-2"},
        results={
            f"task-{name}": {"items": [{"output": value} for value in values]}
            for name, values in results.items()
        },
    )
    return storage, written


def _entries(storage: InMemoryJobStorage, job_id: str = "req-1") -> list[RecordEntry]:
    index = read_index(storage, job_id)
    assert index is not None
    entries = [read_entry(storage, item.digest) for item in index.entries]
    assert all(entry is not None for entry in entries)
    return [entry for entry in entries if entry is not None]


def _keys(entries: list[RecordEntry]) -> dict[tuple[str, str | None], str]:
    return {(entry.address.op, entry.address.record): entry.digest for entry in entries}


async def _store(
    graph: Any, results: dict[str, Any], **kwargs: Any
) -> list[RecordEntry]:
    storage, _ = await _write(graph, results, **kwargs)
    return _entries(storage, kwargs.get("job_id", "req-1"))


def _declared_name(graph: Any, name: str) -> str:
    """The name a record of the DSL op ``name`` carries: the one its workflow
    declares, never the scoped node id."""
    return graph.op_names[_node_id(graph, name)]


def _rows(text: str = "ONE") -> dict[str, list[Any]]:
    return {
        "Mark": [{"id": "a", "text": text}],
        "Join": [{"id": "a", "text": f"{text}!"}],
    }


_BROADCAST_WORKFLOW: dict[str, Any] = {
    **_WORKFLOW,
    "inputs": {"Rows": list(_ROWS), "Cfg": ["shared"]},
    "ops": [
        _WORKFLOW["ops"][0],
        {**_WORKFLOW["ops"][1], "inputs": ["Mark", "Cfg"]},
    ],
}


@pytest.fixture
def graph() -> Any:
    return _runtime_graph(_payload())


@pytest.mark.asyncio
async def test_writer_stores_one_entry_per_keyed_row(graph: Any) -> None:
    """Each keyed row of each op is one entry, both ops declare ``id`` as their
    record key, and a stored object parses back to its own digest."""
    results = {
        "Mark": [{"id": "a", "text": "ONE"}, {"id": "b", "text": "TWO"}],
        "Join": [{"id": "a", "text": "ONE!"}, {"id": "b", "text": "TWO!"}],
    }
    storage, written = await _write(graph, results)
    assert written == 4
    entries = _entries(storage)
    mark, join = _declared_name(graph, "Mark"), _declared_name(graph, "Join")
    assert sorted(_keys(entries)) == [
        (join, "a"),
        (join, "b"),
        (mark, "a"),
        (mark, "b"),
    ]
    for entry in entries:
        body, _ = storage.load_blob(entry_object_name(entry.digest))
        assert RecordEntry.from_body(body).digest == entry.digest
    assert sorted(key["path"] for key in graph.record_keys.values()) == ["id", "id"]


@pytest.mark.asyncio
async def test_slices_of_one_job_store_equal_digests() -> None:
    """One keyed job run whole and sliced batch by batch stores the same records
    under the same keys: neither the address nor the spec digest sees the
    slice, whose graph name and so op ids differ."""
    rows = [{"id": json.loads(row)["id"], "text": "X"} for row in _ROWS]
    whole = _keys(
        await _store(_runtime_graph(_payload()), {"Mark": rows, "Join": rows})
    )
    assert len(whole) == 4
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
    """A consumer's key moves whenever its input record's key or value moves.
    Mark reads the same literal twice, so its key stays and only a changed value
    moves the consumer. An input literal reaches a python step through its
    column plan rather than as a dependency and a record still names the value it
    read, so Mark returns the same value from a changed literal with a new key,
    and the consumer's key moves with it while its own value stays equal."""
    stored = []
    for round_index in ("first", "second"):
        literal = "changed" if constant and round_index == "second" else "one"
        payload = _payload({"Rows": [json.dumps({"id": "a", "text": literal})]})
        if constant:
            payload["ops"][0].update(fn_name="constant", code=_CODE_CONSTANT)
        graph = _runtime_graph(payload)
        text = "SAME" if constant else round_index.upper()
        stored.append(
            {
                (e.address.op, e.address.record): e
                for e in await _store(graph, _rows(text))
            }
        )
        mark, join = _declared_name(graph, "Mark"), _declared_name(graph, "Join")
    first, second = stored
    assert (first[mark, "a"].digest == second[mark, "a"].digest) is not constant
    assert (first[mark, "a"].value_digest == second[mark, "a"].value_digest) is constant
    assert first[join, "a"].digest != second[join, "a"].digest


@pytest.mark.asyncio
async def test_failed_record_keys_on_its_own_inputs(graph: Any) -> None:
    """A failed record keeps its address and never answers a request for a
    success, and two failures on different inputs get different keys, so each
    failure's lineage names the inputs it read. A marker a task returns as JSON
    text is still the marker."""
    join = _declared_name(graph, "Join")
    entries = []
    for literal, value in (
        ("one", {"_fail": {"reason": "boom"}}),
        ("two", json.dumps({"_fail": {"reason": "boom"}})),
        ("one", {"id": "a"}),
    ):
        payload = _payload({"Rows": [json.dumps({"id": "a", "text": literal})]})
        stored = await _store(
            _runtime_graph(payload),
            {"Mark": [{"id": "a", "text": "SAME"}], "Join": [value]},
        )
        entries.append(next(e for e in stored if e.address.op == join))
    first, second, ok = entries
    assert first.status == "failed"
    assert first.address.record == "a"
    assert first.input_digests != second.input_digests
    assert len({first.digest, second.digest, ok.digest}) == 3


@pytest.mark.asyncio
async def test_whole_op_record_names_every_input_row() -> None:
    """A list-mode Lambda reads its whole keyed input, so its one record names
    every input record: a change in the second row gives it a new digest, while
    the first row's own record keeps its key."""
    payload = _payload()
    payload["ops"].append(
        {
            "id": "Read",
            "op": "LambdaOp",
            "inputs": ["Mark"],
            "fn_name": "read",
            "mode": "list",
            "code": "def read(c):\n    return c[0]\n",
        }
    )
    payload["outputs"].append({"name": "read", "ref": "Read"})
    keys = []
    for second in ("TWO", "CHANGED"):
        graph = _runtime_graph(payload)
        mark = [{"id": "a", "text": "ONE"}, {"id": "b", "text": second}]
        keys.append(_keys(await _store(graph, {"Mark": mark, "Read": [mark]})))
    mark_name = _declared_name(graph, "Mark")
    read_name = _declared_name(graph, "Read")
    assert keys[0][mark_name, "a"] == keys[1][mark_name, "a"]
    assert keys[0][read_name, None] != keys[1][read_name, None]


def _paths_graph(**changes: Any) -> Any:
    """The paths workflow with its two chat ops' runtime nodes replaced."""
    graph = _runtime_graph(_PATHS_WORKFLOW)
    for field in ("AskX", "AskY"):
        node_id = _node_id(graph, field)
        graph.nodes[node_id] = replace(graph.nodes[node_id], **changes)
    return graph


@pytest.mark.asyncio
async def test_a_draw_no_seed_reproduces_gets_its_own_key() -> None:
    """An op that samples with no seed draws a value no key reproduces, so each
    run's record names its own production and never shares another run's entry.
    A hosted chat's seed lives in its request body and enters the realization,
    so seeds 7 and 8 are two draws of one op."""
    unseeded = _paths_graph(task_type="inference", inference_spec={"temperature": 0.7})
    listed = []
    for job_id in ("req-one", "req-two"):
        entries = await _store(
            unseeded, {"Mark": [{"id": "a"}], "AskX": ["hi"]}, job_id=job_id
        )
        listed.append(next(e for e in entries if e.address.record is None))
    assert [e.realization.production for e in listed] == [
        "req-one/task-AskX",
        "req-two/task-AskX",
    ]
    assert listed[0].digest != listed[1].digest

    digests = {}
    for seed in (7, 8):
        api_spec = {"json": {"model": "m", "messages": "{{prompt}}", "seed": seed}}
        graph = _paths_graph(
            task_type="api", model_spec={}, inference_spec={}, api_spec=api_spec
        )
        entry = next(
            e
            for e in await _store(graph, {"Mark": [{"id": "a"}], "AskX": ["hi"]})
            if e.address.record is None
        )
        assert entry.realization.seed == seed
        digests[seed] = entry.digest
    assert digests[7] != digests[8]


@pytest.mark.asyncio
async def test_a_seeded_unversioned_record_names_its_production() -> None:
    """A hosted chat reads a source that promises no snapshot, so one seed in
    two jobs names two productions and neither job answers from the other's
    draw; the seed stays beside it as information."""
    api_spec = {"json": {"model": "m", "messages": "{{prompt}}", "seed": 7}}
    graph = _paths_graph(
        task_type="api", model_spec={}, inference_spec={}, api_spec=api_spec
    )
    listed = []
    for job_id in ("req-one", "req-two"):
        entries = await _store(
            graph, {"Mark": [{"id": "a"}], "AskX": ["hi"]}, job_id=job_id
        )
        listed.append(next(e for e in entries if e.address.record is None))
    assert [e.realization.production for e in listed] == [
        "req-one/task-AskX",
        "req-two/task-AskX",
    ]
    assert [e.realization.seed for e in listed] == [7, 7]
    assert listed[0].digest != listed[1].digest


@pytest.mark.asyncio
async def test_broadcast_literal_yields_when_the_row_is_past_it() -> None:
    """A keyed row Lambda that also reads a one-row literal has no second row
    in it: the write takes the literal's whole list instead of failing the
    job, and both rows keep their own entry."""
    graph = _runtime_graph(_BROADCAST_WORKFLOW)
    join = _declared_name(graph, "Join")
    rows = [{"id": "a", "text": "ONE!"}, {"id": "b", "text": "TWO!"}]
    entries = await _store(graph, {"Mark": rows, "Join": rows})
    assert len({entry.digest for entry in entries if entry.address.op == join}) == 2


@pytest.mark.asyncio
async def test_a_batch_writes_and_indexes_per_member_job(graph: Any) -> None:
    """One batch may hold a storing and a non-storing job and two dynamic jobs
    at different rounds: a job that did not ask for the store gets no index,
    and each job that did stores under its own round."""
    storage = InMemoryJobStorage()
    writer = RecordWriter(
        storage=storage, enabled=True, logger=logging.getLogger("test.record_store")
    )
    mark, join = _node_id(graph, "Mark"), _node_id(graph, "Join")
    await writer.write(
        job_id="req-one",
        member_job_ids=["req-one", "req-two", "req-plain"],
        round_number=0,
        runtime_graph=graph,
        node_task_map={mark: "task-Mark", join: "task-Join"},
        task_worker={},
        results={
            task: {"items": [{"output": {"id": "a"}}]}
            for task in ("task-Mark", "task-Join")
        },
        member_store={"req-one": True, "req-two": True, "req-plain": False},
        member_rounds={"req-one": 1, "req-two": 2, "req-plain": 1},
        node_owners={mark: {"req-one"}, join: {"req-two", "req-plain"}},
    )
    first, second = read_index(storage, "req-one"), read_index(storage, "req-two")
    assert first is not None and second is not None
    assert (first.rounds(), second.rounds()) == ([1], [2])
    assert read_index(storage, "req-plain") is None


@pytest.mark.asyncio
async def test_a_node_shared_at_two_rounds_stores_one_entry_each(graph: Any) -> None:
    """Rounds are part of a record's address, so two storing jobs that share a
    deduplicated node at different rounds get one entry each: every index entry
    names the address its own round probes, and the consumer names the inputs
    of its own round."""
    storage = InMemoryJobStorage()
    writer = RecordWriter(
        storage=storage, enabled=True, logger=logging.getLogger("test.record_store")
    )
    mark, join = _node_id(graph, "Mark"), _node_id(graph, "Join")
    await writer.write(
        job_id="req-one",
        member_job_ids=["req-one", "req-two"],
        round_number=0,
        runtime_graph=graph,
        node_task_map={mark: "task-Mark", join: "task-Join"},
        task_worker={},
        results={
            task: {"items": [{"output": {"id": "a"}}]}
            for task in ("task-Mark", "task-Join")
        },
        member_store={"req-one": True, "req-two": True},
        member_rounds={"req-one": 1, "req-two": 2},
        node_owners={mark: {"req-one", "req-two"}, join: {"req-one", "req-two"}},
    )
    by_job: dict[str, dict[str, RecordEntry]] = {}
    for job_id, round_number in (("req-one", 1), ("req-two", 2)):
        index = read_index(storage, job_id)
        assert index is not None
        entries = [read_entry(storage, item.digest) for item in index.entries]
        by_job[job_id] = {entry.digest: entry for entry in entries if entry}
        assert all(
            entry.address.round == round_number for entry in by_job[job_id].values()
        )
    assert set(by_job["req-one"]).isdisjoint(by_job["req-two"])
    join_record = _declared_name(graph, "Join")
    for records_by_digest in by_job.values():
        consumer = next(
            e for e in records_by_digest.values() if e.address.op == join_record
        )
        producer = next(
            e for e in records_by_digest.values() if e.address.op != join_record
        )
        assert consumer.input_digests == [producer.digest]


@pytest.mark.asyncio
async def test_a_stored_value_is_redacted_but_digested_raw(graph: Any) -> None:
    """A response that echoes a credential is stored the way the per-task
    artifact is, redacted, while its value digest stays over the raw value the
    job computed."""
    raw = {"id": "a", "body": {"Authorization": "Bearer sk-live"}}
    (entry,) = await _store(graph, {"Mark": [raw]})
    assert REDACTED_TOKEN_PLACEHOLDER in json.dumps(entry.value)
    assert entry.value_digest == digest_of(raw)


def test_two_chat_ops_reading_different_prompts_differ() -> None:
    """Two ops whose prompt columns read different paths of one producer compute
    different values, so they never share a signature; a structured-output
    schema's own ``items`` stays, so two schemas whose arrays hold different
    element types differ too."""
    graph = _runtime_graph(_PATHS_WORKFLOW)
    names = {
        node_id: graph.op_names.get(node_id, op)
        for op, ids in graph.dsl_to_runtime.items()
        for node_id in ids
    }
    asked = {
        field: op_spec_digest(graph.nodes[_node_id(graph, field)], names)
        for field in ("AskX", "AskY")
    }
    assert asked["AskX"] != asked["AskY"]

    node = graph.nodes[_node_id(graph, "AskX")]
    names = {node.node_id: "AskX"}

    def schema(element: str) -> str:
        env = {
            "response_format": {"json_schema": {"schema": {"items": {"type": element}}}}
        }
        return op_spec_digest(
            replace(node, task_type="api", api_spec={"json": env}), names
        )

    assert schema("string") != schema("integer")


@pytest.mark.asyncio
async def test_signature_changes_with_the_round(graph: Any) -> None:
    first = await _store(graph, _rows(), round_number=1)
    second = await _store(graph, _rows(), round_number=2)
    assert {e.digest for e in first}.isdisjoint(e.digest for e in second)


@pytest.mark.asyncio
async def test_lineage_walks_a_chain_from_the_archive(graph: Any) -> None:
    """The index is read from the archive alone, so a fresh server sees what an
    earlier one wrote, and a chain walks to each record's producer."""
    storage, _ = await _write(graph, _rows())
    index = read_index(storage, "req-1")
    assert index is not None and index.rounds() == [0]
    join, mark = _declared_name(graph, "Join"), _declared_name(graph, "Mark")
    chain = lineage(storage, index=index, op=join, record="a", round_number=0)
    assert [(e.address.op, e.address.record) for e in chain] == [
        (join, "a"),
        (mark, "a"),
    ]
    assert (chain[0].source.task_id, chain[1].source.worker) == (
        "task-Join",
        "worker-1",
    )


@pytest.mark.asyncio
async def test_a_keyed_consumer_of_an_unwritten_node_raises_value_error() -> None:
    """A keyed node owned only by a non-storing member is passed over, so its
    keyed consumer can reach a position with no signature. The write says so
    with a ``ValueError``, which the job catches, so the promise that a store
    write never fails a job holds whatever the ownership invariant does."""
    storage = InMemoryJobStorage()
    writer = RecordWriter(
        storage=storage, enabled=True, logger=logging.getLogger("test.record_store")
    )
    graph = _runtime_graph(_payload())
    mark, join = _node_id(graph, "Mark"), _node_id(graph, "Join")
    with pytest.raises(ValueError, match="has no signature yet"):
        await writer.write(
            job_id="req-one",
            member_job_ids=["req-one", "req-two"],
            round_number=0,
            runtime_graph=graph,
            node_task_map={mark: "task-Mark", join: "task-Join"},
            task_worker={},
            results={
                task: {"items": [{"output": {"id": "a", "text": "ONE"}}]}
                for task in ("task-Mark", "task-Join")
            },
            member_store={"req-one": True, "req-two": False},
            node_owners={mark: {"req-two"}, join: {"req-one"}},
        )


@pytest.mark.asyncio
async def test_a_record_is_written_once(graph: Any) -> None:
    """A second job that computes the same record adds its index entry but
    leaves the record object as the first job wrote it, so the stored source
    still names where the value was first produced."""
    storage = InMemoryJobStorage()
    results = {"Mark": [{"id": "a", "text": "ONE"}]}
    digests = []
    for job_id in ("req-one", "req-two"):
        await _write(graph, results, storage=storage, job_id=job_id)
        digests.append(_entries(storage, job_id)[0].digest)
    assert digests[0] == digests[1]
    stored = read_entry(storage, digests[0])
    assert stored is not None and stored.source.job == "req-one"


@pytest.mark.asyncio
async def test_store_off_writes_nothing(graph: Any) -> None:
    storage, written = await _write(graph, {"Mark": [{"id": "a"}]}, enabled=False)
    assert (written, read_index(storage, "req-1")) == (0, None)


@pytest.mark.asyncio
async def test_unversioned_retrieval_flows_downstream(graph: Any) -> None:
    """A retrieval read records no snapshot, so it and everything it feeds are
    marked unversioned."""
    mark = _node_id(graph, "Mark")
    node = graph.nodes[mark]
    graph.nodes[mark] = replace(
        node, task_type="data_retrieval", model_spec={}, inference_spec={}
    )
    entries = await _store(graph, {"Mark": [{"id": "a", "text": "ONE"}]})
    assert all(entry.versions.unversioned for entry in entries)
