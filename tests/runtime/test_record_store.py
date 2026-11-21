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


@pytest.fixture
def graph() -> Any:
    return _runtime_graph(_payload())


@pytest.mark.asyncio
async def test_writer_stores_one_entry_per_keyed_row(graph: Any) -> None:
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


@pytest.mark.asyncio
async def test_slices_of_one_job_store_equal_digests() -> None:
    """One keyed job run whole and sliced batch by batch stores the same records
    under the same keys: neither the address nor the spec digest sees the
    slice, whose graph name and so op ids differ."""
    payload = _payload()
    rows = [
        {"id": json.loads(row)["id"], "text": "X"} for row in payload["inputs"]["Rows"]
    ]
    whole = _keys(await _store(_runtime_graph(payload), {"Mark": rows, "Join": rows}))
    assert len(whole) == 4
    for index, row in enumerate(rows):
        graph = _runtime_graph(
            _payload({"Rows": [payload["inputs"]["Rows"][index]]}),
            node_prefix=f"slice-{index}",
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
    stored: dict[str, dict[tuple[str, str | None], RecordEntry]] = {}
    for round_index in ("first", "second"):
        literal = "changed" if constant and round_index == "second" else "one"
        payload = _payload({"Rows": [json.dumps({"id": "a", "text": literal})]})
        if constant:
            payload["ops"][0].update(fn_name="constant", code=_CODE_CONSTANT)
        graph = _runtime_graph(payload)
        text = "SAME" if constant else f"{round_index.upper()}"
        stored[round_index] = {
            (entry.address.op, entry.address.record): entry
            for entry in await _store(
                graph,
                {
                    "Mark": [{"id": "a", "text": text}],
                    "Join": [{"id": "a", "text": f"{text}!"}],
                },
            )
        }
        mark, join = _declared_name(graph, "Mark"), _declared_name(graph, "Join")
    first, second = stored["first"], stored["second"]
    assert (first[(mark, "a")].digest == second[(mark, "a")].digest) is not constant
    assert (
        first[(mark, "a")].value_digest == second[(mark, "a")].value_digest
    ) is constant
    assert first[(join, "a")].digest != second[(join, "a")].digest


@pytest.mark.asyncio
async def test_failed_record_keys_on_its_own_inputs(graph: Any) -> None:
    """A failed record keeps its address and never answers a request for a
    success, and two failures on different inputs get different keys, so each
    failure's lineage names the inputs it read."""
    join = _declared_name(graph, "Join")
    entries = []
    for literal, join_value in (
        ("one", {"id": "a", "_fail": {"reason": "boom"}}),
        ("two", {"id": "a", "_fail": {"reason": "boom"}}),
        ("one", {"id": "a"}),
    ):
        payload = _payload({"Rows": [json.dumps({"id": "a", "text": literal})]})
        results = {"Mark": [{"id": "a", "text": "SAME"}], "Join": [join_value]}
        # Join reads Mark, whose key follows its own literal.
        stored = await _store(_runtime_graph(payload), results)
        entries.append(next(entry for entry in stored if entry.address.op == join))
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
        entries = {
            (entry.address.op, entry.address.record): entry.digest
            for entry in await _store(graph, {"Mark": mark, "Read": [mark]})
        }
        keys.append(
            [
                entries[(_declared_name(graph, "Mark"), "a")],
                entries[(_declared_name(graph, "Read"), None)],
            ]
        )
    assert keys[0][0] == keys[1][0]
    assert keys[0][1] != keys[1][1]


def test_ops_reading_different_paths_get_different_signatures() -> None:
    """Two chat ops whose prompt columns read different paths of one producer
    compute different values, so they never share a signature."""
    graph = _runtime_graph(_PATHS_WORKFLOW)
    names = {
        node_id: graph.op_names.get(node_id, op)
        for op, ids in graph.dsl_to_runtime.items()
        for node_id in ids
    }
    digests = {
        field: op_spec_digest(graph.nodes[_node_id(graph, field)], names)
        for field in ("AskX", "AskY")
    }
    assert digests["AskX"] != digests["AskY"]


@pytest.mark.asyncio
async def test_signature_changes_with_the_round(graph: Any) -> None:
    first = await _store(graph, _rows(), round_number=1)
    second = await _store(graph, _rows(), round_number=2)
    assert {entry.digest for entry in first}.isdisjoint(
        entry.digest for entry in second
    )


@pytest.mark.asyncio
async def test_lineage_walks_a_chain_from_the_archive(graph: Any) -> None:
    """The index is read from the archive alone, so a fresh server sees what an
    earlier one wrote, and a chain walks to each record's producer."""
    storage, _ = await _write(graph, _rows())
    index = read_index(storage, "req-1")
    assert index is not None
    assert index.rounds() == [0]
    chain = lineage(
        storage,
        index=index,
        op=_declared_name(graph, "Join"),
        record="a",
        round_number=0,
    )
    assert [entry.address.record for entry in chain] == ["a", "a"]
    assert chain[0].address.op == _declared_name(graph, "Join")
    assert chain[1].address.op == _declared_name(graph, "Mark")
    assert chain[0].source.task_id == "task-Join"
    assert chain[1].source.worker == "worker-1"


@pytest.mark.asyncio
async def test_a_record_is_written_once(graph: Any) -> None:
    """A second job that computes the same record adds its index entry but
    leaves the record object as the first job wrote it, so the stored source
    still names where the value was first produced."""
    storage = InMemoryJobStorage()
    results = {"Mark": [{"id": "a", "text": "ONE"}]}
    await _write(graph, results, storage=storage, job_id="req-one")
    (first,) = _entries(storage, "req-one")
    await _write(graph, results, storage=storage, job_id="req-two")
    (second,) = _entries(storage, "req-two")
    assert first.digest == second.digest
    stored = read_entry(storage, first.digest)
    assert stored is not None
    assert stored.source.job == "req-one"


@pytest.mark.asyncio
async def test_store_off_writes_nothing(graph: Any) -> None:
    storage, written = await _write(graph, {"Mark": [{"id": "a"}]}, enabled=False)
    assert written == 0
    assert read_index(storage, "req-1") is None


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


def test_record_key_rides_the_graph() -> None:
    graph = _runtime_graph(_payload())
    keyed_nodes = {
        node_id: graph.record_keys[node_id]["path"] for node_id in graph.record_keys
    }
    assert sorted(keyed_nodes.values()) == ["id", "id"]
    assert all(graph.nodes[node_id].task_type == "python" for node_id in keyed_nodes)
