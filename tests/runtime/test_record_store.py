"""Per-record store: the writer stores one entry per logical record, keys are
derived downstream, and the lineage read walks a record's inputs from the
archive alone."""

import json
from typing import Any

import pytest
import yaml

from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.python_step import TASK_TYPE
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
_CODE_SECOND = (
    "def second(c):\n"
    "    import json\n"
    "    rows = [json.loads(v) for v in c[0]]\n"
    "    return [{'id': r['id'], 'text': rows_str(r)} for r in rows]\n"
    "\n"
    "def rows_str(r):\n"
    "    return r['text'] + '!'\n"
)

_CODE_CONSTANT = (
    "def constant(c):\n"
    "    import json\n"
    "    rows = [json.loads(v) for v in c[0]]\n"
    "    return [{'id': r['id'], 'text': 'SAME'} for r in rows]\n"
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


def _payload(inputs: dict[str, list[str]] | None = None) -> dict[str, Any]:
    workflow = yaml.safe_load(yaml.safe_dump(_WORKFLOW))
    if inputs is not None:
        workflow["inputs"] = inputs
    return workflow


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


def _batches(inputs: dict[str, list[str]], size: int) -> list[dict[str, list[str]]]:
    """The slices a job with ``input_batch_size`` submits, as the submit path
    cuts them: a single-valued input travels with every slice."""
    lengths = [len(values) for values in inputs.values()]
    total = max(lengths) if lengths else 0
    if total <= size:
        return [inputs]
    batches: list[dict[str, list[str]]] = []
    for start in range(0, total, size):
        batches.append(
            {
                name: values if len(values) == 1 else values[start : start + size]
                for name, values in inputs.items()
            }
        )
    return batches


def _response(values: list[Any]) -> dict[str, Any]:
    return {"items": [{"output": value} for value in values]}


async def _write(
    graph: Any,
    results: dict[str, Any],
    node_ids: dict[str, str],
    *,
    storage: InMemoryJobStorage | None = None,
    round_number: int = 0,
    job_id: str = "req-1",
) -> tuple[InMemoryJobStorage, int]:
    storage = storage or InMemoryJobStorage()
    writer = RecordWriter(storage=storage, enabled=True, logger=_logger())
    node_task_map = {node_ids[name]: f"task-{name}" for name in results}
    task_results = {
        f"task-{name}": _response(values) for name, values in results.items()
    }
    written = await writer.write(
        job_id=job_id,
        member_job_ids=[job_id],
        round_number=round_number,
        runtime_graph=graph,
        node_task_map=node_task_map,
        task_worker={"task-Mark": "worker-1", "task-Join": "worker-2"},
        results=task_results,
    )
    return storage, written


def _logger() -> Any:
    import logging

    return logging.getLogger("test.record_store")


def _entries(storage: InMemoryJobStorage, job_id: str = "req-1") -> list[RecordEntry]:
    index = read_index(storage, job_id)
    assert index is not None
    entries = [read_entry(storage, item.digest) for item in index.entries]
    assert all(entry is not None for entry in entries)
    return [entry for entry in entries if entry is not None]


@pytest.fixture
def graph() -> Any:
    return _runtime_graph(_payload())


def _declared_name(graph: Any, name: str) -> str:
    """The name a record of the DSL op ``name`` carries: the one its workflow
    declares, never the scoped node id."""
    return graph.op_names[_node_id(graph, name)]


@pytest.mark.asyncio
async def test_writer_stores_one_entry_per_keyed_row(graph: Any) -> None:
    node_ids = {"Mark": _node_id(graph, "Mark"), "Join": _node_id(graph, "Join")}
    storage, written = await _write(
        graph,
        {
            "Mark": [{"id": "a", "text": "ONE"}, {"id": "b", "text": "TWO"}],
            "Join": [{"id": "a", "text": "ONE!"}, {"id": "b", "text": "TWO!"}],
        },
        node_ids,
    )
    assert written == 4
    assert sorted(
        (entry.address.op, entry.address.record) for entry in _entries(storage)
    ) == sorted(
        [
            (_declared_name(graph, "Join"), "a"),
            (_declared_name(graph, "Join"), "b"),
            (_declared_name(graph, "Mark"), "a"),
            (_declared_name(graph, "Mark"), "b"),
        ]
    )
    for entry in _entries(storage):
        body, _ = storage.load_blob(entry_object_name(entry.digest))
        assert RecordEntry.from_body(body).digest == entry.digest


@pytest.mark.asyncio
async def test_slices_of_one_job_store_equal_digests() -> None:
    """One keyed job run whole and sliced batch by batch stores the same records
    under the same keys: neither the address nor the spec digest sees the
    slice, whose graph name and so op ids differ."""
    payload = _payload()
    whole = _runtime_graph(payload)
    rows = payload["inputs"]["Rows"]
    per_digest: dict[tuple[str, str | None], str] = {}
    for batch_index, batch in enumerate(_batches(payload["inputs"], size=1)):
        graph = _runtime_graph(_payload(batch), node_prefix=f"slice-{batch_index}")
        node_ids = {
            "Mark": _node_id(graph, "Mark"),
            "Join": _node_id(graph, "Join"),
        }
        model = [{"id": json.loads(row)["id"], "text": "X"} for row in batch["Rows"]]
        storage, _ = await _write(graph, {"Mark": model, "Join": model}, node_ids)
        for entry in _entries(storage):
            key = (entry.address.op, entry.address.record)
            per_digest.setdefault(key, entry.digest)
            assert per_digest[key] == entry.digest
    whole_storage, _ = await _write(
        whole,
        {
            "Mark": [{"id": json.loads(row)["id"], "text": "X"} for row in rows],
            "Join": [{"id": json.loads(row)["id"], "text": "X"} for row in rows],
        },
        {"Mark": _node_id(whole, "Mark"), "Join": _node_id(whole, "Join")},
    )
    whole_digests = {
        (entry.address.op, entry.address.record): entry.digest
        for entry in _entries(whole_storage)
    }
    assert len(whole_digests) == 4
    assert per_digest == whole_digests


@pytest.mark.asyncio
async def test_signature_changes_when_input_value_changes(graph: Any) -> None:
    node_ids = {"Mark": _node_id(graph, "Mark"), "Join": _node_id(graph, "Join")}
    first = _entries(
        (
            await _write(
                graph,
                {
                    "Mark": [{"id": "a", "text": "ONE"}],
                    "Join": [{"id": "a", "text": "ONE!"}],
                },
                node_ids,
            )
        )[0]
    )
    second = _entries(
        (
            await _write(
                graph,
                {
                    "Mark": [{"id": "a", "text": "CHANGED"}],
                    "Join": [{"id": "a", "text": "CHANGED!"}],
                },
                node_ids,
            )
        )[0]
    )

    def digests(entries: list[RecordEntry], op_name: str) -> set[str]:
        return {
            entry.digest
            for entry in entries
            if entry.address.op == _declared_name(graph, op_name)
        }

    # The upstream op read the same literal, so its record keeps its key; the
    # consumer read a different value, so its record gets a new one.
    assert digests(first, "Mark") == digests(second, "Mark")
    assert digests(first, "Join") != digests(second, "Join")


@pytest.mark.asyncio
async def test_consumer_key_changes_when_an_upstream_input_changes() -> None:
    """An input literal reaches a python step through its column plan rather
    than as a dependency, and a record names the value it read. So an upstream
    whose input changed gets a new key even when its value comes out equal, and
    its consumer's key changes with it: the key cuts off by signature, never by
    value."""
    results = {
        "Mark": [{"id": "a", "text": "SAME"}],
        "Join": [{"id": "a", "text": "SAME!"}],
    }
    stored: list[dict[tuple[str, str | None], RecordEntry]] = []
    for literal in ("one", "changed"):
        payload = _payload({"Rows": [json.dumps({"id": "a", "text": literal})]})
        payload["ops"][0].update(fn_name="constant", code=_CODE_CONSTANT)
        graph = _runtime_graph(payload)
        node_ids = {"Mark": _node_id(graph, "Mark"), "Join": _node_id(graph, "Join")}
        storage, _ = await _write(graph, results, node_ids)
        stored.append(
            {
                (entry.address.op, entry.address.record): entry
                for entry in _entries(storage)
            }
        )
    first, second = stored
    mark = _declared_name(graph, "Mark")
    join = _declared_name(graph, "Join")
    assert first[(mark, "a")].value_digest == second[(mark, "a")].value_digest
    assert first[(mark, "a")].digest != second[(mark, "a")].digest
    assert first[(join, "a")].digest != second[(join, "a")].digest


@pytest.mark.asyncio
async def test_signature_changes_when_an_input_literal_changes() -> None:
    """An input literal reaches a python step through its column plan rather
    than as a dependency, and a record still names the value it read, so a
    changed literal gives that record a new key."""
    rows = [json.dumps({"id": "a", "text": "one"})]
    changed = [json.dumps({"id": "a", "text": "one-changed"})]
    stored: dict[str, dict[tuple[str, str | None], str]] = {}
    for label, inputs in (("first", rows), ("second", changed)):
        payload = _payload()
        payload["inputs"]["Rows"] = inputs
        graph = _runtime_graph(payload)
        storage, _ = await _write(
            graph,
            {"Mark": [{"id": "a", "text": "ONE"}]},
            {"Mark": _node_id(graph, "Mark")},
            job_id=f"req-{label}",
        )
        stored[label] = {
            (entry.address.op, entry.address.record): entry.digest
            for entry in _entries(storage, f"req-{label}")
        }

    assert set(stored["first"]) == set(stored["second"])
    assert stored["first"] != stored["second"]


def test_ops_reading_different_paths_get_different_signatures() -> None:
    """Two chat ops whose prompt columns read different paths of one producer
    compute different values, so they never share a signature."""
    graph = _runtime_graph(_PATHS_WORKFLOW)
    names: dict[str, str] = {
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
    node_ids = {"Mark": _node_id(graph, "Mark"), "Join": _node_id(graph, "Join")}
    results = {
        "Mark": [{"id": "a", "text": "ONE"}],
        "Join": [{"id": "a", "text": "ONE!"}],
    }
    first = _entries((await _write(graph, results, node_ids, round_number=1))[0])
    second = _entries((await _write(graph, results, node_ids, round_number=2))[0])
    assert {entry.digest for entry in first}.isdisjoint(
        entry.digest for entry in second
    )


@pytest.mark.asyncio
async def test_lineage_walks_a_chain_from_the_archive(graph: Any) -> None:
    node_ids = {"Mark": _node_id(graph, "Mark"), "Join": _node_id(graph, "Join")}
    storage, _ = await _write(
        graph,
        {
            "Mark": [{"id": "a", "text": "ONE"}],
            "Join": [{"id": "a", "text": "ONE!"}],
        },
        node_ids,
    )
    index = read_index(storage, "req-1")
    assert index is not None
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
async def test_index_outlives_the_server_that_wrote_it(graph: Any) -> None:
    """A second server reads the first one's index: nothing is in process."""
    node_ids = {"Mark": _node_id(graph, "Mark")}
    storage, _ = await _write(graph, {"Mark": [{"id": "a", "text": "ONE"}]}, node_ids)
    fresh = read_index(storage, "req-1")
    assert fresh is not None
    assert fresh.rounds() == [0]
    assert [entry.op for entry in fresh.entries] == [_declared_name(graph, "Mark")]


@pytest.mark.asyncio
async def test_a_record_is_written_once(graph: Any) -> None:
    """A second job that computes the same record adds its index entry but
    leaves the record object as the first job wrote it, so the stored source
    still names where the value was first produced."""
    node_ids = {"Mark": _node_id(graph, "Mark")}
    storage = InMemoryJobStorage()
    results = {"Mark": [{"id": "a", "text": "ONE"}]}
    await _write(graph, results, node_ids, storage=storage, job_id="req-one")
    (first,) = _entries(storage, "req-one")
    await _write(graph, results, node_ids, storage=storage, job_id="req-two")
    (second,) = _entries(storage, "req-two")

    assert first.digest == second.digest
    stored = read_entry(storage, first.digest)
    assert stored is not None
    assert stored.source.job == "req-one"


@pytest.mark.asyncio
async def test_failed_record_keys_on_its_own_inputs() -> None:
    """A failed record keeps its address and never answers a request for a
    success, and two failures on different inputs get different keys, so each
    failure's lineage names the inputs it read."""
    stored: list[RecordEntry] = []
    for literal, join in (
        ("one", {"id": "a", "_fail": {"reason": "boom"}}),
        ("two", {"id": "a", "_fail": {"reason": "boom"}}),
        ("one", {"id": "a"}),
    ):
        payload = _payload({"Rows": [json.dumps({"id": "a", "text": literal})]})
        graph = _runtime_graph(payload)
        node_ids = {"Mark": _node_id(graph, "Mark"), "Join": _node_id(graph, "Join")}
        storage, _ = await _write(
            graph, {"Mark": [{"id": "a", "text": "SAME"}], "Join": [join]}, node_ids
        )
        join_name = _declared_name(graph, "Join")
        stored.append(
            next(entry for entry in _entries(storage) if entry.address.op == join_name)
        )
    first, second, ok = stored
    assert first.status == "failed"
    assert first.address.record == "a"
    assert first.input_digests != second.input_digests
    assert len({first.digest, second.digest, ok.digest}) == 3


@pytest.mark.asyncio
async def test_store_off_writes_nothing(graph: Any) -> None:
    node_ids = {"Mark": _node_id(graph, "Mark")}
    storage = InMemoryJobStorage()
    writer = RecordWriter(storage=storage, enabled=False, logger=_logger())
    written = await writer.write(
        job_id="req-1",
        member_job_ids=["req-1"],
        round_number=0,
        runtime_graph=graph,
        node_task_map={node_ids["Mark"]: "task-Mark"},
        task_worker={},
        results={"task-Mark": _response([{"id": "a"}])},
    )
    assert written == 0
    assert read_index(storage, "req-1") is None


@pytest.mark.asyncio
async def test_unversioned_retrieval_flows_downstream(graph: Any) -> None:
    node_ids = {"Mark": _node_id(graph, "Mark")}
    plan_graph = _runtime_graph(_payload())
    mark = node_ids["Mark"]

    node = plan_graph.nodes[mark]
    plan_graph.nodes[mark] = type(node)(
        node_id=node.node_id,
        task_type="data_retrieval",
        backend=node.backend,
        model=node.model,
        data_spec=node.data_spec,
        model_spec=node.model_spec,
        inference_spec=node.inference_spec,
        api_spec=node.api_spec,
        dependencies=node.dependencies,
    )
    storage, _ = await _write(
        plan_graph, {"Mark": [{"id": "a", "text": "ONE"}]}, node_ids
    )
    assert all(entry.versions.unversioned for entry in _entries(storage))


def test_record_key_rides_the_graph() -> None:
    graph = _runtime_graph(_payload())
    keyed_nodes = {
        node_id: graph.record_keys[node_id]["path"] for node_id in graph.record_keys
    }
    assert sorted(keyed_nodes.values()) == ["id", "id"]
    assert all(graph.nodes[node_id].task_type == TASK_TYPE for node_id in keyed_nodes)
