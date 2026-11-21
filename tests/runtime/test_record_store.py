"""Per-record store: the writer stores one entry per logical record, and the
lineage read walks a record's input chain back through the ops it read."""

import json
from typing import Any

import pytest
import yaml

from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.python_step import TASK_TYPE
from lumilake_server.runtime.record_store import RecordEntry, entry_object_name
from lumilake_server.runtime.record_writer import RecordStore, RecordWriter, lineage
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.utils.job_storage import InMemoryJobStorage

_CODE_KEYED = (
    "def keyed(c):\n"
    "    import json\n"
    "    rows = [json.loads(v) for v in c[0]]\n"
    "    return [{'id': r['id'], 'text': r['text'].upper()} for r in rows]\n"
)
_CODE_REDUCE = (
    "def joined(c):\n"
    "    import json\n"
    "    left = [json.loads(v) for v in c[0]]\n"
    "    totals = {}\n"
    "    for row in left:\n"
    "        totals.setdefault(row['id'], []).append(row['text'])\n"
    "    return [{'id': k, 'text': '|'.join(v)} for k, v in totals.items()]\n"
)

_WORKFLOW: dict[str, Any] = {
    "name": "keyed",
    "inputs": {
        "Rows": [
            json.dumps({"id": "a", "text": "one"}),
            json.dumps({"id": "b", "text": "two"}),
        ]
    },
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
            "fn_name": "joined",
            "mode": "row",
            "code": _CODE_REDUCE,
            "key": {"node": "Mark", "path": "id"},
        },
    ],
    "outputs": [{"name": "joined", "ref": "Join"}],
}


def _runtime_graph(payload: dict[str, Any]) -> Any:
    spec = next(iter(parse_yaml_payload(payload).values()))
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    return RuntimeGraphBuilder().build(compiled)


def _node_id(graph: Any, op_suffix: str) -> str:
    return next(node_id for node_id in graph.dsl_to_runtime if op_suffix in node_id)


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
) -> tuple[RecordStore, InMemoryJobStorage, int]:
    storage = storage or InMemoryJobStorage()
    store = RecordStore()
    writer = RecordWriter(
        storage=storage, store=store, enabled=True, logger=store.logger
    )
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
    return store, storage, written


def _entries(store: RecordStore, job_id: str = "req-1") -> list[RecordEntry]:
    return store.entries_for(job_id)


@pytest.fixture
def graph() -> Any:
    return _runtime_graph(yaml.safe_load(yaml.safe_dump(_WORKFLOW)))


@pytest.mark.asyncio
async def test_writer_stores_one_entry_per_keyed_row(graph: Any) -> None:
    node_ids = {"Mark": _node_id(graph, "Mark"), "Join": _node_id(graph, "Join")}
    results = {
        "Mark": [
            {"id": "a", "text": "ONE"},
            {"id": "b", "text": "TWO"},
        ],
        "Join": [
            {"id": "a", "text": "ONE"},
            {"id": "b", "text": "TWO"},
        ],
    }
    store, storage, written = await _write(graph, results, node_ids)
    assert written == 4
    assert sorted(
        (entry.address.op, entry.address.record) for entry in _entries(store)
    ) == sorted(
        [
            (_op_id(graph, "Join"), "a"),
            (_op_id(graph, "Join"), "b"),
            (_op_id(graph, "Mark"), "a"),
            (_op_id(graph, "Mark"), "b"),
        ]
    )
    for entry in _entries(store):
        body, _ = storage.load_blob(entry_object_name(entry.digest))
        assert RecordEntry.from_body(body).digest == entry.digest


@pytest.mark.asyncio
async def test_signature_is_equal_across_slice_order(graph: Any) -> None:
    node_ids = {"Mark": _node_id(graph, "Mark"), "Join": _node_id(graph, "Join")}
    forward = {
        "Mark": [{"id": "a", "text": "ONE"}, {"id": "b", "text": "TWO"}],
        "Join": [{"id": "a", "text": "ONE"}, {"id": "b", "text": "TWO"}],
    }
    reversed_ = {
        "Mark": [{"id": "b", "text": "TWO"}, {"id": "a", "text": "ONE"}],
        "Join": [{"id": "b", "text": "TWO"}, {"id": "a", "text": "ONE"}],
    }
    first, _, _ = await _write(graph, forward, node_ids)
    second, _, _ = await _write(graph, reversed_, node_ids)

    def by_address(store: RecordStore) -> dict[tuple[str, str | None], str]:
        return {
            (entry.address.op, entry.address.record): entry.digest
            for entry in _entries(store)
        }

    assert by_address(first) == by_address(second)


@pytest.mark.asyncio
async def test_signature_changes_when_input_value_changes(graph: Any) -> None:
    node_ids = {"Mark": _node_id(graph, "Mark"), "Join": _node_id(graph, "Join")}
    first, _, _ = await _write(
        graph,
        {
            "Mark": [{"id": "a", "text": "ONE"}],
            "Join": [{"id": "a", "text": "ONE"}],
        },
        node_ids,
    )
    second, _, _ = await _write(
        graph,
        {
            "Mark": [{"id": "a", "text": "CHANGED"}],
            "Join": [{"id": "a", "text": "CHANGED"}],
        },
        node_ids,
    )

    def digests(store: RecordStore, op_name: str) -> set[str]:
        return {
            entry.digest
            for entry in _entries(store)
            if entry.address.op == _op_id(graph, op_name)
        }

    # The upstream op read the same literal, so its record keeps its key; the
    # consumer read a different value, so its record gets a new one.
    assert digests(first, "Mark") == digests(second, "Mark")
    assert digests(first, "Join") != digests(second, "Join")


@pytest.mark.asyncio
async def test_signature_changes_with_the_round(graph: Any) -> None:
    node_ids = {"Mark": _node_id(graph, "Mark"), "Join": _node_id(graph, "Join")}
    results = {
        "Mark": [{"id": "a", "text": "ONE"}],
        "Join": [{"id": "a", "text": "ONE"}],
    }
    first, _, _ = await _write(graph, results, node_ids, round_number=0)
    second, _, _ = await _write(graph, results, node_ids, round_number=1)
    assert {entry.digest for entry in _entries(first)}.isdisjoint(
        entry.digest for entry in _entries(second)
    )


@pytest.mark.asyncio
async def test_lineage_walks_a_two_op_chain(graph: Any) -> None:
    node_ids = {"Mark": _node_id(graph, "Mark"), "Join": _node_id(graph, "Join")}
    store, storage, _ = await _write(
        graph,
        {
            "Mark": [{"id": "a", "text": "ONE"}],
            "Join": [{"id": "a", "text": "ONE"}],
        },
        node_ids,
    )
    chain = lineage(
        storage,
        job_id="req-1",
        op=_op_id(graph, "Join"),
        record="a",
        round_number=0,
        entries=_entries(store),
    )
    assert [entry.address.record for entry in chain] == ["a", "a"]
    assert chain[0].address.op == _op_id(graph, "Join")
    assert chain[1].address.op == _op_id(graph, "Mark")
    assert chain[0].source.task_id == "task-Join"
    assert chain[1].source.worker == "worker-1"


def _op_id(graph: Any, name: str) -> str:
    return next(op_id for op_id in graph.dsl_to_runtime if f"_{name}_" in op_id)


@pytest.mark.asyncio
async def test_failed_marker_keeps_its_address(graph: Any) -> None:
    node_ids = {"Mark": _node_id(graph, "Mark")}
    store, _, _ = await _write(
        graph,
        {"Mark": [{"id": "a", "_fail": {"reason": "boom"}}]},
        node_ids,
    )
    (entry,) = _entries(store)
    assert entry.status == "failed"
    assert entry.address.record == "a"


@pytest.mark.asyncio
async def test_store_off_writes_nothing(graph: Any) -> None:
    node_ids = {"Mark": _node_id(graph, "Mark")}
    storage = InMemoryJobStorage()
    store = RecordStore()
    writer = RecordWriter(
        storage=storage, store=store, enabled=False, logger=store.logger
    )
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
    assert _entries(store) == []


@pytest.mark.asyncio
async def test_unversioned_retrieval_flows_downstream(graph: Any) -> None:
    node_ids = {"Mark": _node_id(graph, "Mark")}
    plan_graph = _runtime_graph(yaml.safe_load(yaml.safe_dump(_WORKFLOW)))
    mark = node_ids["Mark"]

    def mark_unversioned(node_id: str) -> None:
        node = plan_graph.nodes[node_id]
        plan_graph.nodes[node_id] = type(node)(
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

    mark_unversioned(mark)
    store, _, _ = await _write(
        plan_graph, {"Mark": [{"id": "a", "text": "ONE"}]}, node_ids
    )
    assert all(entry.versions.unversioned for entry in _entries(store))


def test_record_key_rides_the_graph() -> None:
    graph = _runtime_graph(yaml.safe_load(yaml.safe_dump(_WORKFLOW)))
    keyed_nodes = {
        node_id: key_path
        for node_id, key_path in (
            (node_id, graph.record_keys[node_id]["path"])
            for node_id in graph.record_keys
        )
    }
    assert sorted(keyed_nodes.values()) == ["id", "id"]
    assert all(graph.nodes[node_id].task_type == TASK_TYPE for node_id in keyed_nodes)
