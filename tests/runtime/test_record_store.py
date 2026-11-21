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


def _runtime_graph(payload: dict[str, Any]) -> Any:
    spec = next(iter(parse_yaml_payload(payload).values()))
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    return RuntimeGraphBuilder().build(compiled)


def _node_id(graph: Any, op_suffix: str) -> str:
    return next(node_id for node_id in graph.dsl_to_runtime if op_suffix in node_id)


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


def _op_id(graph: Any, name: str) -> str:
    return next(op_id for op_id in graph.dsl_to_runtime if f"_{name}_" in op_id)


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
            (_op_id(graph, "Join"), "a"),
            (_op_id(graph, "Join"), "b"),
            (_op_id(graph, "Mark"), "a"),
            (_op_id(graph, "Mark"), "b"),
        ]
    )
    for entry in _entries(storage):
        body, _ = storage.load_blob(entry_object_name(entry.digest))
        assert RecordEntry.from_body(body).digest == entry.digest


@pytest.mark.asyncio
async def test_signature_is_equal_across_row_order(graph: Any) -> None:
    node_ids = {"Mark": _node_id(graph, "Mark"), "Join": _node_id(graph, "Join")}
    forward = {
        "Mark": [{"id": "a", "text": "ONE"}, {"id": "b", "text": "TWO"}],
        "Join": [{"id": "a", "text": "ONE!"}, {"id": "b", "text": "TWO!"}],
    }
    reversed_ = {
        "Mark": [{"id": "b", "text": "TWO"}, {"id": "a", "text": "ONE"}],
        "Join": [{"id": "b", "text": "TWO!"}, {"id": "a", "text": "ONE!"}],
    }
    first = {
        (entry.address.op, entry.address.record): entry.digest
        for entry in _entries((await _write(graph, forward, node_ids))[0])
    }
    second = {
        (entry.address.op, entry.address.record): entry.digest
        for entry in _entries((await _write(graph, reversed_, node_ids))[0])
    }
    assert first == second


@pytest.mark.asyncio
async def test_slices_of_one_job_store_equal_digests() -> None:
    """The same keyed job run whole and one row per batch stores the same
    records under the same keys: the spec digest never sees the slice."""
    payload = _payload()
    whole = _runtime_graph(payload)
    rows = payload["inputs"]["Rows"]
    per_digest: dict[tuple[str, str | None], str] = {}
    for batch in _batches(payload["inputs"], size=1):
        graph = _runtime_graph(_payload(batch))
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
        (
            _op_id(whole, entry.address.op.rsplit("_", 2)[1]),
            entry.address.record,
        ): entry.digest
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
        op=_op_id(graph, "Join"),
        record="a",
        round_number=0,
    )
    assert [entry.address.record for entry in chain] == ["a", "a"]
    assert chain[0].address.op == _op_id(graph, "Join")
    assert chain[1].address.op == _op_id(graph, "Mark")
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
    assert [entry.op for entry in fresh.entries] == [_op_id(graph, "Mark")]


@pytest.mark.asyncio
async def test_failed_marker_keeps_its_address(graph: Any) -> None:
    node_ids = {"Mark": _node_id(graph, "Mark")}
    storage, _ = await _write(
        graph, {"Mark": [{"id": "a", "_fail": {"reason": "boom"}}]}, node_ids
    )
    (entry,) = _entries(storage)
    assert entry.status == "failed"
    assert entry.address.record == "a"


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
