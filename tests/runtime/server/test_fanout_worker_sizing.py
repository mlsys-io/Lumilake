from types import SimpleNamespace

from lumilake_server.runtime.runtime_ops import RuntimeOp


def _op(node_id: str, data_spec: dict[str, object]) -> RuntimeOp:
    return RuntimeOp(
        node_id=node_id,
        task_type="inference",
        backend="vllm",
        model="m",
        data_spec=data_spec,
        model_spec={},
        inference_spec={},
    )


def _graph(nodes: list[RuntimeOp]):
    return SimpleNamespace(
        nodes={op.node_id: op for op in nodes},
        node_order=[op.node_id for op in nodes],
    )


def _batch(graphs):
    return SimpleNamespace(runtime_graphs={f"g{i}": g for i, g in enumerate(graphs)})


def test_static_partition_total_returns_max_list_length(server_factory) -> None:
    server = server_factory()
    spec = {"type": "list", "items": ["a", "b", "c", "d"]}
    assert server._static_partition_total(spec) == 4


def test_static_partition_total_none_for_single_or_no_list(server_factory) -> None:
    server = server_factory()
    assert server._static_partition_total({"type": "list", "items": ["a"]}) == 1
    assert server._static_partition_total({"type": "value", "value": "x"}) is None


def test_static_partition_total_none_for_inconsistent_lengths(server_factory) -> None:
    server = server_factory()
    spec = {
        "type": "graph_template",
        "template": {
            "columns": [
                {"data": {"type": "list", "items": ["a", "b", "c"]}},
                {"data": {"type": "list", "items": ["x", "y"]}},
            ]
        },
    }
    assert server._static_partition_total(spec) is None


def test_batch_fanout_width_is_max_across_graphs(server_factory) -> None:
    server = server_factory()
    batch = _batch(
        [
            _graph([_op("n0", {"type": "list", "items": ["a", "b", "c"]})]),
            _graph([_op("n1", {"type": "list", "items": ["p", "q"]})]),
        ]
    )
    assert server._batch_fanout_width(batch) == 3


def test_batch_fanout_width_defaults_to_one(server_factory) -> None:
    server = server_factory()
    batch = _batch([_graph([_op("n0", {"type": "value", "value": "x"})])])
    assert server._batch_fanout_width(batch) == 1
