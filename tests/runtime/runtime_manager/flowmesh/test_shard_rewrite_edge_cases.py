import pytest

from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _op(
    *,
    node_id: str,
    data_spec: dict[str, object],
    deps: tuple[str, ...] = (),
) -> RuntimeOp:
    return RuntimeOp(
        node_id=node_id,
        task_type="inference",
        backend="vllm",
        model="meta-llama/Llama-3.1-8B-Instruct",
        data_spec=data_spec,
        model_spec={},
        inference_spec={"max_tokens": 32},
        dependencies=deps,
    )


def _runtime_graph(nodes: dict[str, RuntimeOp], order: list[str]) -> RuntimeGraph:
    return RuntimeGraph(
        nodes=nodes,
        node_order=order,
        output_node_map={},
        dsl_to_runtime={},
    )


def _two_node_graph(path: str) -> RuntimeGraph:
    return _runtime_graph(
        nodes={
            "n0": _op(
                node_id="n0", data_spec={"type": "list", "items": ["a", "b", "c", "d"]}
            ),
            "n1": _op(
                node_id="n1",
                data_spec={
                    "type": "graph_template",
                    "template": {
                        "name": "format",
                        "columns": [
                            {
                                "label": "x",
                                "node": "n0",
                                "path": path,
                            }
                        ],
                        "options": {"format": {"template": "{x}"}},
                    },
                },
                deps=("n0",),
            ),
        },
        order=["n0", "n1"],
    )


def _rewrite(
    manager: FlowmeshRuntimeManager,
    graph: RuntimeGraph,
    worker_assignment: dict[str, list[str]],
    *,
    reverse_nodes: bool = False,
):
    nodes = graph.to_flowmesh_nodes()
    if reverse_nodes:
        nodes = list(reversed(nodes))
    return manager._rewrite_nodes_for_shard_intent(
        nodes=nodes,
        worker_assignment=worker_assignment,
    )


def test_shard_rewrite_skips_internal_merge_nodes(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    rewritten = _rewrite(
        flowmesh_manager,
        _two_node_graph("items.output"),
        {
            "gpu-0": ["n0", "n1"],
            "gpu-1": ["n0", "n1"],
        },
    )
    rewritten_names = {str(node.get("name")) for node in rewritten.nodes}
    assert rewritten_names == {
        "n0__shard_0",
        "n0__shard_1",
        "n1__shard_0",
        "n1__shard_1",
    }
    assert rewritten.worker_assignment == {
        "gpu-0": ["n0__shard_0", "n1__shard_0"],
        "gpu-1": ["n0__shard_1", "n1__shard_1"],
    }


def test_shard_rewrite_binds_boundary_merge_node_to_cpu_worker_when_available(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    rewritten = _rewrite(
        flowmesh_manager,
        _two_node_graph("items.output"),
        {
            "gpu-0": ["n0"],
            "gpu-1": ["n0"],
            "cpu-0": ["n1"],
        },
    )
    assert rewritten.worker_assignment["cpu-0"] == ["n1", "n0"]
    assert "n0" not in rewritten.worker_assignment["gpu-0"]
    assert "n0" not in rewritten.worker_assignment["gpu-1"]


def test_shard_rewrite_shards_with_reverse_input_order(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    rewritten = _rewrite(
        flowmesh_manager,
        _two_node_graph("items.output"),
        {
            "gpu-0": ["n0", "n1"],
            "gpu-1": ["n0", "n1"],
        },
        reverse_nodes=True,
    )

    rewritten_names = {str(node.get("name")) for node in rewritten.nodes}
    assert rewritten_names == {
        "n0__shard_0",
        "n0__shard_1",
        "n1__shard_0",
        "n1__shard_1",
    }
    assert set(rewritten.worker_assignment["gpu-0"]) == {"n0__shard_0", "n1__shard_0"}
    assert set(rewritten.worker_assignment["gpu-1"]) == {"n0__shard_1", "n1__shard_1"}


def test_shard_rewrite_fails_fast_on_misaligned_static_cardinalities(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    graph = _runtime_graph(
        nodes={
            "n0": _op(node_id="n0", data_spec={"type": "list", "items": ["a", "b"]}),
            "n1": _op(
                node_id="n1",
                data_spec={
                    "type": "graph_template",
                    "template": {
                        "name": "format",
                        "columns": [
                            {"label": "x", "node": "n0", "path": "items.output"}
                        ],
                        "options": {"format": {"template": "{x}"}},
                    },
                },
                deps=("n0",),
            ),
            "n2": _op(
                node_id="n2", data_spec={"type": "list", "items": ["u", "v", "w", "z"]}
            ),
            "n3": _op(
                node_id="n3",
                data_spec={
                    "type": "graph_template",
                    "template": {
                        "name": "format",
                        "columns": [
                            {"label": "x", "node": "n0", "path": "items.output"}
                        ],
                        "options": {"format": {"template": "{x}"}},
                    },
                },
                deps=("n0", "n2"),
            ),
        },
        order=["n0", "n1", "n2", "n3"],
    )
    with pytest.raises(ValueError, match="reason: cardinality mismatch"):
        _rewrite(
            flowmesh_manager,
            graph,
            {
                "gpu-0": ["n0", "n1", "n2", "n3"],
                "gpu-1": ["n0", "n1", "n2", "n3"],
                "cpu-0": [],
            },
        )


def test_shard_rewrite_allows_independent_non_sharded_branch_cardinality_mismatch(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    graph = _runtime_graph(
        nodes={
            "n0": _op(node_id="n0", data_spec={"type": "list", "items": ["a", "b"]}),
            "n1": _op(
                node_id="n1",
                data_spec={
                    "type": "graph_template",
                    "template": {
                        "name": "format",
                        "columns": [
                            {"label": "x", "node": "n0", "path": "items.output"}
                        ],
                        "options": {"format": {"template": "{x}"}},
                    },
                },
                deps=("n0",),
            ),
            "n2": _op(
                node_id="n2", data_spec={"type": "list", "items": ["u", "v", "w", "z"]}
            ),
            "n3": _op(
                node_id="n3",
                data_spec={
                    "type": "graph_template",
                    "template": {
                        "name": "format",
                        "columns": [
                            {"label": "x", "node": "n2", "path": "items.output"}
                        ],
                        "options": {"format": {"template": "{x}"}},
                    },
                },
                deps=("n2",),
            ),
        },
        order=["n0", "n1", "n2", "n3"],
    )
    rewritten = _rewrite(
        flowmesh_manager,
        graph,
        {
            "gpu-0": ["n0", "n1", "n2", "n3"],
            "gpu-1": ["n0", "n1"],
            "cpu-0": [],
        },
    )
    rewritten_names = {str(node.get("name")) for node in rewritten.nodes}
    assert "n0__shard_0" in rewritten_names
    assert "n1__shard_0" in rewritten_names
    assert "n2__shard_0" not in rewritten_names
    assert "n3__shard_0" not in rewritten_names


def test_shard_rewrite_pins_non_contracted_dependencies_to_single_workers(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    graph = _runtime_graph(
        nodes={
            "n0": _op(node_id="n0", data_spec={"type": "list", "items": ["a", "b"]}),
            "n1": _op(
                node_id="n1",
                data_spec={
                    "type": "graph_template",
                    "template": {
                        "name": "format",
                        "columns": [
                            {"label": "x", "node": "n3", "path": "items.output"},
                            {
                                "label": "seed",
                                "data": {"type": "list", "items": ["left", "right"]},
                            },
                        ],
                        "options": {"format": {"template": "{x}-{seed}"}},
                    },
                },
                deps=("n3",),
            ),
            "n2": _op(
                node_id="n2",
                data_spec={
                    "type": "graph_template",
                    "template": {
                        "name": "format",
                        "columns": [
                            {
                                "label": "a",
                                "node": "n0",
                                "path": "items[0].output",
                            },
                            {
                                "label": "b",
                                "node": "n1",
                                "path": "items[0].output",
                            },
                            {
                                "label": "seed",
                                "data": {"type": "list", "items": ["r0", "r1"]},
                            },
                        ],
                        "options": {"format": {"template": "{a}:{b}:{seed}"}},
                    },
                },
                deps=("n0", "n1"),
            ),
            "n3": _op(node_id="n3", data_spec={"type": "list", "items": ["s"]}),
        },
        order=["n0", "n1", "n2", "n3"],
    )
    rewritten = _rewrite(
        flowmesh_manager,
        graph,
        {
            "gpu-0": ["n2"],
            "gpu-1": ["n2"],
            "cpu-0": ["n0"],
            "cpu-1": ["n1"],
            "cpu-2": ["n3"],
        },
    )

    by_name = {str(node.get("name")): node for node in rewritten.nodes}
    assert "n2__shard_0" in by_name
    assert "n2__shard_1" in by_name
    assert "n1" in by_name
    assert "n0" in by_name
    assert "n1__shard_0" not in by_name
    assert "n0__shard_0" not in by_name
    assert by_name["n2__shard_0"].get("dependsOn") == ["n0", "n1"]
    assert by_name["n2__shard_1"].get("dependsOn") == ["n0", "n1"]

    assert rewritten.worker_assignment["cpu-0"] == ["n0"]
    assert rewritten.worker_assignment["cpu-1"] == ["n1"]
    assert rewritten.worker_assignment["cpu-2"] == ["n3"]
    assert "n0" not in rewritten.worker_assignment["gpu-0"]
    assert "n0" not in rewritten.worker_assignment["gpu-1"]
    assert "n1" not in rewritten.worker_assignment["gpu-0"]
    assert "n1" not in rewritten.worker_assignment["gpu-1"]
