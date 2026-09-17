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


def _fanout_graph() -> RuntimeGraph:
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
                            {"label": "x", "node": "n0", "path": "items.output"}
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
    worker_kinds: dict[str, str],
):
    nodes = graph.to_flowmesh_nodes()
    return manager._rewrite_nodes_for_shard_intent(
        nodes=nodes,
        worker_assignment=worker_assignment,
        worker_kinds=worker_kinds,
    )


def test_row_aligned_fanout_shards_across_eligible_workers(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    # HALO assigns the fan-out node n1 to a single worker (gpu-0). With two
    # eligible GPU workers in the pool, the row-aligned fan-out must be sharded
    # across both, not left as a single node on one worker.
    rewritten = _rewrite(
        flowmesh_manager,
        _fanout_graph(),
        {
            "gpu-0": ["n0", "n1"],
            "gpu-1": ["n0"],
        },
        {"gpu-0": "gpu", "gpu-1": "gpu"},
    )

    by_name = {str(node.get("name")): node for node in rewritten.nodes}
    assert "n1__shard_0" in by_name
    assert "n1__shard_1" in by_name
    assert by_name["n1__shard_0"].get("dependsOn") == ["n0__shard_0"]
    assert by_name["n1__shard_1"].get("dependsOn") == ["n0__shard_1"]


def test_row_aligned_fanout_not_sharded_with_single_worker(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    # With only one eligible worker in the pool, the fan-out stays a single node.
    rewritten = _rewrite(
        flowmesh_manager,
        _fanout_graph(),
        {
            "gpu-0": ["n0", "n1"],
        },
        {"gpu-0": "gpu"},
    )

    by_name = {str(node.get("name")): node for node in rewritten.nodes}
    assert "n1__shard_0" not in by_name
    assert "n1" in by_name


def test_row_aligned_fanout_ignores_ineligible_worker_kind(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    # A GPU fan-out node must not be sharded onto a CPU worker.
    rewritten = _rewrite(
        flowmesh_manager,
        _fanout_graph(),
        {
            "gpu-0": ["n0", "n1"],
            "cpu-0": ["n0"],
        },
        {"gpu-0": "gpu", "cpu-0": "cpu"},
    )

    by_name = {str(node.get("name")): node for node in rewritten.nodes}
    assert "n1__shard_0" not in by_name
    assert "n1__shard_1" not in by_name
    assert "n1" in by_name


def test_fanout_merge_node_preserves_row_order(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    # The merge node must concatenate shards in ascending shard order so row
    # alignment is preserved: shard_0 covers rows [0,2), shard_1 covers [2,4).
    base_node = {
        "name": "n0",
        "spec": {
            "taskType": "inference",
            "data": {"type": "list", "items": ["a", "b", "c", "d"]},
        },
    }
    merge = flowmesh_manager._build_merge_node(
        raw_node_id="n0",
        base_node=base_node,
        shard_names=["n0__shard_0", "n0__shard_1"],
        partitions=[(0, 2), (2, 4)],
    )
    assert merge["name"] == "n0"
    assert merge["spec"]["taskType"] == "echo"
    merge_items = merge["spec"]["data"]["items"]
    assert [item["node"] for item in merge_items] == ["n0__shard_0", "n0__shard_1"]
    assert [item["path"] for item in merge_items] == ["items.output", "items.output"]
