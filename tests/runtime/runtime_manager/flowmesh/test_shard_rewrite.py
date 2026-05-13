import pytest

from lumilake.runtime.runtime_graph import RuntimeGraph
from lumilake.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager
from lumilake.runtime.runtime_ops import RuntimeOp


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


def _two_node_graph(path: str, items: list[str] | None = None) -> RuntimeGraph:
    values = ["a", "b", "c", "d"] if items is None else items
    return _runtime_graph(
        nodes={
            "n0": _op(node_id="n0", data_spec={"type": "list", "items": values}),
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
):
    nodes = graph.to_flowmesh_nodes()
    return manager._rewrite_nodes_for_shard_intent(
        nodes=nodes,
        worker_assignment=worker_assignment,
    )


def test_shard_rewrite_caps_shards_when_items_less_than_workers(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    rewritten = _rewrite(
        flowmesh_manager,
        _two_node_graph("items[0].output", ["a", "b"]),
        {
            "gpu-0": ["n0", "n1"],
            "gpu-1": ["n0", "n1"],
            "gpu-2": ["n0", "n1"],
        },
    )

    rewritten_names = {str(node.get("name")) for node in rewritten.nodes}
    assert "n0__shard_2" not in rewritten_names
    assert "n1__shard_2" not in rewritten_names
    assert rewritten.worker_assignment["gpu-2"] == []
    by_name = {str(node.get("name")): node for node in rewritten.nodes}
    assert by_name["n1__shard_0"].get("dependsOn") == ["n0__shard_0"]
    assert by_name["n1__shard_1"].get("dependsOn") == ["n0__shard_1"]


def test_shard_rewrite_fails_fast_without_rewritable_partition_source(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    graph = _runtime_graph(
        nodes={
            "n0": _op(
                node_id="n0",
                data_spec={
                    "type": "graph_template",
                    "template": {
                        "name": "format",
                        "columns": [
                            {"label": "seed", "data": {"type": "list", "items": ["a"]}}
                        ],
                        "options": {"format": {"template": "{seed}"}},
                    },
                },
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
    with pytest.raises(ValueError, match="reason: unresolved sharded deps"):
        _rewrite(
            flowmesh_manager,
            graph,
            {
                "gpu-0": ["n1"],
                "gpu-1": ["n1"],
                "cpu-0": ["n0"],
            },
        )


def test_shard_rewrite_keeps_non_contracted_dependency_pinned(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    graph = _runtime_graph(
        nodes={
            "n0": _op(node_id="n0", data_spec={"type": "list", "items": ["a"]}),
            "n1": _op(
                node_id="n1",
                data_spec={
                    "type": "graph_template",
                    "template": {
                        "name": "format",
                        "columns": [
                            {"label": "x", "node": "n0", "path": "items.output"},
                            {
                                "label": "seed",
                                "data": {"type": "list", "items": ["p", "q"]},
                            },
                        ],
                        "options": {"format": {"template": "{x}{seed}"}},
                    },
                },
                deps=("n0",),
            ),
        },
        order=["n0", "n1"],
    )
    rewritten = _rewrite(
        flowmesh_manager,
        graph,
        {
            "gpu-0": ["n1"],
            "gpu-1": ["n1"],
            "cpu-0": ["n0"],
        },
    )
    rewritten_names = {str(node.get("name")) for node in rewritten.nodes}
    assert "n1__shard_0" in rewritten_names
    assert "n1__shard_1" in rewritten_names
    assert "n0__shard_0" not in rewritten_names
    by_name = {str(node.get("name")): node for node in rewritten.nodes}
    assert by_name["n1__shard_0"].get("dependsOn") == ["n0"]
    assert by_name["n1__shard_1"].get("dependsOn") == ["n0"]
    template_0 = ((by_name["n1__shard_0"].get("spec") or {}).get("data") or {}).get(
        "template"
    ) or {}
    template_1 = ((by_name["n1__shard_1"].get("spec") or {}).get("data") or {}).get(
        "template"
    ) or {}
    col_0 = (template_0.get("columns") or [])[0]
    col_1 = (template_1.get("columns") or [])[0]
    assert "node" not in col_0 and "path" not in col_0
    assert "node" not in col_1 and "path" not in col_1
    assert ((col_0.get("data") or {}).get("items") or [{}])[0].get(
        "path"
    ) == "items[0].output"
    assert ((col_1.get("data") or {}).get("items") or [{}])[0].get(
        "path"
    ) == "items[0].output"
    assert rewritten.worker_assignment["cpu-0"] == ["n0"]
    assert "n0" not in rewritten.worker_assignment["gpu-0"]
    assert "n0" not in rewritten.worker_assignment["gpu-1"]


def test_shard_rewrite_expands_flowmesh_nodes_for_duplicate_assignment(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    rewritten = _rewrite(
        flowmesh_manager,
        _two_node_graph("items[0].output"),
        {
            "gpu-0": ["n0", "n1"],
            "gpu-1": ["n0"],
        },
    )

    rewritten_names = [str(node.get("name")) for node in rewritten.nodes]
    assert rewritten_names == ["n0__shard_0", "n0__shard_1", "n0", "n1"]
    assert rewritten.worker_assignment == {
        "gpu-0": ["n0__shard_0", "n0", "n1"],
        "gpu-1": ["n0__shard_1"],
    }
    n1 = next(node for node in rewritten.nodes if node.get("name") == "n1")
    assert n1.get("dependsOn") == ["n0"]

    merge = next(node for node in rewritten.nodes if node.get("name") == "n0")
    assert merge.get("dependsOn") == ["n0__shard_0", "n0__shard_1"]
    merge_spec = merge.get("spec") or {}
    assert merge_spec.get("taskType") == "echo"


def test_shard_rewrite_uses_merge_node_for_non_indexed_boundary(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    rewritten = _rewrite(
        flowmesh_manager,
        _two_node_graph("items.output"),
        {
            "gpu-0": ["n0", "n1"],
            "gpu-1": ["n0"],
        },
    )
    n1 = next(node for node in rewritten.nodes if node.get("name") == "n1")
    template = ((n1.get("spec") or {}).get("data") or {}).get("template") or {}
    col = (template.get("columns") or [])[0]
    assert col["node"] == "n0"
    assert col["path"] == "items.output"


def test_inlines_sliced_dep_refs_when_non_contracted_dep_stays_pinned(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    rewritten = _rewrite(
        flowmesh_manager,
        _two_node_graph("items.output", ["a", "b"]),
        {
            "gpu-0": ["n1"],
            "gpu-1": ["n1"],
            "cpu-0": ["n0"],
        },
    )
    by_name = {str(node.get("name")): node for node in rewritten.nodes}
    assert by_name["n1__shard_0"].get("dependsOn") == ["n0"]
    assert by_name["n1__shard_1"].get("dependsOn") == ["n0"]
    cols_0 = (
        (((by_name["n1__shard_0"].get("spec") or {}).get("data") or {}).get("template"))
        or {}
    ).get("columns") or []
    cols_1 = (
        (((by_name["n1__shard_1"].get("spec") or {}).get("data") or {}).get("template"))
        or {}
    ).get("columns") or []
    assert ((cols_0[0].get("data") or {}).get("items") or [{}])[0].get(
        "path"
    ) == "items[0].output"
    assert ((cols_1[0].get("data") or {}).get("items") or [{}])[0].get(
        "path"
    ) == "items[1].output"


def test_shard_rewrite_inlines_unsharded_list_dependency_without_boundary_nodes(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    graph = _runtime_graph(
        nodes={
            "n0": _op(
                node_id="n0",
                data_spec={
                    "type": "graph_template",
                    "template": {
                        "name": "format",
                        "columns": [
                            {"label": "x", "data": {"type": "value", "value": "a"}}
                        ],
                        "options": {"format": {"template": "{x}"}},
                    },
                },
            ),
            "n1": _op(
                node_id="n1",
                data_spec={
                    "type": "graph_template",
                    "template": {
                        "name": "format",
                        "columns": [
                            {"label": "x", "node": "n0", "path": "items.output"},
                            {
                                "label": "seed",
                                "data": {"type": "list", "items": ["p", "q"]},
                            },
                        ],
                        "options": {"format": {"template": "{x}:{seed}"}},
                    },
                },
                deps=("n0",),
            ),
        },
        order=["n0", "n1"],
    )
    rewritten = _rewrite(
        flowmesh_manager,
        graph,
        {
            "gpu-0": ["n1"],
            "gpu-1": ["n1"],
            "cpu-0": ["n0"],
        },
    )
    by_name = {str(node.get("name")): node for node in rewritten.nodes}
    assert by_name["n1__shard_0"].get("dependsOn") == ["n0"]
    assert by_name["n1__shard_1"].get("dependsOn") == ["n0"]
    template_0 = ((by_name["n1__shard_0"].get("spec") or {}).get("data") or {}).get(
        "template"
    ) or {}
    template_1 = ((by_name["n1__shard_1"].get("spec") or {}).get("data") or {}).get(
        "template"
    ) or {}
    col_0 = (template_0.get("columns") or [])[0]
    col_1 = (template_1.get("columns") or [])[0]
    assert "node" not in col_0 and "path" not in col_0
    assert "node" not in col_1 and "path" not in col_1
    assert ((col_0.get("data") or {}).get("items") or [{}])[0].get(
        "path"
    ) == "items[0].output"
    assert ((col_1.get("data") or {}).get("items") or [{}])[0].get(
        "path"
    ) == "items[1].output"
