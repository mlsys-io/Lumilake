from lumilake.runtime.runtime_graph import RuntimeGraph, merge_runtime_graphs
from lumilake.runtime.runtime_ops import RuntimeOp


def _op(node_id: str) -> RuntimeOp:
    return RuntimeOp(
        node_id=node_id,
        task_type="data_profiling",
        backend="data_profiling",
        model="data_profiling",
        data_spec={"type": "sql"},
        model_spec={},
        inference_spec={},
    )


def test_merge_keeps_all_dsl_to_runtime_entries_for_duplicate_raw_ids() -> None:
    left = RuntimeGraph(
        nodes={"slice1__sql": _op("slice1__sql")},
        node_order=["slice1__sql"],
        output_node_map={},
        dsl_to_runtime={"shared_sql": ["slice1__sql"]},
    )
    right = RuntimeGraph(
        nodes={"slice2__sql": _op("slice2__sql")},
        node_order=["slice2__sql"],
        output_node_map={},
        dsl_to_runtime={"shared_sql": ["slice2__sql"]},
    )

    merged, _ = merge_runtime_graphs({"left": left, "right": right})

    assert "shared_sql" in merged.dsl_to_runtime
    assert merged.dsl_to_runtime["shared_sql"] == ["slice1__sql", "slice2__sql"]
