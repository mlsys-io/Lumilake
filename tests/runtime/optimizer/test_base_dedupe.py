from lumilake_server.runtime.optimizer.halo import HaloOptimizer
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _build_scoped_graph(scope: str) -> RuntimeGraph:
    root_id = f"{scope}__root"
    child_id = f"{scope}__child"
    root_label = f"input_{scope}_Stock_1001"
    child_label = f"context_{scope}_root_2002"

    root = RuntimeOp(
        node_id=root_id,
        task_type="inference",
        backend="vllm",
        model="meta-llama/Llama-3.1-8B-Instruct",
        data_spec={
            "type": "graph_template",
            "template": {
                "name": "format",
                "columns": [
                    {
                        "label": root_label,
                        "data": {"type": "list", "items": ["NVDA"]},
                    }
                ],
                "options": {"format": {"template": f"Stock={{{root_label}}}"}},
            },
        },
        model_spec={},
        inference_spec={"echo": True},
        dependencies=(),
    )
    child = RuntimeOp(
        node_id=child_id,
        task_type="inference",
        backend="vllm",
        model="meta-llama/Llama-3.1-8B-Instruct",
        data_spec={
            "type": "graph_template",
            "template": {
                "name": "format",
                "columns": [
                    {
                        "label": child_label,
                        "node": root_id,
                        "path": "items.output",
                    }
                ],
                "options": {"format": {"template": f"Ctx={{{child_label}}}"}},
            },
        },
        model_spec={},
        inference_spec={"echo": True},
        dependencies=(root_id,),
    )
    return RuntimeGraph(
        nodes={root_id: root, child_id: child},
        node_order=[root_id, child_id],
        output_node_map={},
        dsl_to_runtime={},
    )


def test_optimize_graphs_dedupes_scoped_label_variants() -> None:
    optimizer = HaloOptimizer()
    runtime_graphs = {
        "graph_a": _build_scoped_graph("A"),
        "graph_b": _build_scoped_graph("B"),
    }

    merged, _ = optimizer.optimize_graphs(runtime_graphs)

    assert merged.node_count == 2


def test_optimize_graphs_dedupes_prefix_only_differences() -> None:
    optimizer = HaloOptimizer()

    def _graph(scope: str) -> RuntimeGraph:
        seed_id = f"{scope}__seed"
        infer_id = f"{scope}__infer"
        return RuntimeGraph(
            nodes={
                seed_id: RuntimeOp(
                    node_id=seed_id,
                    task_type="inference",
                    backend="dummy",
                    model="dummy-model",
                    data_spec={"type": "list", "items": ["a", "b"]},
                    model_spec={},
                    inference_spec={},
                    dependencies=(),
                ),
                infer_id: RuntimeOp(
                    node_id=infer_id,
                    task_type="inference",
                    backend="dummy",
                    model="dummy-model",
                    data_spec={
                        "type": "graph_template",
                        "template": {
                            "name": "format",
                            "columns": [
                                {
                                    "label": "x",
                                    "node": seed_id,
                                    "path": "items.output",
                                }
                            ],
                            "options": {"format": {"template": "{x}"}},
                        },
                    },
                    model_spec={},
                    inference_spec={},
                    dependencies=(seed_id,),
                ),
            },
            node_order=[seed_id, infer_id],
            output_node_map={},
            dsl_to_runtime={},
        )

    merged, _ = optimizer.optimize_graphs({"g1": _graph("req1"), "g2": _graph("req2")})
    assert merged.node_count == 2
