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


def test_dedupe_remaps_placeholder_refs_in_api_spec() -> None:
    """When dedupe merges an API predecessor, ``${node.path}`` references to it
    inside a downstream ``api_spec`` body must be remapped to the canonical node,
    or the placeholder would name a node that no longer exists."""
    api0 = RuntimeOp(
        node_id="Api0",
        task_type="api",
        backend="api",
        model="model-a",
        data_spec={},
        model_spec={},
        inference_spec={},
        api_spec={"json": {"messages": [{"role": "user", "content": "hello"}]}},
        dependencies=(),
    )
    api1 = RuntimeOp(
        node_id="Api1",
        task_type="api",
        backend="api",
        model="model-a",
        data_spec={},
        model_spec={},
        inference_spec={},
        api_spec={"json": {"messages": [{"role": "user", "content": "hello"}]}},
        dependencies=(),
    )
    downstream = RuntimeOp(
        node_id="Down",
        task_type="api",
        backend="api",
        model="model-a",
        data_spec={},
        model_spec={},
        inference_spec={},
        api_spec={
            "json": {
                "messages": [{"role": "user", "content": "${Api1.items.0.output}"}]
            }
        },
        dependencies=("Api1",),
    )
    graph = RuntimeGraph(
        nodes={"Api0": api0, "Api1": api1, "Down": downstream},
        node_order=["Api0", "Api1", "Down"],
        output_node_map={"Down": "result"},
        dsl_to_runtime={},
    )

    optimized, _ = HaloOptimizer().optimize_graphs({"wf": graph})

    assert optimized.node_count == 2
    down = optimized.nodes["Down"]
    assert down.dependencies == ("Api0",)
    assert down.api_spec["json"]["messages"] == [
        {"role": "user", "content": "${Api0.items.0.output}"}
    ]


def test_dedupe_keys_on_condition_and_preserves_it() -> None:
    """Two ops identical except for ``condition`` must not be merged, and a
    surviving op must keep its ``condition`` through the rebuild."""

    def _op(node_id: str, expr: str) -> RuntimeOp:
        return RuntimeOp(
            node_id=node_id,
            task_type="inference",
            backend="vllm",
            model="model-a",
            data_spec={},
            model_spec={},
            inference_spec={},
            dependencies=(),
            condition={"node": "gate", "expr": expr},
        )

    graph = RuntimeGraph(
        nodes={
            "A": _op("A", "gate == 'on'"),
            "B": _op("B", "gate == 'off'"),
        },
        node_order=["A", "B"],
        output_node_map={},
        dsl_to_runtime={},
    )

    optimized, _ = HaloOptimizer().optimize_graphs({"wf": graph})

    assert optimized.node_count == 2
    assert optimized.nodes["A"].condition == {"node": "gate", "expr": "gate == 'on'"}
    assert optimized.nodes["B"].condition == {"node": "gate", "expr": "gate == 'off'"}


def test_dedupe_merges_ops_with_identical_condition() -> None:
    """Two ops with the same ``condition`` are still dedupable, and the merged
    op keeps the condition."""

    def _op(node_id: str) -> RuntimeOp:
        return RuntimeOp(
            node_id=node_id,
            task_type="inference",
            backend="vllm",
            model="model-a",
            data_spec={},
            model_spec={},
            inference_spec={},
            dependencies=(),
            condition={"node": "gate", "expr": "gate == 'on'"},
        )

    graph = RuntimeGraph(
        nodes={"C": _op("C"), "D": _op("D")},
        node_order=["C", "D"],
        output_node_map={},
        dsl_to_runtime={},
    )

    optimized, _ = HaloOptimizer().optimize_graphs({"wf": graph})

    assert optimized.node_count == 1
    (survivor,) = optimized.nodes.values()
    assert survivor.condition == {"node": "gate", "expr": "gate == 'on'"}


def test_dedupe_remaps_condition_node_reference() -> None:
    """A surviving op's ``condition.node`` must be remapped when it references
    a predecessor that dedupe merged away, or the gate would name a node that
    no longer exists."""

    def _op(node_id: str) -> RuntimeOp:
        return RuntimeOp(
            node_id=node_id,
            task_type="inference",
            backend="vllm",
            model="model-a",
            data_spec={},
            model_spec={},
            inference_spec={},
            dependencies=(),
        )

    a = _op("A")
    b = _op("B")
    c = RuntimeOp(
        node_id="C",
        task_type="inference",
        backend="vllm",
        model="model-a",
        data_spec={},
        model_spec={},
        inference_spec={},
        dependencies=("B",),
        condition={"node": "B", "expr": "B == 'ok'"},
    )
    graph = RuntimeGraph(
        nodes={"A": a, "B": b, "C": c},
        node_order=["A", "B", "C"],
        output_node_map={"C": "result"},
        dsl_to_runtime={},
    )

    optimized, _ = HaloOptimizer().optimize_graphs({"wf": graph})

    assert optimized.node_count == 2
    assert optimized.nodes["C"].dependencies == ("A",)
    assert optimized.nodes["C"].condition == {"node": "A", "expr": "B == 'ok'"}
