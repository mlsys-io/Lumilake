from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp


def build_dummy_runtime_graph(
    name: str, *, output_name: str = "output"
) -> RuntimeGraph:
    node_id = f"{name}_node"
    op = RuntimeOp(
        node_id=node_id,
        task_type="inference",
        backend="dummy",
        model="dummy-model",
        data_spec={},
        model_spec={},
        inference_spec={},
    )
    return RuntimeGraph(
        nodes={node_id: op},
        node_order=[node_id],
        output_node_map={node_id: output_name},
    )
