from lumilake_server.runtime.optimizer.topological_sort import TopologicalSortOptimizer
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _op(node_id: str, backend: str, task_type: str = "inference") -> RuntimeOp:
    return RuntimeOp(
        node_id=node_id,
        task_type=task_type,
        backend=backend,
        model="m",
        data_spec={"type": "list", "items": []},
        model_spec={},
        inference_spec={},
        dependencies=(),
    )


def test_topological_sort_assigns_omni_node_to_gpu_worker() -> None:
    omni_id = "image_gen"
    cpu_id = "post_process"
    graph = RuntimeGraph(
        nodes={
            omni_id: _op(omni_id, backend="omni", task_type="omni_text2image"),
            cpu_id: _op(cpu_id, backend="text", task_type="format"),
        },
        node_order=[omni_id, cpu_id],
        output_node_map={},
        dsl_to_runtime={},
    )

    schedule = TopologicalSortOptimizer().generate_schedule(
        graph=graph,
        worker_names=["gpu-0", "cpu-0"],
        worker_profiles={"gpu-0": {"has_gpu": True}, "cpu-0": {"has_gpu": False}},
    )

    assert omni_id in schedule.worker_assignment["gpu-0"]
    assert omni_id not in schedule.worker_assignment["cpu-0"]
    assert cpu_id in schedule.worker_assignment["cpu-0"]


def test_topological_sort_assigns_vllm_and_diffusers_to_gpu() -> None:
    nodes = {
        "vllm_node": _op("vllm_node", backend="vllm"),
        "transformers_node": _op("transformers_node", backend="transformers"),
        "diffusers_node": _op("diffusers_node", backend="diffusers"),
        "cpu_node": _op("cpu_node", backend="text"),
    }
    graph = RuntimeGraph(
        nodes=nodes,
        node_order=list(nodes),
        output_node_map={},
        dsl_to_runtime={},
    )

    schedule = TopologicalSortOptimizer().generate_schedule(
        graph=graph,
        worker_names=["gpu-0", "cpu-0"],
        worker_profiles={"gpu-0": {"has_gpu": True}, "cpu-0": {"has_gpu": False}},
    )

    gpu_assigned = set(schedule.worker_assignment["gpu-0"])
    assert gpu_assigned == {"vllm_node", "transformers_node", "diffusers_node"}
    assert schedule.worker_assignment["cpu-0"] == ["cpu_node"]
