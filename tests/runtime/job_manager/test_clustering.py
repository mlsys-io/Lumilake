from lumilake.runtime.job_manager.cluster_algo.clustering import (
    select_affinity_batch_ids,
)
from lumilake.runtime.runtime_graph import RuntimeGraph
from lumilake.runtime.runtime_ops import RuntimeOp


def _graph(workflow_id: str, *, model: str, system_prompt: str) -> RuntimeGraph:
    node_id = f"{workflow_id}_node"
    return RuntimeGraph(
        nodes={
            node_id: RuntimeOp(
                node_id=node_id,
                task_type="inference",
                backend="vllm",
                model=model,
                data_spec={
                    "messages": [{"role": "system", "content": system_prompt}],
                },
                model_spec={},
                inference_spec={},
            )
        },
        node_order=[node_id],
        output_node_map={node_id: "output"},
    )


def test_select_affinity_batch_ids_includes_pinned_ids() -> None:
    graphs = {
        "pin": _graph("pin", model="model-a", system_prompt="market risk"),
        "near": _graph("near", model="model-a", system_prompt="market risk"),
        "far": _graph("far", model="model-b", system_prompt="sports highlights"),
    }
    enqueued_at = {"pin": 3.0, "near": 2.0, "far": 1.0}

    selected = select_affinity_batch_ids(
        graphs,
        enqueued_at,
        2,
        pinned_ids=["pin"],
    )

    assert "pin" in selected
    assert len(selected) == 2


def test_select_affinity_batch_ids_clusters_around_pinned_ids() -> None:
    graphs = {
        "pin": _graph("pin", model="model-a", system_prompt="market risk"),
        "near": _graph("near", model="model-a", system_prompt="market risk"),
        "far": _graph("far", model="model-b", system_prompt="sports highlights"),
    }
    enqueued_at = {"pin": 1.0, "near": 2.0, "far": 3.0}

    selected = select_affinity_batch_ids(
        graphs,
        enqueued_at,
        2,
        pinned_ids=["pin"],
    )

    assert selected == ["pin", "near"]
