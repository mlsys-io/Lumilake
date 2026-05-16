import pytest

from lumilake_server.runtime.optimizer.halo import HaloOptimizer
from lumilake_server.runtime.optimizer.multimodal_cost import (
    MultimodalCostCoefficients,
    NodeCostType,
    classify_multimodal_node,
    compute_gpu_exec_cost,
)
from lumilake_server.runtime.optimizer.schedule.models import Node


def test_multimodal_node_type_classification() -> None:
    text_node = Node(id="t", type="inference", engine="vllm", model="foo-7B", raw={})
    embed_node = Node(
        id="e", type="embedding", engine="vllm", model="foo-7B", raw={"type": "list"}
    )
    vlm_node = Node(
        id="v",
        type="inference",
        engine="vllm",
        model="foo-7B",
        raw={"image_embedding": {"node": "img_embed", "path": "embedding_file"}},
    )
    diff_node = Node(id="d", type="diffusion", engine="vllm", model="foo-7B", raw={})

    assert classify_multimodal_node(text_node) == NodeCostType.TEXT_INFERENCE
    assert classify_multimodal_node(embed_node) == NodeCostType.VISION_EMBEDDING
    assert classify_multimodal_node(vlm_node) == NodeCostType.VISION_INFERENCE
    assert classify_multimodal_node(diff_node) == NodeCostType.IMAGE_GENERATION


def test_multimodal_exec_cost_formulas() -> None:
    coeffs = MultimodalCostCoefficients()
    s = 8.0
    q = 4

    text = compute_gpu_exec_cost(
        node=Node(id="t", type="inference", engine="vllm", model="foo-8B", raw={}),
        model_size_b=s,
        input_query_count=q,
        coeffs=coeffs,
    )
    embed = compute_gpu_exec_cost(
        node=Node(id="e", type="embedding", engine="vllm", model="foo-8B", raw={}),
        model_size_b=s,
        input_query_count=q,
        coeffs=coeffs,
    )
    vlm = compute_gpu_exec_cost(
        node=Node(
            id="v",
            type="inference",
            engine="vllm",
            model="foo-8B",
            raw={"image_embedding": {"node": "n1"}},
        ),
        model_size_b=s,
        input_query_count=q,
        coeffs=coeffs,
    )
    diff = compute_gpu_exec_cost(
        node=Node(
            id="d",
            type="diffusion",
            engine="vllm",
            model="foo-8B",
            raw={
                "_inference_spec": {
                    "num_inference_steps": 60,
                    "height": 768,
                    "width": 768,
                }
            },
        ),
        model_size_b=s,
        input_query_count=q,
        coeffs=coeffs,
    )

    assert text == pytest.approx(0.25 * s + 0.15 * q)
    assert embed == pytest.approx(0.10 * s + 0.08 * q)
    assert vlm == pytest.approx(0.30 * s + 0.25 * q)
    assert diff == pytest.approx((0.20 * s + 0.05 * q) * (60.0 / 30.0))


def test_diffusion_cost_scales_with_resolution() -> None:
    coeffs = MultimodalCostCoefficients()
    base = compute_gpu_exec_cost(
        node=Node(
            id="d1",
            type="diffusion",
            engine="vllm",
            model="foo-8B",
            raw={
                "_inference_spec": {
                    "num_inference_steps": 30,
                    "height": 768,
                    "width": 768,
                }
            },
        ),
        model_size_b=8.0,
        input_query_count=1,
        coeffs=coeffs,
    )
    high_res = compute_gpu_exec_cost(
        node=Node(
            id="d2",
            type="diffusion",
            engine="vllm",
            model="foo-8B",
            raw={
                "_inference_spec": {
                    "num_inference_steps": 30,
                    "height": 1536,
                    "width": 1536,
                }
            },
        ),
        model_size_b=8.0,
        input_query_count=1,
        coeffs=coeffs,
    )
    assert high_res == pytest.approx(base * 4.0)


def test_model_init_cost_pins_switch_penalty() -> None:
    optimizer = HaloOptimizer()
    init_sec_per_b = optimizer._model_init_sec_per_b

    model_a = "google/gemma-3-27b-it"
    model_b = "llava-hf/llava-1.5-7b"
    size_a = 27.0
    size_b = 7.0

    def node(node_id: str, model: str) -> Node:
        return Node(id=node_id, type="inference", engine="vllm", model=model, raw={})

    a1, a2, a3, a4 = (node(f"a{i}", model_a) for i in range(4))
    b1, b2 = node("b1", model_b), node("b2", model_b)

    seq_ab = [a1, a2, b1, b2, a3, a4]
    total_ab = 0.0
    last: str | None = None
    for n in seq_ab:
        total_ab += optimizer._model_init_cost(n, last)
        last = n.model
    expected_ab = init_sec_per_b * (size_a + size_b + size_a)

    seq_b_then_a = [a1, b1, a2, a3, a4, b2]
    total_b_then_a = 0.0
    last = None
    for n in seq_b_then_a:
        total_b_then_a += optimizer._model_init_cost(n, last)
        last = n.model
    expected_b_then_a = init_sec_per_b * (size_a + size_b + size_a + size_b)

    assert total_ab == pytest.approx(expected_ab)
    assert total_b_then_a == pytest.approx(expected_b_then_a)
    assert total_ab < total_b_then_a


def test_invalid_diffusion_config_raises() -> None:
    coeffs = MultimodalCostCoefficients()
    with pytest.raises(ValueError):
        compute_gpu_exec_cost(
            node=Node(
                id="d",
                type="diffusion",
                engine="vllm",
                model="foo-8B",
                raw={"_inference_spec": {"num_inference_steps": 0}},
            ),
            model_size_b=8.0,
            input_query_count=1,
            coeffs=coeffs,
        )
