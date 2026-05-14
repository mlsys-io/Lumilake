import pytest

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
