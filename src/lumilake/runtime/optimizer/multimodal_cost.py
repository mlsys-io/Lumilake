from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .schedule.models import Node


class NodeCostType(StrEnum):
    TEXT_INFERENCE = "text_inference"
    VISION_EMBEDDING = "vision_embedding"
    VISION_INFERENCE = "vision_inference"
    IMAGE_GENERATION = "image_generation"


class MultimodalCostCoefficients(BaseModel):
    alpha_text: float = Field(default=0.25, ge=0)
    beta_text: float = Field(default=0.15, ge=0)
    alpha_embed: float = Field(default=0.10, ge=0)
    beta_embed: float = Field(default=0.08, ge=0)
    alpha_vlm: float = Field(default=0.30, ge=0)
    beta_vlm: float = Field(default=0.25, ge=0)
    alpha_diff: float = Field(default=0.20, ge=0)
    beta_diff: float = Field(default=0.05, ge=0)


class VisionMarker(BaseModel):
    model_config = ConfigDict(extra="ignore")
    image_embedding: dict[str, Any] | None = None


class DiffusionInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    num_inference_steps: float = Field(default=30.0, gt=0)
    height: float = Field(default=768.0, gt=0)
    width: float = Field(default=768.0, gt=0)


def classify_multimodal_node(node: Node) -> NodeCostType:
    node_type = (node.type or "").strip().lower()
    if node_type == "diffusion":
        return NodeCostType.IMAGE_GENERATION
    if node_type == "embedding":
        return NodeCostType.VISION_EMBEDDING
    marker = VisionMarker.model_validate(node.raw)
    if node_type == "inference" and marker.image_embedding is not None:
        return NodeCostType.VISION_INFERENCE
    return NodeCostType.TEXT_INFERENCE


def compute_gpu_exec_cost(
    *,
    node: Node,
    model_size_b: float,
    input_query_count: int,
    coeffs: MultimodalCostCoefficients,
) -> float:
    node_type = classify_multimodal_node(node)
    query_count = max(1, int(input_query_count))
    model_size = max(0.0, float(model_size_b))
    if node_type == NodeCostType.TEXT_INFERENCE:
        return coeffs.alpha_text * model_size + coeffs.beta_text * query_count
    if node_type == NodeCostType.VISION_EMBEDDING:
        return coeffs.alpha_embed * model_size + coeffs.beta_embed * query_count
    if node_type == NodeCostType.VISION_INFERENCE:
        return coeffs.alpha_vlm * model_size + coeffs.beta_vlm * query_count

    config = _parse_diffusion_inference_config(node)
    step_scale = config.num_inference_steps / 30.0
    pixel_scale = (config.height * config.width) / (768.0 * 768.0)
    base = coeffs.alpha_diff * model_size + coeffs.beta_diff * query_count
    return base * step_scale * pixel_scale


def _parse_diffusion_inference_config(node: Node) -> DiffusionInferenceConfig:
    raw = node.raw
    if "_inference_spec" not in raw:
        return DiffusionInferenceConfig()
    payload = raw["_inference_spec"]
    if isinstance(payload, dict):
        return DiffusionInferenceConfig.model_validate(payload)
    return DiffusionInferenceConfig()


__all__ = [
    "DiffusionInferenceConfig",
    "MultimodalCostCoefficients",
    "NodeCostType",
    "classify_multimodal_node",
    "compute_gpu_exec_cost",
]
