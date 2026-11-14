from typing import Any

import pytest

from lumilake_server.common import ApiConfig, GenerationConfig, OutputEnforcement
from lumilake_server.graphs import Graph
from lumilake_server.ops import LLMVisionOp, OpMessage, as_output
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder


def test_generation_config_requires_model_without_api() -> None:
    """A locally-loaded backend has no model to fall back to: GenerationConfig
    rejects a missing/empty ``model`` when ``api`` is not set."""
    with pytest.raises(ValueError, match="model is required"):
        GenerationConfig()


def test_generation_config_requires_model_with_api() -> None:
    """API mode is a backend switch, not a model source: it must reject a
    missing ``model`` exactly like local mode, so the workflow spec reads the
    same either way."""
    with pytest.raises(ValueError, match="model is required"):
        GenerationConfig(api=ApiConfig())


def test_generation_config_resolved_model_prefers_api_model_over_top_level() -> None:
    cfg = GenerationConfig(model="top-level-model", api=ApiConfig(model="api-model"))

    assert cfg.resolved_model() == "api-model"


def test_generation_config_resolved_model_falls_back_to_top_level_model() -> None:
    cfg = GenerationConfig(model="top-level-model", api=ApiConfig())

    assert cfg.resolved_model() == "top-level-model"


def test_generation_config_resolved_model_returns_local_model_without_api() -> None:
    cfg = GenerationConfig(model="local-model")

    assert cfg.resolved_model() == "local-model"


def test_generation_config_rejects_non_dict_api() -> None:
    """A YAML ``api: "x"`` must fail validation where the config is
    constructed, not crash later as an AttributeError when the runtime graph
    calls ``api_config.url``. GenerationConfig rejects any non-dict,
    non-``ApiConfig``, non-``None`` value for ``api``."""
    with pytest.raises(ValueError, match="api must be a mapping or ApiConfig"):
        GenerationConfig(model="local-model", api="x")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"json_schema": {"type": "object"}, "text_regex": "x"},
        {"mode": "x", "text_regex": "x"},
        {"max_attempts": 0, "text_regex": "x"},
    ],
)
def test_output_enforcement_rejects_invalid_configuration(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="output_enforcement"):
        OutputEnforcement(**kwargs)


@pytest.mark.parametrize("num_beams", [0, True])
def test_generation_config_rejects_invalid_num_beams(num_beams: int) -> None:
    with pytest.raises(ValueError, match="num_beams"):
        GenerationConfig(model="local-model", num_beams=num_beams)


def test_output_enforcement_rejected_on_vision_op() -> None:
    vision = LLMVisionOp(
        [OpMessage(role="user", content="x")],
        image_source="images",
        config=GenerationConfig(
            model="m", output_enforcement=OutputEnforcement(text_regex="x")
        ),
    )
    with pytest.raises(ValueError, match="LLMVisionOp"):
        RuntimeGraphBuilder().build(Graph.from_ops([as_output("r", vision)]).compile())
