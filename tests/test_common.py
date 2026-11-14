import pytest

from lumilake_server.common import ApiConfig, GenerationConfig


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
