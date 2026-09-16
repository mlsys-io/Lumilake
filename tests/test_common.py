import pytest

from lumilake_server.common import DEFAULT_API_MODEL, ApiConfig, GenerationConfig


def test_generation_config_requires_model_without_api() -> None:
    """A locally-loaded backend has no model to fall back to: GenerationConfig
    rejects a missing/empty ``model`` when ``api`` is not set."""
    with pytest.raises(ValueError, match="model is required"):
        GenerationConfig()


def test_generation_config_resolved_model_defaults_when_api_mode_omits_model() -> None:
    """API mode has a typed default to fall back to when neither
    ``config.api.model`` nor the top-level ``config.model`` is set."""
    cfg = GenerationConfig(api=ApiConfig())

    assert cfg.resolved_model() == DEFAULT_API_MODEL


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
