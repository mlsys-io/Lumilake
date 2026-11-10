import pytest

from lumilake_server.common import ApiConfig, GenerationConfig, OutputEnforcement


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


def test_generation_config_coerces_output_enforcement_mapping() -> None:
    cfg = GenerationConfig(
        model="local-model",
        output_enforcement={  # type: ignore[arg-type]
            "mode": "retry",
            "max_attempts": 3,
            "text_regex": "^ok$",
        },
    )
    assert cfg.output_enforcement == OutputEnforcement(
        mode="retry", max_attempts=3, text_regex="^ok$"
    )
    assert "output_enforcement" not in cfg.inference_spec()
    assert "output_enforcement" not in cfg.openai_kwargs()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0, "text_regex": "ok"},
        {"max_attempts": 11, "text_regex": "ok"},
        {"text_regex": "ok", "json_schema": {"type": "object"}},
        {},
        {"text_regex": "("},
        {"json_schema": {"type": "not-a-type"}},
    ],
)
def test_output_enforcement_rejects_invalid_configuration(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="output_enforcement"):
        OutputEnforcement(**kwargs)  # type: ignore[arg-type]


def test_output_enforcement_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="output_enforcement.mode"):
        OutputEnforcement(mode="other", text_regex="ok")


def test_api_output_enforcement_rejects_constrained_mode() -> None:
    with pytest.raises(ValueError, match="only for local vLLM inference"):
        GenerationConfig(
            model="api-model",
            api=ApiConfig(),
            output_enforcement=OutputEnforcement(mode="constrained", text_regex="ok"),
        )
