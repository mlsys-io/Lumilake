from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar


@dataclass(slots=True)
class Message:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


DEFAULT_API_MODEL = "deepseek-v4-flash"


@dataclass
class ApiConfig:
    """External OpenAI-compatible LLM API endpoint. When ``url`` is absent it
    defaults to the serving endpoint ``lum.id/llm``. ``authorization`` lets the
    caller supply their own credential for an untrusted origin; for a
    trusted origin (see ``LUMILAKE_API_TRUSTED_ORIGINS``) the server attaches
    its own ``LUMILAKE_RUNTIME_TOKEN`` PAT instead. That header becomes part
    of the FlowMesh task spec submitted for execution and is redacted only
    before archival and logging, never before dispatch. When ``model`` is
    also absent, :meth:`GenerationConfig.resolved_model` falls back to the
    top-level ``config.model`` and then to :data:`DEFAULT_API_MODEL`."""

    url: str | None = None
    model: str | None = None
    authorization: str | None = None
    timeout_sec: float | None = None


@dataclass
class GenerationConfig:
    """LLM generation parameters. Add a typed field here and both the YAML
    parser allowlist and the runtime inference_spec pick it up automatically;
    use ``extra_sampling_params`` for vendor-specific keys not worth typing.

    ``model`` is required unless ``api`` is set: a locally-loaded backend has
    no default to fall back to, but API mode resolves one via
    :meth:`resolved_model`."""

    model: str = ""
    api: ApiConfig | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, int] | None = None
    logprobs: int | None = None
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    stream_options: Any = None
    temperature: float | None = None
    top_p: float | None = None
    ignore_eos: bool | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    min_tokens: int | None = None
    repetition_penalty: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    extra_sampling_params: dict[str, Any] | None = None

    # Engine-level
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    tensor_parallel_size: int | None = None
    dtype: str | None = None
    extra_engine_kwargs: dict[str, Any] | None = None

    # Stripped by openai_kwargs() — OpenAI Chat API rejects these.
    _NON_OPENAI_FIELDS: ClassVar[tuple[str, ...]] = (
        "api",
        "ignore_eos",
        "chat_template_kwargs",
        "repetition_penalty",
        "top_k",
        "min_p",
        "min_tokens",
        "extra_sampling_params",
        "max_model_len",
        "gpu_memory_utilization",
        "tensor_parallel_size",
        "dtype",
        "extra_engine_kwargs",
    )
    # Skipped by inference_spec() — not per-request sampler args.
    _NON_SAMPLER_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "model",
            "api",
            "stream",
            "stream_options",
            "extra_sampling_params",
            "max_model_len",
            "gpu_memory_utilization",
            "tensor_parallel_size",
            "dtype",
            "extra_engine_kwargs",
        }
    )
    # Engine-level typed fields the runtime overlays onto the backend config.
    _ENGINE_OVERLAY_FIELDS: ClassVar[tuple[str, ...]] = (
        "max_model_len",
        "gpu_memory_utilization",
        "tensor_parallel_size",
        "dtype",
    )

    def __post_init__(self) -> None:
        if isinstance(self.api, dict):
            self.api = ApiConfig(**self.api)
        elif self.api is not None and not isinstance(self.api, ApiConfig):
            raise ValueError(
                "GenerationConfig.api must be a mapping or ApiConfig, got "
                f"{type(self.api).__name__}."
            )
        if self.api is None and not self.model:
            raise ValueError(
                "GenerationConfig.model is required when config.api is not set."
            )

    def resolved_model(self) -> str:
        """The model name to submit. Local mode always has a required
        ``model``; API mode prefers ``api.model``, then the top-level
        ``model``, then :data:`DEFAULT_API_MODEL`."""
        if self.api is not None:
            return self.api.model or self.model or DEFAULT_API_MODEL
        return self.model

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def openai_kwargs(self) -> dict[str, Any]:
        kwargs = self.to_dict()
        for field_name in self._NON_OPENAI_FIELDS:
            kwargs.pop(field_name, None)
        return kwargs

    def inference_spec(self) -> dict[str, Any]:
        """Non-None typed samplers + extra_sampling_params. Raises on conflict."""
        spec: dict[str, Any] = {}
        for f in fields(self):
            if f.name in self._NON_SAMPLER_FIELDS:
                continue
            value = getattr(self, f.name)
            if value is not None:
                spec[f.name] = value
        extras = self.extra_sampling_params or {}
        conflicts = sorted(set(extras) & set(spec))
        if conflicts:
            raise ValueError(
                f"extra_sampling_params conflict with typed fields: {conflicts}"
            )
        spec.update(extras)
        return spec

    def engine_overlay(self) -> dict[str, Any]:
        """Engine-level typed fields + extra_engine_kwargs. Raises on conflict."""
        overlay: dict[str, Any] = {
            name: getattr(self, name)
            for name in self._ENGINE_OVERLAY_FIELDS
            if getattr(self, name) is not None
        }
        extras = self.extra_engine_kwargs or {}
        conflicts = sorted(set(extras) & set(overlay))
        if conflicts:
            raise ValueError(
                f"extra_engine_kwargs conflict with typed fields: {conflicts}"
            )
        overlay.update(extras)
        return overlay

    @classmethod
    def from_env(cls, **kwargs: Any) -> "GenerationConfig":
        if "model" not in kwargs:
            raise ValueError("GenerationConfig.from_env requires 'model'.")
        return cls(**kwargs)
