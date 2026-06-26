from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar


@dataclass(slots=True)
class Message:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class GenerationConfig:
    """LLM generation parameters. Add a typed field here and both the YAML
    parser allowlist and the runtime inference_spec pick it up automatically;
    use ``extra_sampling_params`` for vendor-specific keys not worth typing."""

    model: str
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
