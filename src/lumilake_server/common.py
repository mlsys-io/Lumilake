import json
from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar, TypedDict

RETRIEVAL_MODES = ("sql", "s3", "agent")


class InputRef(TypedDict):
    """The input key and ref path a rowwise column was projected from."""

    key: str
    path: str | None


def project_input_values(
    values: list[str], path: str | None, node_name: str
) -> list[Any]:
    """Project a field out of literal input values (JSON object strings) for a
    rowwise column, or keep the whole item when no field is read. The whole-item
    forms are ``item``, ``item.output`` and the documented ``item.json``; any
    other path walks the nested JSON keys in order."""
    normalized = (path or "").strip()
    if normalized in {"item", "item.output", "item.json"}:
        return list(values)
    if normalized.startswith("item.json."):
        keys = normalized[len("item.json.") :].split(".")
    elif normalized.startswith("item.output."):
        keys = normalized[len("item.output.") :].split(".")
    elif normalized.startswith("item."):
        keys = normalized[len("item.") :].split(".")
    else:
        keys = normalized.split(".")
    if not keys or keys == [""]:
        return list(values)
    projected: list[Any] = []
    for value in values:
        try:
            obj = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Input '{node_name}' value is not JSON at field path"
                f" '{path}': {value!r}"
            ) from exc
        for key in keys:
            if not isinstance(obj, dict) or key not in obj:
                raise ValueError(
                    f"Input '{node_name}' value missing field at path"
                    f" '{path}': {value!r}"
                )
            obj = obj[key]
        projected.append(obj)
    return projected


def retrieval_items_path(mode: str) -> str:
    """Return the items path that carries a retrieval's rows for ``mode``.

    An agent-mode retrieval replays a SQL plan, so its executor emits rows
    under ``table`` like SQL mode; s3 emits ``content``. Any other mode is
    rejected loudly rather than silently bound to a field the executor never
    emits.
    """
    if mode == "sql":
        return "items.table"
    if mode == "s3":
        return "items.content"
    if mode == "agent":
        return "items.table"
    raise ValueError(
        f"Unsupported retrieval mode {mode!r}; expected one of "
        f"{', '.join(RETRIEVAL_MODES)}"
    )


@dataclass(slots=True)
class Message:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class ApiConfig:
    """External OpenAI-compatible LLM API endpoint; ``url`` defaults to the
    serving endpoint ``lum.id/llm``, ``authorization`` supplies a caller
    credential for an untrusted origin, and ``model`` overrides the top-level
    ``config.model`` for the request."""

    url: str | None = None
    model: str | None = None
    authorization: str | None = None
    timeout_sec: float | None = None
    retries: int | None = None


@dataclass
class GenerationConfig:
    """LLM generation parameters; ``model`` is required in both local and API
    mode, so ``config.api`` routes the op externally without relaxing it."""

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
        if not self.model:
            raise ValueError(
                "GenerationConfig.model is required in both local and API mode."
            )

    def resolved_model(self) -> str:
        """The model name to submit. API mode prefers ``api.model``, then the
        top-level ``model``; local mode returns ``model`` directly. ``model``
        is required in both modes, so this never falls back to a default."""
        if self.api is not None:
            return self.api.model or self.model
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
