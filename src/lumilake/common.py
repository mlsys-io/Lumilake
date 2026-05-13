from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class Message:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class GenerationConfig:
    model: str
    frequency_penalty: float | None = None
    logit_bias: dict[str, int] | None = None
    logprobs: int | None = None
    max_tokens: int | None = None
    n: int | None = 1
    presence_penalty: float | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = False
    stream_options: Any = None
    temperature: float | None = None
    top_p: float | None = None
    ignore_eos: bool = False
    # Forwarded to tokenizer.apply_chat_template (e.g. enable_thinking=False for Qwen3).
    chat_template_kwargs: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def openai_kwargs(self) -> dict[str, Any]:
        kwargs = self.to_dict()
        kwargs.pop("ignore_eos")
        kwargs.pop("chat_template_kwargs")
        return kwargs

    @classmethod
    def from_env(cls, **kwargs) -> "GenerationConfig":
        # Construct a GenerationConfig from kwargs only. We intentionally do
        # not read LLM service credentials or defaults from environment here;
        # callers must pass explicit parameters (model is required).
        if "model" not in kwargs:
            raise ValueError(
                "GenerationConfig.from_env requires 'model' to be provided explicitly."
            )
        return cls(**kwargs)
