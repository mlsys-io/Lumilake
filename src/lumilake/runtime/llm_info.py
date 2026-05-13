from dataclasses import asdict, dataclass, field
from typing import Any, Self

import numpy as np


@dataclass(slots=True)
class LLMServiceInfo:
    cache_capacity: int = 0
    max_num_reqs: int = 1
    max_num_batched_tokens: int = 0
    alpha: float = 1
    is_memory_limited: bool = False
    prefix_caching_enabled: bool = False
    busy_threshold: int = 10

    accumulation_window: float = 0
    """Accumulation window in seconds.
    Allows short delay to accumulate more indices before dispatching, improving 
    batch size. Set to 0 to disable.
    """
    max_accumulation_time: float = 0
    """Maximum accumulation time before forcing a yield"""

    def __post_init__(self) -> None:
        self.is_memory_limited = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            cache_capacity=data.get("cache_capacity", 0),
            max_num_reqs=data.get("max_num_reqs", 1),
            max_num_batched_tokens=data.get("max_num_batched_tokens", 0),
            alpha=data.get("alpha", 1),
            is_memory_limited=data.get("is_memory_limited", False),
            prefix_caching_enabled=data.get("prefix_caching_enabled", False),
        )

    @property
    def token_budget(self) -> int:
        if self.is_memory_limited:
            return self.cache_capacity
        return self.max_num_batched_tokens


@dataclass
class LLMServiceConfig:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    info: LLMServiceInfo = field(default_factory=LLMServiceInfo)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        name = data["name"]
        args = data.get("args", {})
        info = LLMServiceInfo.from_dict(data.get("info", {}))
        return cls(name=name, args=args, info=info)


@dataclass(slots=True)
class UsageInfo:
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class LLMProfilingInfo:
    request_count: int
    prompt_tokens_avg: float
    prompt_tokens_std: float
    output_tokens_avg: float
    output_tokens_std: float
    total_tokens_avg: float
    total_tokens_std: float

    @classmethod
    def aggregate(cls, usage_infos: list[UsageInfo]) -> "LLMProfilingInfo":
        def stats(attr: str) -> tuple[float, float]:
            values = [
                getattr(info, attr)
                for info in usage_infos
                if getattr(info, attr) is not None
            ]
            if not values:
                return 0.0, 0.0
            return float(np.mean(values)), float(np.std(values))

        prompt_avg, prompt_std = stats("prompt_tokens")
        output_avg, output_std = stats("output_tokens")
        total_avg, total_std = stats("total_tokens")

        return cls(
            request_count=len(usage_infos),
            prompt_tokens_avg=prompt_avg,
            prompt_tokens_std=prompt_std,
            output_tokens_avg=output_avg,
            output_tokens_std=output_std,
            total_tokens_avg=total_avg,
            total_tokens_std=total_std,
        )

    @classmethod
    def merge(cls, profiling_info_list: list["LLMProfilingInfo"]) -> "LLMProfilingInfo":
        if not profiling_info_list:
            return cls.default()

        request_count = 0
        prompt_tokens_avg = 0.0
        prompt_tokens_std = 0.0
        output_tokens_avg = 0.0
        output_tokens_std = 0.0
        total_tokens_avg = 0.0
        total_tokens_std = 0.0

        for info in profiling_info_list:
            request_count += info.request_count
            prompt_tokens_avg += info.prompt_tokens_avg * info.request_count
            if prompt_tokens_std < info.prompt_tokens_std:
                prompt_tokens_std = info.prompt_tokens_std
            output_tokens_avg += info.output_tokens_avg * info.request_count
            if output_tokens_std < info.output_tokens_std:
                output_tokens_std = info.output_tokens_std
            total_tokens_avg += info.total_tokens_avg * info.request_count
            if total_tokens_std < info.total_tokens_std:
                total_tokens_std = info.total_tokens_std

        request_count = request_count
        prompt_tokens_avg /= request_count
        output_tokens_avg /= request_count
        total_tokens_avg /= request_count

        return cls(
            request_count=request_count,
            prompt_tokens_avg=prompt_tokens_avg,
            prompt_tokens_std=prompt_tokens_std,
            output_tokens_avg=output_tokens_avg,
            output_tokens_std=output_tokens_std,
            total_tokens_avg=total_tokens_avg,
            total_tokens_std=total_tokens_std,
        )

    @classmethod
    def default(cls) -> "LLMProfilingInfo":
        return cls(
            request_count=0,
            prompt_tokens_avg=0.0,
            prompt_tokens_std=0.0,
            output_tokens_avg=0.0,
            output_tokens_std=0.0,
            total_tokens_avg=0.0,
            total_tokens_std=0.0,
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, json_obj: dict[str, Any]) -> "LLMProfilingInfo":
        return cls(**json_obj)
