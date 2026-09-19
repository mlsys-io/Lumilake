"""Unit tests for per-job planner engine options on a dynamic spec.

A dynamic spec may size the planner's vLLM engine per job via
``driver.max_model_len`` and ``driver.gpu_memory_utilization``. When unset the
proposer op's config must be byte-identical to today (the fields stay ``None``
and the server-wide env default still applies); when set they land on the
proposer ``LLMChatOp``'s ``config``.
"""

import pytest
from pydantic import ValidationError

from lumilake_server.dynamic.blocks import PROPOSER_NODE_ID, fused_round_graph
from lumilake_server.dynamic.driver import build_round
from lumilake_server.dynamic.spec import DriverSettings


def _proposer_config(
    *,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    dtype: str | None = None,
    extra_engine_kwargs: dict[str, object] | None = None,
) -> dict[str, object]:
    driver = DriverSettings(
        model="Qwen/Qwen3-8B",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        extra_engine_kwargs=extra_engine_kwargs,
    )
    round_build = build_round(
        [],
        node_registry={},
        round_index=0,
        goal="analyze market data",
        observations=[],
        topology=[],
        preview_width=driver.preview_width,
        model=driver.model,
        max_tokens=driver.max_tokens,
        temperature=driver.temperature,
        threshold=driver.threshold,
        library=None,
        chat_template_kwargs=driver.chat_template_kwargs,
        max_model_len=driver.max_model_len,
        gpu_memory_utilization=driver.gpu_memory_utilization,
        dtype=driver.dtype,
        extra_engine_kwargs=driver.extra_engine_kwargs,
    )
    return round_build.graph[PROPOSER_NODE_ID]["config"]


def test_no_engine_opts_matches_default_proposer() -> None:
    """A spec with no engine options produces the same proposer config as today."""
    default = fused_round_graph(
        {},
        leaf_ids=[],
        proposer_system="system",
        proposer_user="user",
        lambda_code="",
    )[PROPOSER_NODE_ID]["config"]
    config = _proposer_config()
    assert config == default
    assert config["max_model_len"] is None
    assert config["gpu_memory_utilization"] is None
    assert config["dtype"] is None
    assert config["extra_engine_kwargs"] is None


def test_max_model_len_lands_on_proposer() -> None:
    config = _proposer_config(max_model_len=4096)
    assert config["max_model_len"] == 4096
    assert config["gpu_memory_utilization"] is None


def test_gpu_memory_utilization_lands_on_proposer() -> None:
    config = _proposer_config(gpu_memory_utilization=0.9)
    assert config["gpu_memory_utilization"] == 0.9
    assert config["max_model_len"] is None


@pytest.mark.parametrize("bad", [0, -1, -4096])
def test_max_model_len_must_be_positive(bad: int) -> None:
    with pytest.raises(ValidationError):
        DriverSettings(model="Qwen/Qwen3-8B", max_model_len=bad)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_gpu_memory_utilization_must_be_in_open_unit_interval(bad: float) -> None:
    with pytest.raises(ValidationError):
        DriverSettings(model="Qwen/Qwen3-8B", gpu_memory_utilization=bad)


def test_gpu_memory_utilization_upper_bound_inclusive() -> None:
    driver = DriverSettings(model="Qwen/Qwen3-8B", gpu_memory_utilization=1.0)
    assert driver.gpu_memory_utilization == 1.0


def test_dtype_lands_on_proposer() -> None:
    config = _proposer_config(dtype="fp8")
    assert config["dtype"] == "fp8"
    assert config["extra_engine_kwargs"] is None


def test_extra_engine_kwargs_lands_on_proposer() -> None:
    config = _proposer_config(extra_engine_kwargs={"quantization": "fp8"})
    assert config["extra_engine_kwargs"] == {"quantization": "fp8"}
    assert config["dtype"] is None


@pytest.mark.parametrize("bad", ["", "   ", "\t"])
def test_dtype_must_not_be_blank(bad: str) -> None:
    with pytest.raises(ValidationError):
        DriverSettings(model="Qwen/Qwen3-8B", dtype=bad)
