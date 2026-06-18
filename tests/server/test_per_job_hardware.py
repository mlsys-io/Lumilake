"""Tests for per-job hardware overrides.

Covers:
- ``HardwareRequirements`` pydantic validation (forbid-extra, positive cpu,
  non-negative gpu, non-empty memory strings).
- The flowmesh resolver helpers fall back to ``HARDWARE_*`` env defaults
  when fields are unset and respect overrides field-by-field.
- The priority-queue partition key includes the hardware signature so jobs
  with different hardware tuples land in distinct dispatches; jobs with no
  override co-batch.
- ``_any_graph_requires_gpu`` correctly classifies vLLM ops.
"""

from typing import Any

import pytest
from lumilake import envs
from pydantic import ValidationError

from lumilake_server.common import GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    DataRetrievalOp,
    LLMChatOp,
    OpMessage,
    as_output,
    input_placeholder,
)
from lumilake_server.routes.jobs import _any_graph_requires_gpu
from lumilake_server.runtime.job_manager.priority_queue import (
    _hardware_signature,
)
from lumilake_server.runtime.protocol import HardwareRequirements
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.runtime.runtime_manager.flowmesh import (
    _resolve_cpu,
    _resolve_gpu_count,
    _resolve_gpu_memory,
    _resolve_memory,
)

# ---------------------------------------------------------------------------
# HardwareRequirements validation
# ---------------------------------------------------------------------------


def test_hardware_requirements_accepts_empty_for_full_env_fallback() -> None:
    hw = HardwareRequirements()
    assert hw.cpu is None
    assert hw.memory is None
    assert hw.gpu is None
    assert hw.gpu_memory is None


def test_hardware_requirements_rejects_unknown_field() -> None:
    """``extra="forbid"`` keeps typos from silently becoming no-ops."""
    with pytest.raises(ValidationError):
        HardwareRequirements.model_validate({"cpu": 1, "ram": "16Gi"})


def test_hardware_requirements_cpu_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        HardwareRequirements(cpu=0)
    with pytest.raises(ValidationError):
        HardwareRequirements(cpu=-1)


def test_hardware_requirements_cpu_rejects_above_cap() -> None:
    """Per-job CPU count is capped to prevent cross-tenant DoS / quota bypass."""
    assert HardwareRequirements(cpu=1024).cpu == 1024
    with pytest.raises(ValidationError):
        HardwareRequirements(cpu=1025)
    with pytest.raises(ValidationError):
        HardwareRequirements(cpu=1_000_000)
    with pytest.raises(ValidationError):
        HardwareRequirements(cpu=2_147_483_647)


def test_hardware_requirements_gpu_accepts_zero() -> None:
    """``gpu=0`` is meaningful — it's the explicit "no-GPU" request that the
    server pairs with the workflow-vs-gpu guard."""
    hw = HardwareRequirements(gpu=0)
    assert hw.gpu == 0


def test_hardware_requirements_gpu_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        HardwareRequirements(gpu=-1)


def test_hardware_requirements_gpu_rejects_above_cap() -> None:
    """Per-job GPU count is capped to prevent cross-tenant DoS / quota bypass."""
    assert HardwareRequirements(gpu=8).gpu == 8
    with pytest.raises(ValidationError):
        HardwareRequirements(gpu=9)
    with pytest.raises(ValidationError):
        HardwareRequirements(gpu=999)
    with pytest.raises(ValidationError):
        HardwareRequirements(gpu=1_000_000)


def test_hardware_requirements_memory_rejects_empty_string() -> None:
    with pytest.raises(ValidationError):
        HardwareRequirements(memory="")


@pytest.mark.parametrize(
    "bad",
    [
        "abc",
        "-1Gi",
        "0",
        "0Gi",
        "16gI",
        "16 Gi",
        "\x00",
        "16Gi\n[CRIT] root pwned",
        "16Gb",
        "16",
        "1" * 17 + "Gi",
        "9999999999999Ti",
        "10000Gi",
    ],
)
def test_hardware_requirements_memory_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValidationError):
        HardwareRequirements(memory=bad)
    with pytest.raises(ValidationError):
        HardwareRequirements(gpu_memory=bad)


@pytest.mark.parametrize("good", ["1Ki", "16Gi", "128Mi", "2Ti", "999Gi"])
def test_hardware_requirements_memory_accepts_well_formed(good: str) -> None:
    assert HardwareRequirements(memory=good).memory == good
    assert HardwareRequirements(gpu_memory=good).gpu_memory == good


# ---------------------------------------------------------------------------
# Flowmesh resolver fallback semantics
# ---------------------------------------------------------------------------


def test_resolve_cpu_uses_override_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.envs.HARDWARE_CPU_REQUIREMENT",
        8,
    )
    assert _resolve_cpu(HardwareRequirements(cpu=16)) == 16


def test_resolve_cpu_falls_back_to_env_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.envs.HARDWARE_CPU_REQUIREMENT",
        8,
    )
    assert _resolve_cpu(None) == 8
    assert _resolve_cpu(HardwareRequirements(memory="32Gi")) == 8


def test_resolve_memory_uses_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.envs.HARDWARE_MEMORY_REQUIREMENT",
        "16Gi",
    )
    assert _resolve_memory(HardwareRequirements(memory="64Gi")) == "64Gi"
    assert _resolve_memory(None) == "16Gi"


def test_resolve_gpu_count_uses_override_including_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.envs.HARDWARE_GPU_REQUIREMENT",
        1,
    )
    # gpu=0 must beat the env default — that's the "explicit no-GPU" semantic.
    assert _resolve_gpu_count(HardwareRequirements(gpu=0)) == 0
    assert _resolve_gpu_count(HardwareRequirements(gpu=4)) == 4
    assert _resolve_gpu_count(None) == 1


def test_resolve_gpu_memory_uses_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.envs.HARDWARE_GPU_MEMORY_REQUIREMENT",
        "8Gi",
    )
    assert _resolve_gpu_memory(HardwareRequirements(gpu_memory="48Gi")) == "48Gi"
    assert _resolve_gpu_memory(None) == "8Gi"


# ---------------------------------------------------------------------------
# Hardware signature stability for partition keys
# ---------------------------------------------------------------------------


def test_hardware_signature_none_collapses_to_all_none() -> None:
    """Two jobs with no override must produce the same signature so they
    co-batch (and so we don't split partitions on a no-op."""
    assert _hardware_signature(None) == (None, None, None, None)


def test_hardware_signature_distinct_cpu_partitions_distinct() -> None:
    sig_a = _hardware_signature(HardwareRequirements(cpu=8))
    sig_b = _hardware_signature(HardwareRequirements(cpu=16))
    assert sig_a != sig_b


def test_hardware_signature_partial_override_distinct_from_full_none() -> None:
    """Setting just cpu still differs from no-override — otherwise a partial
    override would silently co-batch with default-env jobs."""
    assert _hardware_signature(HardwareRequirements(cpu=8)) != _hardware_signature(None)


def test_hardware_signature_equal_overrides_match() -> None:
    sig_a = _hardware_signature(
        HardwareRequirements(cpu=8, memory="16Gi", gpu=1, gpu_memory="24Gi")
    )
    sig_b = _hardware_signature(
        HardwareRequirements(cpu=8, memory="16Gi", gpu=1, gpu_memory="24Gi")
    )
    assert sig_a == sig_b


# ---------------------------------------------------------------------------
# _any_graph_requires_gpu
# ---------------------------------------------------------------------------


def _cpu_only_compiled() -> Any:
    """CompiledGraph whose only op is a CPU-only DataRetrievalOp."""
    symbol = input_placeholder("Symbol")
    retrieval = DataRetrievalOp(
        data_spec={
            "type": "lumid",
            "mode": "sql",
            "template": "SELECT * FROM t WHERE symbol = :symbol",
            "params": [{"name": "symbol", "node": symbol.id}],
        },
        inputs=[symbol],
    )
    output = as_output("result", retrieval)
    return Graph.from_ops([output]).compile(Symbol=["NVDA"])


def _gpu_compiled() -> Any:
    """CompiledGraph containing an LLMChatOp — RuntimeGraphBuilder assigns the
    ``vllm`` backend, which ``_runtime_op_requires_gpu`` classifies as GPU."""
    prompt = input_placeholder("Prompt")
    llm = LLMChatOp(
        [OpMessage(role="user", content=prompt)],
        config=GenerationConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
    )
    output = as_output("result", llm)
    return Graph.from_ops([output]).compile(Prompt=["hello"])


def _make_server_stub() -> Any:
    return type("_S", (), {"_runtime_builder": RuntimeGraphBuilder()})()


def test_any_graph_requires_gpu_detects_vllm_backend_in_real_compiled_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must build a RuntimeGraph from each CompiledGraph before
    classifying ops so it sees the ``vllm`` backend on ``LLMChatOp``."""
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")

    assert _any_graph_requires_gpu(_make_server_stub(), {"g": _gpu_compiled()}) is True


def test_any_graph_requires_gpu_false_for_cpu_only_real_compiled_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")

    assert (
        _any_graph_requires_gpu(_make_server_stub(), {"g": _cpu_only_compiled()})
        is False
    )


def test_any_graph_requires_gpu_scans_multiple_real_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")

    graphs = {"cpu_only": _cpu_only_compiled(), "has_gpu": _gpu_compiled()}
    assert _any_graph_requires_gpu(_make_server_stub(), graphs) is True
