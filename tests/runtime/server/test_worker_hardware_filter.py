"""``_worker_meets_hardware`` filters out workers that can't satisfy the
per-job hardware override before scheduler dispatch / preview selection."""

from typing import Any

import pytest

from lumilake_server.runtime.protocol import HardwareRequirements
from lumilake_server.runtime.server import LumilakeServer


def _profile(
    cores: int | None = 16,
    mem_bytes: int | None = 64 * (1024**3),
    gpu_count: int = 0,
    gpu_mem_bytes: int | None = None,
    gpu_names: list[str] | None = None,
) -> dict[str, Any]:
    names = gpu_names or ["NVIDIA GeForce RTX 5080"] * gpu_count
    devices: list[dict[str, Any]] = [
        {"memory_total_bytes": gpu_mem_bytes, "name": names[i]}
        for i in range(gpu_count)
    ]
    return {
        "cpu": {"logical_cores": cores},
        "memory": {"total_bytes": mem_bytes},
        "gpu": {"devices": devices},
    }


def test_no_requirement_accepts_every_worker() -> None:
    assert LumilakeServer._worker_meets_hardware(_profile(), None) is True


def test_meets_when_worker_exceeds_requirement() -> None:
    hw = HardwareRequirements(cpu=8, memory="16Gi", gpu=1, gpu_memory="8Gi")
    profile = _profile(
        cores=32,
        mem_bytes=64 * (1024**3),
        gpu_count=2,
        gpu_mem_bytes=24 * (1024**3),
    )
    assert LumilakeServer._worker_meets_hardware(profile, hw) is True


def test_rejects_when_cpu_below_requirement() -> None:
    hw = HardwareRequirements(cpu=32)
    assert LumilakeServer._worker_meets_hardware(_profile(cores=8), hw) is False


def test_rejects_when_memory_below_requirement() -> None:
    hw = HardwareRequirements(memory="128Gi")
    assert (
        LumilakeServer._worker_meets_hardware(_profile(mem_bytes=16 * (1024**3)), hw)
        is False
    )


def test_rejects_when_gpu_count_below_requirement() -> None:
    hw = HardwareRequirements(gpu=2)
    assert LumilakeServer._worker_meets_hardware(_profile(gpu_count=1), hw) is False


def test_rejects_when_gpu_memory_below_requirement() -> None:
    hw = HardwareRequirements(gpu=1, gpu_memory="48Gi")
    profile = _profile(gpu_count=1, gpu_mem_bytes=24 * (1024**3))
    assert LumilakeServer._worker_meets_hardware(profile, hw) is False


def test_unknown_worker_field_is_treated_as_no_constraint() -> None:
    """A worker that advertises no ``cpu.logical_cores`` (None) must NOT be
    filtered out — Lumilake's job is to drop the obvious no-go workers, not
    enforce strict typing on FlowMesh's profile."""
    hw = HardwareRequirements(cpu=8)
    assert LumilakeServer._worker_meets_hardware(_profile(cores=None), hw) is True


@pytest.mark.parametrize(
    "memory_str,bytes_value",
    [
        ("1Ki", 1024),
        ("16Gi", 16 * 1024**3),
        ("128Mi", 128 * 1024**2),
        ("2Ti", 2 * 1024**4),
    ],
)
def test_parse_memory_to_bytes_well_formed(memory_str: str, bytes_value: int) -> None:
    assert LumilakeServer._parse_memory_to_bytes(memory_str) == bytes_value


def test_parse_memory_to_bytes_returns_none_for_malformed() -> None:
    """Validation lives at the ``HardwareRequirements`` boundary; the parser's
    only job for an unknown shape is to degrade to "no constraint" so a
    malformed value never crashes the scheduler."""
    assert LumilakeServer._parse_memory_to_bytes("abc") is None
    assert LumilakeServer._parse_memory_to_bytes("") is None


# ---------------------------------------------------------------------------
# Role-aware filtering: hardware.gpu must NOT reject CPU workers
# ---------------------------------------------------------------------------


def test_cpu_worker_passes_when_only_gpu_constraint_set() -> None:
    """A CPU-only worker (no GPU devices) must still pass when the user
    requests ``hardware.gpu=N``. The GPU count constraint only applies to
    GPU-capable workers; otherwise mixed CPU+GPU graphs stall because every
    CPU worker is rejected for "missing N GPUs"."""
    hw = HardwareRequirements(gpu=2, gpu_memory="48Gi")
    cpu_only = _profile(cores=32, mem_bytes=64 * (1024**3), gpu_count=0)
    assert LumilakeServer._worker_meets_hardware(cpu_only, hw) is True


def test_cpu_worker_still_checked_for_cpu_and_memory() -> None:
    """CPU/memory constraints apply to every worker regardless of role."""
    hw = HardwareRequirements(cpu=32, gpu=1)
    small_cpu_worker = _profile(cores=8, gpu_count=0)
    assert LumilakeServer._worker_meets_hardware(small_cpu_worker, hw) is False


def test_gpu_worker_still_checked_for_gpu_constraints() -> None:
    hw = HardwareRequirements(gpu=4)
    one_gpu = _profile(gpu_count=1, gpu_mem_bytes=24 * (1024**3))
    assert LumilakeServer._worker_meets_hardware(one_gpu, hw) is False


def test_explicit_worker_has_gpu_hint_avoids_redundant_classification() -> None:
    """Callers that already classified the worker can pass ``worker_has_gpu``
    to skip the internal ``_has_gpu`` lookup."""
    hw = HardwareRequirements(gpu=2)
    cpu_only = _profile(gpu_count=0)
    assert (
        LumilakeServer._worker_meets_hardware(cpu_only, hw, worker_has_gpu=False)
        is True
    )


# ---------------------------------------------------------------------------
# gpu_model: substring, case-insensitive, GPU-role-only
# ---------------------------------------------------------------------------


def test_unset_gpu_model_changes_nothing() -> None:
    """Backward-compat guard: omitting ``gpu_model`` must not alter the
    existing filter behavior."""
    hw = HardwareRequirements(gpu=1, gpu_memory="24Gi")
    profile = _profile(gpu_count=1, gpu_mem_bytes=24 * (1024**3))
    assert LumilakeServer._worker_meets_hardware(profile, hw) is True


def test_gpu_model_matches_device_name_substring() -> None:
    hw = HardwareRequirements(gpu_model="RTX 5080")
    profile = _profile(gpu_count=1, gpu_names=["NVIDIA GeForce RTX 5080"])
    assert LumilakeServer._worker_meets_hardware(profile, hw) is True


def test_gpu_model_matches_short_substring() -> None:
    """``"5080"`` matches ``"NVIDIA GeForce RTX 5080"`` without the vendor's
    exact string."""
    hw = HardwareRequirements(gpu_model="5080")
    profile = _profile(gpu_count=1, gpu_names=["NVIDIA GeForce RTX 5080"])
    assert LumilakeServer._worker_meets_hardware(profile, hw) is True


def test_gpu_model_matching_is_case_insensitive() -> None:
    hw = HardwareRequirements(gpu_model="rtx 5080")
    profile = _profile(gpu_count=1, gpu_names=["NVIDIA GeForce RTX 5080"])
    assert LumilakeServer._worker_meets_hardware(profile, hw) is True


def test_gpu_model_rejects_non_matching_device() -> None:
    hw = HardwareRequirements(gpu_model="RTX 5090")
    profile = _profile(gpu_count=1, gpu_names=["NVIDIA GeForce RTX 5080"])
    assert LumilakeServer._worker_meets_hardware(profile, hw) is False


def test_gpu_model_ignores_cpu_only_worker() -> None:
    """A CPU-only worker is unaffected by ``gpu_model`` — the constraint only
    applies to GPU-capable workers."""
    hw = HardwareRequirements(gpu_model="RTX 5080")
    cpu_only = _profile(gpu_count=0)
    assert LumilakeServer._worker_meets_hardware(cpu_only, hw) is True


def test_gpu_model_unknown_device_names_still_pass() -> None:
    """A GPU worker whose devices report no ``name`` is treated as unknown and
    passes (unknown-passes rule)."""
    hw = HardwareRequirements(gpu_model="RTX 5080")
    profile = {
        "cpu": {"logical_cores": 16},
        "memory": {"total_bytes": 64 * (1024**3)},
        "gpu": {"devices": [{"memory_total_bytes": 24 * (1024**3)}]},
    }
    assert LumilakeServer._worker_meets_hardware(profile, hw) is True
