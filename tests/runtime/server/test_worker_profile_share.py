"""A normalized worker profile keeps the share of a card the worker owns, so a
placement decision made from that profile can see a co-tenant's share."""

from typing import Any

from lumilake_server.runtime.server import LumilakeServer


def _raw_profile(**extra: Any) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "gpu": {"devices": [{"index": 0, "memory_share": 0.1}]},
        "supported_task_types": ["echo"],
    }
    profile.update(extra)
    return profile


def test_normalized_profile_carries_the_share() -> None:
    normalized = LumilakeServer._normalize_worker_profile(_raw_profile())

    assert normalized["gpu_memory_share"] == 0.1


def test_a_whole_device_worker_normalizes_to_no_share() -> None:
    normalized = LumilakeServer._normalize_worker_profile(
        _raw_profile(gpu={"devices": [{"index": 0, "memory_share": None}]})
    )

    assert normalized["gpu_memory_share"] is None


def test_a_cpu_worker_normalizes_to_no_share() -> None:
    normalized = LumilakeServer._normalize_worker_profile({"supported_task_types": []})

    assert normalized["gpu_memory_share"] is None
    assert normalized["has_gpu"] is False
