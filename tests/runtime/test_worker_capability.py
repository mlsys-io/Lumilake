"""Capability-gated task types in the shared helpers and the capacity claim."""

from typing import Any

from lumilake_server.runtime.capacity import FreeCapacity
from lumilake_server.runtime.worker_capability import (
    advertised_task_types,
    gpu_memory_share,
    required_task_types,
    requires_cpu_worker,
)


def test_only_python_is_gated_today() -> None:
    assert required_task_types(["api", "python", "inference"]) == {"python"}
    assert required_task_types(["api", "data_retrieval"]) == frozenset()


def test_requires_cpu_worker_for_cpu_only_task_types() -> None:
    assert requires_cpu_worker(["inference", "api"])
    assert requires_cpu_worker([" Data_Retrieval "])
    assert requires_cpu_worker(["python"])
    assert not requires_cpu_worker(["inference", "embedding", "diffusion"])
    assert not requires_cpu_worker([])


def test_advertised_task_types_reads_the_profile() -> None:
    assert advertised_task_types({"supported_task_types": ["echo", "python"]}) == {
        "echo",
        "python",
    }
    assert advertised_task_types({}) == frozenset()


def test_gpu_memory_share_reads_a_normalized_profile() -> None:
    assert gpu_memory_share({"has_gpu": True, "gpu_memory_share": 0.1}) == 0.1
    assert gpu_memory_share({"has_gpu": True, "gpu_memory_share": None}) is None


def test_gpu_memory_share_reads_a_raw_device_list() -> None:
    profile = {"gpu": {"devices": [{"memory_share": 0.3}]}}
    assert gpu_memory_share(profile) == 0.3


def test_a_whole_device_reads_as_no_share() -> None:
    assert gpu_memory_share({"gpu": {"devices": [{"memory_share": None}]}}) is None
    assert gpu_memory_share({"gpu": {"devices": [{"index": 0}]}}) is None
    assert gpu_memory_share({"has_gpu": False}) is None
    assert gpu_memory_share({}) is None


def _free(profiles: dict[str, dict[str, Any]]) -> FreeCapacity:
    return FreeCapacity(
        cpu_worker_ids=tuple(profiles), gpu_worker_ids=(), profiles=profiles
    )


def _meets_everything(profile: dict[str, Any], hardware: Any) -> bool:
    return True


def _claim(free: FreeCapacity, required: frozenset[str]) -> list[str] | None:
    selected = free.eligible_workers(
        cpu_group_size=1,
        gpu_group_size=0,
        worker_meets_hardware=_meets_everything,
        hardware=None,
        required_task_types=required,
    )
    return None if selected is None else selected[0]


_PLAIN = {"supported_task_types": ["echo"]}
_PYTHON = {"supported_task_types": ["python"]}


def test_claim_takes_a_worker_that_advertises_the_required_type() -> None:
    free = _free({"cpu-plain": _PLAIN, "cpu-python": _PYTHON})
    assert _claim(free, frozenset({"python"})) == ["cpu-python"]


def test_claim_puts_a_capable_worker_in_a_larger_group() -> None:
    free = _free({"cpu-a": _PLAIN, "cpu-b": _PLAIN, "cpu-python": _PYTHON})
    selected = free.eligible_workers(
        cpu_group_size=2,
        gpu_group_size=0,
        worker_meets_hardware=_meets_everything,
        hardware=None,
        required_task_types=frozenset({"python"}),
    )
    assert selected is not None
    assert len(selected[0]) == 2 and "cpu-python" in selected[0]


def test_claim_waits_when_the_only_capable_worker_is_busy() -> None:
    """A busy worker is absent from the free snapshot; a plain idle worker must
    not be claimed in its place."""
    free = _free({"cpu-plain": _PLAIN})
    assert _claim(free, frozenset({"python"})) is None


def test_claim_without_a_requirement_keeps_the_original_order() -> None:
    free = _free({"cpu-plain": _PLAIN, "cpu-python": _PYTHON})
    assert _claim(free, frozenset()) == ["cpu-plain"]


def test_claim_never_hands_a_gpu_worker_to_the_cpu_group() -> None:
    free = FreeCapacity(
        cpu_worker_ids=("cpu-plain",),
        gpu_worker_ids=("gpu-python",),
        profiles={"cpu-plain": _PLAIN, "gpu-python": _PYTHON},
    )
    selected = free.eligible_workers(
        cpu_group_size=1,
        gpu_group_size=0,
        worker_meets_hardware=_meets_everything,
        hardware=None,
        required_task_types=frozenset({"python"}),
    )
    assert selected is None


def test_cpu_worker_advertising_ignores_gpu_workers() -> None:
    free = FreeCapacity(
        cpu_worker_ids=("cpu-plain",),
        gpu_worker_ids=("gpu-python",),
        profiles={"cpu-plain": _PLAIN, "gpu-python": _PYTHON},
    )
    assert not free.has_cpu_worker_advertising(frozenset({"python"}))
    assert free.has_cpu_worker_advertising(frozenset({"echo"}))


def _gpu_free(profiles: dict[str, dict[str, Any]]) -> FreeCapacity:
    return FreeCapacity(
        cpu_worker_ids=(), gpu_worker_ids=tuple(profiles), profiles=profiles
    )


def _claim_gpu(
    free: FreeCapacity, utilizations: frozenset[float], group_size: int = 1
) -> list[str] | None:
    selected = free.eligible_workers(
        cpu_group_size=0,
        gpu_group_size=group_size,
        worker_meets_hardware=_meets_everything,
        hardware=None,
        required_gpu_utilizations=utilizations,
    )
    return None if selected is None else selected[1]


_SMALL = {"has_gpu": True, "gpu_memory_share": 0.1}
_BIG = {"has_gpu": True, "gpu_memory_share": 0.8}


def test_gpu_claim_takes_a_worker_whose_share_holds_the_utilization() -> None:
    """The idle small worker comes first; the claim still takes the one that fits."""
    free = _gpu_free({"gpu-small": _SMALL, "gpu-big": _BIG})
    assert _claim_gpu(free, frozenset({0.8})) == ["gpu-big"]


def test_gpu_claim_waits_when_the_free_shares_are_too_small() -> None:
    free = _gpu_free({"gpu-small": _SMALL})
    assert _claim_gpu(free, frozenset({0.8})) is None


def test_gpu_claim_holds_every_utilization_the_group_covers() -> None:
    """One node at 0.8 and one at 0.1 need a group whose largest share is 0.8."""
    free = _gpu_free({"gpu-small": _SMALL, "gpu-big": _BIG})
    assert _claim_gpu(free, frozenset({0.1, 0.8}), group_size=2) == [
        "gpu-small",
        "gpu-big",
    ]


def test_gpu_claim_without_a_utilization_keeps_the_original_order() -> None:
    free = _gpu_free({"gpu-small": _SMALL, "gpu-big": _BIG})
    assert _claim_gpu(free, frozenset()) == ["gpu-small"]


def test_gpu_claim_takes_a_whole_device_for_any_utilization() -> None:
    free = _gpu_free({"gpu-whole": {"has_gpu": True, "gpu_memory_share": None}})
    assert _claim_gpu(free, frozenset({0.9})) == ["gpu-whole"]


def test_gpu_worker_holding_ignores_cpu_workers() -> None:
    free = FreeCapacity(
        cpu_worker_ids=("cpu-0",),
        gpu_worker_ids=("gpu-small",),
        profiles={"cpu-0": {}, "gpu-small": _SMALL},
    )
    assert not free.has_gpu_worker_holding(frozenset({0.8}))
    assert free.has_gpu_worker_holding(frozenset({0.1}))
