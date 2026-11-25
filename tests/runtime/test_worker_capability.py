"""Capability-gated task types in the shared helpers and the capacity claim."""

from typing import Any

from lumilake_server.data_specs import EmptyData, GpuRequest
from lumilake_server.runtime.capacity import FreeCapacity
from lumilake_server.runtime.worker_capability import (
    advertised_task_types,
    required_task_types,
    requires_cpu_worker,
)


def test_only_python_is_gated_today() -> None:
    assert required_task_types(["api", "python", "inference"]) == {"python"}
    assert required_task_types(["api", "data_retrieval"]) == frozenset()


def test_requires_cpu_worker_for_cpu_only_task_types() -> None:
    nodes = [("inference", EmptyData()), ("api", EmptyData())]
    assert requires_cpu_worker(nodes)
    assert requires_cpu_worker([(" Data_Retrieval ", EmptyData())])
    assert not requires_cpu_worker([("embedding", EmptyData())])
    assert not requires_cpu_worker([])


def test_advertised_task_types_reads_the_profile() -> None:
    assert advertised_task_types({"supported_task_types": ["echo", "python"]}) == {
        "echo",
        "python",
    }
    assert advertised_task_types({}) == frozenset()


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


_GPU_VLLM = {"has_gpu": True, "supported_task_types": ["echo"]}
_GPU_PYTHON = {"has_gpu": True, "supported_task_types": ["python"]}


def _gpu_free(profiles: dict[str, dict[str, Any]]) -> FreeCapacity:
    return FreeCapacity(
        cpu_worker_ids=("cpu-python",),
        gpu_worker_ids=tuple(profiles),
        profiles={"cpu-python": _PYTHON, **profiles},
    )


def _gpu_claim(free: FreeCapacity, **required: Any) -> list[str] | None:
    selected = free.eligible_workers(
        cpu_group_size=1,
        gpu_group_size=1,
        worker_meets_hardware=_meets_everything,
        hardware=None,
        **required,
    )
    return None if selected is None else selected[1]


def test_gpu_group_claims_a_worker_that_advertises_python_and_carries_the_request() -> (
    None
):
    required = frozenset({"python"})
    free = _gpu_free({"gpu-vllm": _GPU_VLLM, "gpu-python": _GPU_PYTHON})
    assert _gpu_claim(free, required_gpu_task_types=required) == ["gpu-python"]
    plain = _gpu_free({"gpu-vllm": _GPU_VLLM})
    assert _gpu_claim(plain, required_gpu_task_types=required) is None
    assert not plain.has_gpu_worker_advertising(required)

    h100 = {"name": "H100", "memory_total_bytes": 24 * 1024**3}
    cards = {
        "gpu-rtx": {**_GPU_PYTHON, "gpu": {"devices": [{"name": "RTX 4090"}]}},
        "gpu-big": {**_GPU_PYTHON, "gpu": {"devices": [h100, h100]}},
    }
    passes = GpuRequest(count=2, type="H100")
    assert _gpu_claim(_gpu_free(cards), required_gpu_requests=(passes,)) == ["gpu-big"]
    small = _gpu_free({"gpu-rtx": {**_GPU_PYTHON, "gpu": {"devices": [h100]}}})
    assert _gpu_claim(small, required_gpu_requests=(passes,)) is None
    assert not small.has_gpu_worker_carrying((passes,))
