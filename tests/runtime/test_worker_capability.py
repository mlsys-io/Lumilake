"""Capability-gated task types in the shared helpers and the capacity claim."""

from typing import Any

from lumilake_server.data_specs import EmptyData, GpuRequest
from lumilake_server.runtime.capacity import FreeCapacity
from lumilake_server.runtime.worker_capability import (
    CAPABILITY_GATED_TASK_TYPES,
    advertised_task_types,
    requires_cpu_worker,
)


def test_only_python_is_gated_today() -> None:
    assert CAPABILITY_GATED_TASK_TYPES == frozenset({"python"})


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


_H100 = {"name": "H100", "memory_total_bytes": 24 * 1024**3}
_A100 = {"name": "A100", "memory_total_bytes": 40 * 1024**3}


def _gpu_profile(
    devices: list[dict[str, Any]], task_types: list[str]
) -> dict[str, Any]:
    """A GPU worker profile in the raw shape the server's snapshot stores."""
    return {"gpu": {"devices": devices}, "supported_task_types": task_types}


_GPU_VLLM = _gpu_profile([_A100], ["echo"])
_GPU_PYTHON = _gpu_profile([_A100], ["python"])


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
    assert not plain.gpu_group_serves(required)

    cards = {
        "gpu-rtx": _gpu_profile([{"name": "RTX 4090"}], ["python"]),
        "gpu-big": _gpu_profile([_H100, _H100], ["python"]),
    }
    passes = GpuRequest(count=2, type="H100")
    assert _gpu_claim(_gpu_free(cards), required_gpu_requests=(passes,)) == ["gpu-big"]
    small = _gpu_free({"gpu-rtx": _gpu_profile([_H100], ["python"])})
    assert _gpu_claim(small, required_gpu_requests=(passes,)) is None
    assert not small.gpu_group_serves(requests=(passes,))


def _group_claim(
    free: FreeCapacity, group_size: int, **required: Any
) -> list[str] | None:
    selected = free.eligible_workers(
        cpu_group_size=1,
        gpu_group_size=group_size,
        worker_meets_hardware=_meets_everything,
        hardware=None,
        **required,
    )
    return None if selected is None else selected[1]


def test_a_gpu_type_wildcard_places_no_type_constraint() -> None:
    rtx = {"name": "RTX 4090", "memory_total_bytes": 24 * 1024**3}
    for wildcard in ("any", "auto", "*", "ANY"):
        passes = GpuRequest(count=1, type=wildcard)
        free = _gpu_free({"gpu-rtx": _gpu_profile([rtx], ["python"])})
        assert _gpu_claim(free, required_gpu_requests=(passes,)) == ["gpu-rtx"]
    literal = _gpu_free({"gpu-rtx": _gpu_profile([rtx], ["python"])})
    assert (
        _gpu_claim(literal, required_gpu_requests=(GpuRequest(count=1, type="H100"),))
        is None
    )


def test_memory_floor_uses_binary_units() -> None:
    cards = {
        "gpu-16": _gpu_profile(
            [{"name": "L4", "memory_total_bytes": 16 << 30}], ["python"]
        )
    }
    assert (
        _gpu_claim(_gpu_free(cards), required_gpu_requests=(GpuRequest(memory="24Gi"),))
        is None
    )
    assert _gpu_claim(
        _gpu_free(cards), required_gpu_requests=(GpuRequest(memory="16Gi"),)
    ) == ["gpu-16"]


def test_one_slot_group_needs_a_worker_that_carries_every_gate() -> None:
    """The fraction gate must not swap in a worker the python gate placed out."""
    h100 = {**_gpu_profile([_H100], ["python"]), "gpu_memory_fraction": 0.2}
    other = {**_gpu_profile([_A100], []), "gpu_memory_fraction": 0.9}
    free = _gpu_free({"gpu-h100": h100, "gpu-split": other})
    assert (
        _group_claim(
            free,
            1,
            required_gpu_task_types=frozenset({"python"}),
            required_gpu_utilizations=frozenset({0.9}),
        )
        is None
    )
    assert _group_claim(
        _gpu_free(
            {
                "gpu-h100": {**h100, "gpu_memory_fraction": 0.9},
                "gpu-split": other,
            }
        ),
        1,
        required_gpu_task_types=frozenset({"python"}),
        required_gpu_utilizations=frozenset({0.9}),
    ) == ["gpu-h100"]


def test_two_slot_group_serves_one_request_per_worker() -> None:
    """An H100 step and an A100 step share a two-worker group; neither worker
    carries both requests, so a single-carrier gate would stall the batch."""
    profiles = {
        "gpu-h100": _gpu_profile([_H100], ["python"]),
        "gpu-a100": _gpu_profile([_A100], ["python"]),
    }
    h100 = GpuRequest(count=1, type="H100")
    a100 = GpuRequest(count=1, type="A100")
    free = _gpu_free(profiles)
    assert sorted(_group_claim(free, 2, required_gpu_requests=(h100, a100)) or []) == [
        "gpu-a100",
        "gpu-h100",
    ]
    assert _group_claim(free, 1, required_gpu_requests=(h100, a100)) is None
    assert free.gpu_group_serves(requests=(h100, a100))
    assert not _gpu_free({"gpu-h100": profiles["gpu-h100"]}).gpu_group_serves(
        requests=(h100, a100)
    )
    # One slot is enough when the same worker carries both requests.
    both = _gpu_free({"gpu-two": _gpu_profile([_H100, _A100], ["python"])})
    assert _group_claim(both, 1, required_gpu_requests=(h100, a100)) == ["gpu-two"]
