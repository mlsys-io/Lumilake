"""Capability-gated task types in the shared helpers and the capacity claim."""

from typing import Any

from support.worker_profiles import worker_profile

from lumilake_hook import GpuDevice, WorkerProfile
from lumilake_server.data_specs import EmptyData, GpuRequest, PythonData
from lumilake_server.runtime.capacity import FreeCapacity
from lumilake_server.runtime.worker_capability import (
    CAPABILITY_GATED_TASK_TYPES,
    SessionLane,
    profile_fits_gpu_request,
    python_step_session_lane,
    requires_cpu_worker,
    session_limits_admit,
)


def test_only_python_is_gated_today() -> None:
    assert CAPABILITY_GATED_TASK_TYPES == frozenset({"python"})


def test_requires_cpu_worker_for_cpu_only_task_types() -> None:
    nodes = [("inference", EmptyData()), ("api", EmptyData())]
    assert requires_cpu_worker(nodes)
    assert requires_cpu_worker([(" Data_Retrieval ", EmptyData())])
    assert not requires_cpu_worker([("embedding", EmptyData())])
    assert not requires_cpu_worker([])


def _free(profiles: dict[str, WorkerProfile]) -> FreeCapacity:
    return FreeCapacity(
        cpu_worker_ids=tuple(profiles), gpu_worker_ids=(), profiles=profiles
    )


def _meets_everything(profile: WorkerProfile, hardware: Any) -> bool:
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


_PLAIN = worker_profile(task_types={"echo"})
_PYTHON = worker_profile(task_types={"python"})


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


_H100 = GpuDevice(name="H100", memory_total_bytes=24 * 1024**3)
_A100 = GpuDevice(name="A100", memory_total_bytes=40 * 1024**3)


_GPU_VLLM = worker_profile(devices=(_A100,), task_types={"echo"})
_GPU_PYTHON = worker_profile(devices=(_A100,), task_types={"python"})


def _gpu_free(profiles: dict[str, WorkerProfile]) -> FreeCapacity:
    return FreeCapacity(
        cpu_worker_ids=("cpu-python",),
        gpu_worker_ids=tuple(profiles),
        profiles={"cpu-python": _PYTHON, **profiles},
    )


def _gpu_claim(
    free: FreeCapacity, group_size: int = 1, **required: Any
) -> list[str] | None:
    selected = free.eligible_workers(
        cpu_group_size=1,
        gpu_group_size=group_size,
        worker_meets_hardware=_meets_everything,
        hardware=None,
        **required,
    )
    return None if selected is None else selected[1]


def test_gpu_group_claims_a_worker_that_advertises_python_and_carries_the_request() -> (
    None
):
    free = _gpu_free({"gpu-vllm": _GPU_VLLM, "gpu-python": _GPU_PYTHON})
    one = (GpuRequest(),)
    assert _gpu_claim(free, required_gpu_requests=one) == ["gpu-python"]
    plain = _gpu_free({"gpu-vllm": _GPU_VLLM})
    assert _gpu_claim(plain, required_gpu_requests=one) is None

    cards = {
        "gpu-rtx": worker_profile(
            devices=(GpuDevice(name="RTX 4090"),), task_types={"python"}
        ),
        "gpu-big": worker_profile(devices=(_H100, _H100), task_types={"python"}),
    }
    passes = GpuRequest(count=2, type="H100")
    assert _gpu_claim(_gpu_free(cards), required_gpu_requests=(passes,)) == ["gpu-big"]
    small = _gpu_free(
        {"gpu-rtx": worker_profile(devices=(_H100,), task_types={"python"})}
    )
    assert _gpu_claim(small, required_gpu_requests=(passes,)) is None
    split = _gpu_free(
        {
            "gpu-h100": worker_profile(devices=(_H100,)),
            "gpu-rtx": cards["gpu-rtx"],
        }
    )
    h100 = GpuRequest(type="H100")
    assert _gpu_claim(split, 2, required_gpu_requests=(h100,)) is None


def test_the_fit_rule_reads_type_wildcards_and_binary_memory() -> None:
    l4 = worker_profile(
        devices=(GpuDevice(name="L4", memory_total_bytes=16 << 30),),
        task_types={"python"},
    )
    for wildcard in ("any", "auto", "*", "ANY"):
        assert profile_fits_gpu_request(l4, GpuRequest(type=wildcard))
    assert not profile_fits_gpu_request(l4, GpuRequest(type="H100"))
    assert not profile_fits_gpu_request(l4, GpuRequest(memory="24Gi"))
    assert profile_fits_gpu_request(l4, GpuRequest(memory="16Gi"))


def test_two_slot_group_serves_one_request_per_worker() -> None:
    """An H100 step and an A100 step share a group of two, one worker each."""
    profiles = {
        "gpu-h100": worker_profile(devices=(_H100,), task_types={"python"}),
        "gpu-a100": worker_profile(devices=(_A100,), task_types={"python"}),
    }
    h100 = GpuRequest(count=1, type="H100")
    a100 = GpuRequest(count=1, type="A100")
    free = _gpu_free(profiles)
    assert sorted(_gpu_claim(free, 2, required_gpu_requests=(h100, a100)) or []) == [
        "gpu-a100",
        "gpu-h100",
    ]
    assert _gpu_claim(free, 1, required_gpu_requests=(h100, a100)) is None
    # One slot is enough when the same worker carries both requests.
    both = _gpu_free(
        {"gpu-two": worker_profile(devices=(_H100, _A100), task_types={"python"})}
    )
    assert _gpu_claim(both, 1, required_gpu_requests=(h100, a100)) == ["gpu-two"]
    # Two A100s ahead of the H100: the swap keeps one A100 and takes the H100.
    order = _gpu_free(
        {
            "a": profiles["gpu-a100"],
            "b": profiles["gpu-a100"],
            "c": profiles["gpu-h100"],
        }
    )
    assert _gpu_claim(order, 2, required_gpu_requests=(a100, h100)) == ["a", "c"]
    # Leading workers in no gate give way to both python workers.
    lead = {
        "v1": _GPU_VLLM,
        "v2": _GPU_VLLM,
        "a": profiles["gpu-a100"],
        "c": profiles["gpu-h100"],
    }
    claim = _gpu_claim(
        _gpu_free(lead),
        2,
        required_gpu_requests=(a100, h100),
    )
    assert sorted(claim or []) == ["a", "c"]


GIB = 1024**3


def test_python_step_session_lane_splits_the_cpu_and_gpu_groups() -> None:
    nodes = [
        ("python", PythonData()),
        (" Python ", PythonData(memory_mb=4096)),
        ("python", PythonData(memory_mb=8192, gpu=GpuRequest())),
        ("inference", EmptyData()),
    ]
    assert python_step_session_lane(nodes) == SessionLane(
        cpu_memory_mb=frozenset({None, 4096}), gpu_memory_mb=frozenset({8192})
    )
    assert python_step_session_lane([("inference", EmptyData())]) == SessionLane()


def _capped(cpu_cap: float | None, memory_cap: int | None) -> WorkerProfile:
    return WorkerProfile(
        cpu_cores=32,
        memory_total_bytes=29 * GIB,
        max_session_cpu_cores=cpu_cap,
        max_session_memory_bytes=memory_cap,
    )


def test_session_limits_admit_refuses_a_cpu_cap_below_the_job() -> None:
    assert not session_limits_admit(_capped(4.0, None), 8, None)


def test_session_limits_admit_passes_what_the_profile_does_not_report() -> None:
    assert session_limits_admit(_capped(None, None), 8, 64 * GIB)
    assert session_limits_admit(WorkerProfile(max_session_memory_bytes=GIB), 8, 2 * GIB)
