"""Worker placement facts for tests."""

from collections.abc import Collection

from lumilake_hook import GpuDevice, WorkerGpu, WorkerProfile


def worker_profile(
    *,
    devices: Collection[GpuDevice] = (),
    task_types: Collection[str] = (),
    gpu_memory_fraction: float | None = None,
    cpu_cores: int | None = None,
    memory_total_bytes: int | None = None,
    max_session_cpu_cores: float | None = None,
    max_session_memory_bytes: int | None = None,
) -> WorkerProfile:
    """A worker profile whose GPU section exists only when it carries cards."""
    cards = tuple(devices)
    return WorkerProfile(
        supported_task_types=frozenset(task_types),
        gpu=WorkerGpu(devices=cards) if cards else None,
        gpu_memory_fraction=gpu_memory_fraction,
        cpu_cores=cpu_cores,
        memory_total_bytes=memory_total_bytes,
        max_session_cpu_cores=max_session_cpu_cores,
        max_session_memory_bytes=max_session_memory_bytes,
    )
