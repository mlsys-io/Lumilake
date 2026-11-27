"""Worker placement facts for tests."""

from lumilake_server.runtime.worker_capability import GpuDevice, WorkerProfile


def worker_profile(
    *,
    has_gpu: bool = False,
    task_types: frozenset[str] = frozenset(),
    gpu_memory_fraction: float | None = None,
    cpu_cores: int | None = None,
    memory_total_bytes: int | None = None,
    gpu_devices: tuple[GpuDevice, ...] = (),
    max_session_cpu_cores: float | None = None,
    max_session_memory_bytes: int | None = None,
) -> WorkerProfile:
    """A worker profile with the facts a test cares about and defaults elsewhere."""
    return WorkerProfile(
        has_gpu=has_gpu,
        supported_task_types=task_types,
        gpu_memory_fraction=gpu_memory_fraction,
        cpu_cores=cpu_cores,
        memory_total_bytes=memory_total_bytes,
        gpu_devices=gpu_devices,
        max_session_cpu_cores=max_session_cpu_cores,
        max_session_memory_bytes=max_session_memory_bytes,
    )
