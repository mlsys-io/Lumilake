import pytest

from lumilake_server.runtime.server import LumilakeServerConfig


def test_server_config_rejects_removed_worker_group_size_alias() -> None:
    with pytest.raises(TypeError):
        LumilakeServerConfig(worker_group_size=1)  # type: ignore[call-arg]


def test_server_config_accepts_cpu_and_gpu_group_sizes() -> None:
    config = LumilakeServerConfig(
        is_local=True,
        runtime_url="http://localhost:18080",
        runtime_token="test-token",
        cpu_worker_group_size=1,
        gpu_worker_group_size=2,
    )
    assert config.cpu_worker_group_size == 1
    assert config.gpu_worker_group_size == 2


def test_server_config_requires_one_non_zero_group_size() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        LumilakeServerConfig(
            is_local=True,
            runtime_url="http://localhost:18080",
            runtime_token="test-token",
            cpu_worker_group_size=0,
            gpu_worker_group_size=0,
        )
