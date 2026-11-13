import pytest

from lumilake import envs
from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager
from lumilake_server.utils.job_storage import InMemoryJobStorage


@pytest.fixture
def flowmesh_manager() -> FlowmeshRuntimeManager:
    return FlowmeshRuntimeManager()


@pytest.fixture(autouse=True)
def job_storage(monkeypatch: pytest.MonkeyPatch) -> InMemoryJobStorage:
    """Keep archived artifacts and published task blobs in memory."""
    storage = InMemoryJobStorage()
    for module in (
        "lumilake_server.runtime.runtime_manager.base",
        "lumilake_server.runtime.runtime_manager.flowmesh",
    ):
        monkeypatch.setattr(f"{module}.get_job_storage", lambda: storage)
    return storage


@pytest.fixture(autouse=True)
def lumid_blob_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")
