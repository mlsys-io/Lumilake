import pytest

from lumilake.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


@pytest.fixture
def flowmesh_manager() -> FlowmeshRuntimeManager:
    return FlowmeshRuntimeManager()
