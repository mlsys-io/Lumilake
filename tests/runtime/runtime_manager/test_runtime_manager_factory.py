import pytest

from lumilake.runtime.runtime_manager import create_runtime_manager
from lumilake.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


@pytest.mark.parametrize("runtime_manager_type", ["default", "flowmesh"])
def test_create_runtime_manager_selects_flowmesh(runtime_manager_type: str) -> None:
    manager = create_runtime_manager(runtime_manager_type)
    assert isinstance(manager, FlowmeshRuntimeManager)


def test_create_runtime_manager_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown runtime manager type"):
        create_runtime_manager("unknown")
