"""Integration test: _install_plugins_sync registers OptimizerProvider bindings.

Pre-fix: _resolve_subprocess_install discarded the returned bindings, so
OPTIMIZER_PROVIDERS in the subprocess stayed empty and create_optimizer raised
"Unknown optimizer type".  Post-fix: the returned HookBindings are passed to
hooks.register(), making provider-advertised types visible.
"""

import logging
import queue
import sys
import types
from collections.abc import Generator
from typing import Any

import lumid_hooks
import pytest
from lumilake import envs
from lumilake_hook import BaseBindings

from lumilake_server.hooks import IDENTITY_PROVIDERS
from lumilake_server.runtime.optimizer import OPTIMIZER_PROVIDERS, OPTIMIZER_TYPES
from lumilake_server.runtime.optimizer.base import BaseOptimizer, Schedule
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp
from lumilake_server.runtime.server import (
    _install_plugins_sync,
    _optimizer_subprocess_entry,
)

# ---------------------------------------------------------------------------
# Minimal fake optimizer + provider
# ---------------------------------------------------------------------------


class _FakeOptimizer(BaseOptimizer):
    def generate_schedule(
        self,
        graph: RuntimeGraph,
        worker_names: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        data_profile_results: dict[str, list[dict[str, Any]]] | None = None,
    ) -> Schedule:
        assignment: dict[str, list[str]] = {w: [] for w in worker_names}
        for node_id in graph.node_order:
            if worker_names:
                assignment[worker_names[0]].append(node_id)
        return Schedule(worker_assignment=assignment)


class _FakeProvider:
    def list_optimizers(self) -> list[str]:
        return ["fake-remote-x"]

    def create_optimizer(self, optimizer_type: str, **kwargs: Any) -> BaseOptimizer:
        return _FakeOptimizer(**kwargs)


# ---------------------------------------------------------------------------
# Fake plugin module installed into sys.modules for the duration of the test
# ---------------------------------------------------------------------------

_FAKE_MODULE_NAME = "tests_fake_plugin_subprocess_provider"


def _make_fake_plugin_module() -> types.ModuleType:
    mod = types.ModuleType(_FAKE_MODULE_NAME)

    def install() -> BaseBindings:
        return BaseBindings(optimizer_providers=(_FakeProvider(),))

    mod.install = install  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_providers():
    """Snapshot and restore OPTIMIZER_PROVIDERS around each test."""
    snapshot = list(OPTIMIZER_PROVIDERS)
    OPTIMIZER_PROVIDERS.clear()
    yield
    OPTIMIZER_PROVIDERS.clear()
    OPTIMIZER_PROVIDERS.extend(snapshot)


@pytest.fixture()
def fake_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[types.ModuleType, None, None]:
    mod = _make_fake_plugin_module()
    sys.modules[_FAKE_MODULE_NAME] = mod
    monkeypatch.setenv("LUMILAKE_PLUGINS", _FAKE_MODULE_NAME)
    # Also patch the envs list that _install_plugins_sync reads
    monkeypatch.setattr(envs, "LUMILAKE_PLUGINS", [_FAKE_MODULE_NAME])
    yield mod
    sys.modules.pop(_FAKE_MODULE_NAME, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_install_plugins_sync_registers_provider(
    fake_plugin: types.ModuleType,
) -> None:
    """_install_plugins_sync must call hooks.register() with the returned bindings."""
    assert "fake-remote-x" not in OPTIMIZER_TYPES
    assert not any("fake-remote-x" in p.list_optimizers() for p in OPTIMIZER_PROVIDERS)

    _install_plugins_sync()

    assert any(
        "fake-remote-x" in p.list_optimizers() for p in OPTIMIZER_PROVIDERS
    ), "Provider was not registered — bindings from install() were discarded"


def test_install_plugins_sync_none_return_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plugins whose install() returns None must not raise."""
    mod = types.ModuleType("tests_fake_plugin_none_return")

    def install() -> None:
        return None

    mod.install = install  # type: ignore[attr-defined]
    sys.modules["tests_fake_plugin_none_return"] = mod
    monkeypatch.setattr(envs, "LUMILAKE_PLUGINS", ["tests_fake_plugin_none_return"])
    try:
        _install_plugins_sync()  # must not raise
    finally:
        sys.modules.pop("tests_fake_plugin_none_return", None)

    # OPTIMIZER_PROVIDERS unchanged (still empty per autouse fixture)
    assert OPTIMIZER_PROVIDERS == []


def test_optimizer_subprocess_entry_uses_registered_provider(
    fake_plugin: types.ModuleType,
) -> None:
    """_optimizer_subprocess_entry must succeed with a provider-advertised type.

    This test calls the entry function in-process (bypassing mp.Process) so
    that monkeypatching of envs and hooks is effective.  It verifies the full
    install → register → create_optimizer path rather than the subprocess
    transport layer.
    """
    op = RuntimeOp(
        node_id="n1",
        task_type="inference",
        backend="vllm",
        model="gpt2",
        data_spec={},
        model_spec={},
        inference_spec={},
        dependencies=(),
    )
    graph = RuntimeGraph(
        nodes={"n1": op},
        node_order=["n1"],
        output_node_map={"n1": "out"},
        dsl_to_runtime={},
    )

    result_q: queue.Queue[Any] = queue.Queue(maxsize=1)
    _optimizer_subprocess_entry(
        optimizer_type="fake-remote-x",
        runtime_graph=graph,
        selected_workers=["w1"],
        worker_profiles={"w1": {}},
        data_profile_results={},
        result_queue=result_q,
    )

    payload = result_q.get_nowait()
    assert payload.get("ok") is True, (
        f"_optimizer_subprocess_entry failed: {payload.get('error')}\n"
        f"{payload.get('traceback', '')}"
    )
    schedule = payload["schedule"]
    assert "w1" in schedule.worker_assignment


def test_install_plugins_sync_registers_shared_only_identity_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plugin returning bare lumid_hooks.BaseBindings (no optimizer_providers)
    must have its identity_providers registered via _install_plugins_sync.

    Pre-fix the subprocess isinstance check used lumilake_hook.HookBindings, so
    shared-only bindings triggered the "does not satisfy HookBindings" warning and
    were skipped.  Post-fix the check uses lumid_hooks.HookBindings and delegates
    the optimizer_providers split to hooks.register() internally.
    """

    class _FakeIdp:
        async def resolve(
            self, raw_token: str, logger: logging.Logger
        ) -> lumid_hooks.PrincipalContext | None:
            return None

    fake_idp: lumid_hooks.IdentityProvider = _FakeIdp()  # type: ignore[assignment]

    mod_name = "tests_fake_plugin_shared_only_idp"
    mod = types.ModuleType(mod_name)

    def install() -> lumid_hooks.BaseBindings:
        return lumid_hooks.BaseBindings(identity_providers=(fake_idp,))

    mod.install = install  # type: ignore[attr-defined]
    sys.modules[mod_name] = mod
    monkeypatch.setattr(envs, "LUMILAKE_PLUGINS", [mod_name])

    snapshot_idp = list(IDENTITY_PROVIDERS)
    IDENTITY_PROVIDERS.clear()
    try:
        _install_plugins_sync()
        assert (
            fake_idp in IDENTITY_PROVIDERS
        ), "shared-only lumid_hooks.BaseBindings identity_provider was not registered"
    finally:
        IDENTITY_PROVIDERS.clear()
        IDENTITY_PROVIDERS.extend(snapshot_idp)
        sys.modules.pop(mod_name, None)
