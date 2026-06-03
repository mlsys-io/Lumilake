"""Tests for hook registration compatibility with shared lumid_hooks bindings.

Verifies that plugins returning a bare ``lumid_hooks.BaseBindings`` (no
``optimizer_providers`` field) have their shared fields — e.g.
``identity_providers`` — registered correctly by the main-process
``register()`` function.
"""

import logging

import lumid_hooks
import pytest

from lumilake_server.hooks import IDENTITY_PROVIDERS, register
from lumilake_server.runtime.optimizer import OPTIMIZER_PROVIDERS


class _FakeIdentityProvider:
    async def resolve(
        self, raw_token: str, logger: logging.Logger
    ) -> lumid_hooks.PrincipalContext | None:
        return None


@pytest.fixture(autouse=True)
def _clean_registries():
    idp_snapshot = list(IDENTITY_PROVIDERS)
    op_snapshot = list(OPTIMIZER_PROVIDERS)
    IDENTITY_PROVIDERS.clear()
    OPTIMIZER_PROVIDERS.clear()
    yield
    IDENTITY_PROVIDERS.clear()
    IDENTITY_PROVIDERS.extend(idp_snapshot)
    OPTIMIZER_PROVIDERS.clear()
    OPTIMIZER_PROVIDERS.extend(op_snapshot)


def test_shared_only_bindings_registers_identity_provider() -> None:
    """A plugin returning lumid_hooks.BaseBindings (no optimizer_providers) must
    have its identity_providers registered by register()."""
    fake_idp: lumid_hooks.IdentityProvider = _FakeIdentityProvider()  # type: ignore[assignment]
    bindings = lumid_hooks.BaseBindings(identity_providers=(fake_idp,))

    assert isinstance(bindings, lumid_hooks.HookBindings)

    register(bindings)

    assert fake_idp in IDENTITY_PROVIDERS


def test_shared_only_bindings_does_not_extend_optimizer_providers() -> None:
    """Shared-only bindings must not add anything to OPTIMIZER_PROVIDERS."""
    bindings = lumid_hooks.BaseBindings()
    register(bindings)
    assert OPTIMIZER_PROVIDERS == []
