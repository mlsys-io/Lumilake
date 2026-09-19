"""The base runtime manager's credential and dispatch-token stubs are an
abstract contract: each must raise ``NotImplementedError`` rather than
silently returning ``None``. A silent ``None`` return would let a backend
that never stores credentials appear to work while the caller's secret is
dropped — the same inert-feature pattern this branch keeps catching."""

from typing import cast

import pytest

from lumilake_server.runtime.runtime_manager.base import BaseRuntimeManager


def _bare() -> BaseRuntimeManager:
    """A stand-in ``self`` for invoking the un-overridden base bodies. The
    stubs never touch ``self``, so any object works."""
    return cast(BaseRuntimeManager, object())


def test_base_set_dispatch_token_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        BaseRuntimeManager.set_dispatch_token(_bare(), "req-1", "tok")


def test_base_get_dispatch_token_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        BaseRuntimeManager.get_dispatch_token(_bare(), "req-1")


def test_base_set_api_credential_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        BaseRuntimeManager.set_api_credential(_bare(), "req-1", "Bearer key")


def test_base_get_api_credential_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        BaseRuntimeManager.get_api_credential(_bare(), "req-1")


def test_base_clear_dispatch_token_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        BaseRuntimeManager.clear_dispatch_token(_bare(), "req-1")
