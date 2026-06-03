"""Tests for the OptimizerProvider Protocol and BaseBindings.optimizer_providers."""

from typing import Any

from lumilake_hook import BaseBindings, OptimizerProvider


class _ConcreteProvider:
    def list_optimizers(self) -> list[str]:
        return ["halo-greedy"]

    def create_optimizer(self, optimizer_type: str, **kwargs: Any) -> Any:
        return object()


class _MissingCreate:
    def list_optimizers(self) -> list[str]:
        return []


class _MissingList:
    def create_optimizer(self, optimizer_type: str, **kwargs: Any) -> Any:
        return object()


def test_concrete_provider_satisfies_protocol() -> None:
    provider = _ConcreteProvider()
    assert isinstance(provider, OptimizerProvider)


def test_missing_create_optimizer_does_not_satisfy_protocol() -> None:
    assert not isinstance(_MissingCreate(), OptimizerProvider)


def test_missing_list_optimizers_does_not_satisfy_protocol() -> None:
    assert not isinstance(_MissingList(), OptimizerProvider)


def test_base_bindings_default_optimizer_providers_is_empty() -> None:
    bindings = BaseBindings()
    assert list(bindings.optimizer_providers) == []


def test_base_bindings_accepts_optimizer_providers() -> None:
    provider = _ConcreteProvider()
    bindings = BaseBindings(optimizer_providers=(provider,))
    providers = list(bindings.optimizer_providers)
    assert len(providers) == 1
    assert providers[0] is provider


def test_base_bindings_is_frozen() -> None:
    bindings = BaseBindings(optimizer_providers=(_ConcreteProvider(),))
    try:
        bindings.optimizer_providers = ()  # type: ignore[misc]
    except (AttributeError, TypeError):
        pass
    else:
        raise AssertionError("BaseBindings should be frozen")
