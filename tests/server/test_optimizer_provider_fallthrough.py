"""Tests for create_optimizer fall-through to registered OptimizerProviders."""

from typing import Any

import pytest

from lumilake_server.runtime.optimizer import (
    OPTIMIZER_PROVIDERS,
    OPTIMIZER_TYPES,
    create_optimizer,
)
from lumilake_server.runtime.optimizer.base import BaseOptimizer, Schedule
from lumilake_server.runtime.runtime_graph import RuntimeGraph


class _StubOptimizer(BaseOptimizer):
    def generate_schedule(
        self,
        graph: RuntimeGraph,
        worker_names: list[str],
        worker_profiles: dict[str, dict[str, Any]],
        data_profile_results: dict[str, list[dict[str, Any]]] | None = None,
    ) -> Schedule:
        return Schedule(worker_assignment={})


class _StubProvider:
    def __init__(self, types: list[str]) -> None:
        self._types = types
        self.create_calls: list[tuple[str, dict[str, Any]]] = []

    def list_optimizers(self) -> list[str]:
        return list(self._types)

    def create_optimizer(self, optimizer_type: str, **kwargs: Any) -> BaseOptimizer:
        self.create_calls.append((optimizer_type, kwargs))
        return _StubOptimizer()


@pytest.fixture(autouse=True)
def _clean_providers():
    snapshot = list(OPTIMIZER_PROVIDERS)
    OPTIMIZER_PROVIDERS.clear()
    yield
    OPTIMIZER_PROVIDERS.clear()
    OPTIMIZER_PROVIDERS.extend(snapshot)


def test_local_type_returns_local_optimizer() -> None:
    assert "halo" in OPTIMIZER_TYPES
    optimizer = create_optimizer("halo")
    assert isinstance(optimizer, OPTIMIZER_TYPES["halo"])


def test_provider_type_calls_provider_create() -> None:
    provider = _StubProvider(["remote-alpha"])
    OPTIMIZER_PROVIDERS.append(provider)

    optimizer = create_optimizer("remote-alpha")

    assert isinstance(optimizer, _StubOptimizer)
    assert provider.create_calls == [("remote-alpha", {})]


def test_provider_receives_kwargs() -> None:
    provider = _StubProvider(["remote-beta"])
    OPTIMIZER_PROVIDERS.append(provider)

    create_optimizer("remote-beta", extra_param="val")

    assert provider.create_calls == [("remote-beta", {"extra_param": "val"})]


@pytest.mark.parametrize(
    "with_provider",
    [
        pytest.param(False, id="no_providers"),
        pytest.param(True, id="with_providers"),
    ],
)
def test_unknown_type_raises_value_error(with_provider: bool) -> None:
    if with_provider:
        OPTIMIZER_PROVIDERS.append(_StubProvider(["provider-only"]))
    with pytest.raises(ValueError, match="totally-unknown"):
        create_optimizer("totally-unknown")


def test_first_matching_provider_wins() -> None:
    first = _StubProvider(["shared-type"])
    second = _StubProvider(["shared-type"])
    OPTIMIZER_PROVIDERS.extend([first, second])

    create_optimizer("shared-type")

    assert len(first.create_calls) == 1
    assert len(second.create_calls) == 0


def test_local_type_takes_priority_over_provider() -> None:
    provider = _StubProvider(["halo"])
    OPTIMIZER_PROVIDERS.append(provider)

    optimizer = create_optimizer("halo")

    assert isinstance(optimizer, OPTIMIZER_TYPES["halo"])
    assert provider.create_calls == []


@pytest.mark.parametrize("variant", ["remotex", "RemoteX", "REMOTEX"])
def test_provider_mixed_case_name_resolved(variant: str) -> None:
    """create_optimizer resolves a provider's "RemoteX" regardless of input casing."""
    provider = _StubProvider(["RemoteX"])
    OPTIMIZER_PROVIDERS.append(provider)

    optimizer = create_optimizer(variant)

    assert isinstance(optimizer, _StubOptimizer)
    assert provider.create_calls == [("RemoteX", {})]
