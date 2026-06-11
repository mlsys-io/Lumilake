"""Tests for GET /api/v1/optimizer."""

import logging
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lumilake_server.routes import optimizer as optimizer_routes
from lumilake_server.runtime.optimizer import OPTIMIZER_PROVIDERS, OPTIMIZER_TYPES


class _StubProvider:
    def __init__(self, types: list[str]) -> None:
        self._types = types

    def list_optimizers(self) -> list[str]:
        return list(self._types)

    def create_optimizer(self, optimizer_type: str, **kwargs: Any) -> Any:
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _clean_providers():
    snapshot = list(OPTIMIZER_PROVIDERS)
    OPTIMIZER_PROVIDERS.clear()
    yield
    OPTIMIZER_PROVIDERS.clear()
    OPTIMIZER_PROVIDERS.extend(snapshot)


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.state.logger = logging.getLogger("test.optimizer_list")
    app.include_router(optimizer_routes.router)
    return TestClient(app, raise_server_exceptions=True)


def test_list_returns_sorted_deduped_lowercased_union(client: TestClient) -> None:
    """GET /optimizer returns the union of local + provider types,
    lowercased, sorted, and deduplicated.

    Setup: two providers — one with mixed-case names and one that duplicates a
    local type — so a single call exercises all four properties at once.
    """
    # Provider advertising mixed case (should be lowercased) + a name that
    # duplicates a known local type (should be deduped).
    OPTIMIZER_PROVIDERS.append(_StubProvider(["RemoteX", "AlphaOpt", "halo"]))
    # Second provider advertising names that will sort to extremes.
    OPTIMIZER_PROVIDERS.append(_StubProvider(["zzz-type", "aaa-type"]))

    resp = client.get("/optimizer")
    assert resp.status_code == 200
    types = resp.json()["types"]

    # All local types must appear.
    for local_type in OPTIMIZER_TYPES:
        assert local_type in types

    # Provider names are lowercased.
    assert "remotex" in types
    assert "alphaopt" in types
    assert "RemoteX" not in types
    assert "AlphaOpt" not in types

    # No duplicates (halo appears in both OPTIMIZER_TYPES and the provider).
    assert types.count("halo") == 1

    # Result is sorted.
    assert types == sorted(types)
