"""Top-level LumilakeClient + AsyncLumilakeClient composition tests."""

from pathlib import Path

import httpx
import pytest
import respx
from lumilake import AsyncLumilakeClient, LumilakeClient, LumilakeConfig


def test_sync_client_has_all_resources(client: LumilakeClient) -> None:
    assert client.deploy is not None
    assert client.info is not None
    assert client.jobs is not None
    assert client.workers is not None
    assert client.traces is not None


def test_async_client_has_all_resources(async_client: AsyncLumilakeClient) -> None:
    assert async_client.deploy is not None
    assert async_client.info is not None
    assert async_client.jobs is not None
    assert async_client.workers is not None
    assert async_client.traces is not None


def test_sync_health_passes_through(client: LumilakeClient, base_url: str) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/healthz").mock(
            return_value=httpx.Response(
                200, json={"ok": True, "service": "lumilake-server"}
            )
        )
        result = client.health()
        assert result["ok"] is True


def test_sync_info_status_uses_health_endpoint(
    client: LumilakeClient, base_url: str
) -> None:
    with respx.mock(base_url=base_url) as mocked:
        health = mocked.get("/healthz").mock(
            return_value=httpx.Response(
                200, json={"ok": True, "service": "lumilake-server"}
            )
        )
        result = client.info.status()
        assert result["ok"] is True
        assert health.called


@pytest.mark.asyncio
async def test_async_health_passes_through(
    async_client: AsyncLumilakeClient, base_url: str
) -> None:
    with respx.mock(base_url=base_url) as mocked:
        mocked.get("/healthz").mock(
            return_value=httpx.Response(
                200, json={"ok": True, "service": "lumilake-server"}
            )
        )
        result = await async_client.health()
        assert result["ok"] is True
        await async_client.close()


@pytest.mark.asyncio
async def test_async_info_status_uses_health_endpoint(
    async_client: AsyncLumilakeClient, base_url: str
) -> None:
    with respx.mock(base_url=base_url) as mocked:
        health = mocked.get("/healthz").mock(
            return_value=httpx.Response(
                200, json={"ok": True, "service": "lumilake-server"}
            )
        )
        result = await async_client.info.status()
        assert result["ok"] is True
        assert health.called
        await async_client.close()


def test_sync_from_config_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "config.toml"
    LumilakeConfig(base_url="http://lumilake.test").save(target)
    with LumilakeClient.from_config(target) as c:
        assert c.base_url == "http://lumilake.test"


@pytest.mark.asyncio
async def test_async_from_config_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "config.toml"
    LumilakeConfig(base_url="http://lumilake.test").save(target)
    async with AsyncLumilakeClient.from_config(target) as c:
        assert c.base_url == "http://lumilake.test"


def test_sync_context_manager_closes_client(base_url: str) -> None:
    """Entering / exiting the sync client closes the underlying httpx.Client."""
    with LumilakeClient(base_url=base_url) as client:
        assert client._http is not None
    client.close()


@pytest.mark.asyncio
async def test_async_context_manager_closes_client(base_url: str) -> None:
    async with AsyncLumilakeClient(base_url=base_url) as client:
        assert client._http is not None
    await client.close()
