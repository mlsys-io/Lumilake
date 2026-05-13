"""Shared fixtures for the ``lumilake.sdk`` test suite."""

from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio

from lumilake.sdk import (
    AsyncLumilakeClient,
    BaseAsyncClient,
    BaseClient,
    LumilakeClient,
)


@pytest.fixture
def base_url() -> str:
    return "http://lumilake.test"


@pytest.fixture
def http(base_url: str) -> Iterator[BaseClient]:
    transport = BaseClient(base_url=base_url)
    try:
        yield transport
    finally:
        transport.close()


@pytest_asyncio.fixture
async def async_http(base_url: str) -> AsyncIterator[BaseAsyncClient]:
    transport = BaseAsyncClient(base_url=base_url)
    try:
        yield transport
    finally:
        await transport.close()


@pytest.fixture
def client(base_url: str) -> Iterator[LumilakeClient]:
    instance = LumilakeClient(base_url=base_url)
    try:
        yield instance
    finally:
        instance.close()


@pytest_asyncio.fixture
async def async_client(base_url: str) -> AsyncIterator[AsyncLumilakeClient]:
    instance = AsyncLumilakeClient(base_url=base_url)
    try:
        yield instance
    finally:
        await instance.close()
