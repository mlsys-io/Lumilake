"""Base classes for API resource namespaces.

Each resource module subclasses one of these two: ``SyncResource`` for the
sync class, ``AsyncResource`` for the async class. They hold the
``BaseClient`` / ``BaseAsyncClient`` reference resources call to make HTTP
requests; no other state.
"""

from lumilake.sdk._base_client import BaseAsyncClient, BaseClient


class SyncResource:
    """Base for synchronous API resource namespaces."""

    _client: BaseClient

    def __init__(self, client: BaseClient) -> None:
        self._client = client


class AsyncResource:
    """Base for asynchronous API resource namespaces."""

    _client: BaseAsyncClient

    def __init__(self, client: BaseAsyncClient) -> None:
        self._client = client
