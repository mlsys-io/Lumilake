"""Base classes for API resource namespaces."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._base_client import BaseAsyncClient, BaseClient


class SyncResource:
    """Base for synchronous API resource namespaces."""

    _client: "BaseClient"

    def __init__(self, client: "BaseClient") -> None:
        self._client = client


class AsyncResource:
    """Base for asynchronous API resource namespaces."""

    _client: "BaseAsyncClient"

    def __init__(self, client: "BaseAsyncClient") -> None:
        self._client = client
