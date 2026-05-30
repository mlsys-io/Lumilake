"""Shared HTTP transport for the SDK.

Two parallel classes — ``BaseClient`` for sync, ``BaseAsyncClient`` for async —
own the connection lifecycle, version-prefix handling (``/api/v1``), default
headers, and error → exception mapping (404 → ``NotFoundError``, 5xx →
``HttpError``). Resource classes consume them via
``self._client.get/post/...`` and call ``unwrap()`` on the response.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import httpx
from lumilake import envs
from lumilake.config import DEFAULT_CONFIG_PATH, LumilakeConfig
from lumilake.errors import ConfigNotFoundError, HttpError, NotFoundError

logger = logging.getLogger(__name__)

API_VERSION_PREFIX = "/api/v1"
DEFAULT_TIMEOUT = 300.0
DEFAULT_BASE_URL = "http://127.0.0.1:9000"


def _resolve_timeout(timeout: float | None) -> float:
    if timeout is not None and timeout > 0:
        return timeout
    return envs.get_lumilake_timeout(default=DEFAULT_TIMEOUT)


def resolve_config(
    base_url: str | None = None,
    api_key: str | None = None,
    config_path: Path | None = DEFAULT_CONFIG_PATH,
) -> LumilakeConfig:
    """Resolve configuration: params → env → config file → defaults."""
    if base_url is None:
        base_url = envs.get_lumilake_base_url()
    if api_key is None:
        api_key = envs.get_lumilake_api_key()

    if base_url is not None and api_key is not None:
        return LumilakeConfig(base_url=base_url, api_key=api_key)

    try:
        cfg = LumilakeConfig.from_file(config_path or DEFAULT_CONFIG_PATH)
    except ConfigNotFoundError:
        return LumilakeConfig(
            base_url=DEFAULT_BASE_URL if base_url is None else base_url,
            api_key=api_key,
        )

    if base_url is not None:
        cfg.base_url = base_url
    if api_key is not None:
        cfg.api_key = api_key
    return cfg


def _headers(
    extra: Mapping[str, str] | None, *, api_key: str | None = None
) -> dict[str, str]:
    headers: dict[str, str] = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra:
        headers.update(extra)
    return headers


def _url(base_url: str, path: str, *, version_prefix: bool) -> str:
    p = path if path.startswith("/") else f"/{path}"
    if version_prefix and not p.startswith(API_VERSION_PREFIX):
        p = f"{API_VERSION_PREFIX}{p}"
    return f"{base_url.rstrip('/')}{p}"


def _raise_for_status(response: httpx.Response, url: str) -> None:
    if response.status_code == 404:
        raise NotFoundError(response.status_code, response.text, url=url)
    if response.status_code >= 400:
        raise HttpError(response.status_code, response.text, url=url)


def unwrap(response: httpx.Response) -> Any:
    """Unwrap the ``{"ok": ..., "data": ...}`` envelope used by the lumilake API."""
    payload = response.json()
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


class BaseClient:
    """Sync HTTP transport. Wraps ``httpx.Client`` with shared headers,
    ``/api/v1`` prefixing, and error-to-exception mapping. Owns the
    underlying client when constructed without an injected one;
    ``close()`` / ``__exit__`` close it."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float | None = None,
        verify: bool | str = True,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self._http = http_client or httpx.Client(
            timeout=_resolve_timeout(timeout),
            verify=verify,
        )
        self._owns_client = http_client is None

    def request(
        self,
        method: str,
        path: str,
        *,
        version_prefix: bool = True,
        params: Mapping[str, Any] | None = None,
        json_body: Any = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        url = _url(self.base_url, path, version_prefix=version_prefix)
        extra: dict[str, Any] = {}
        if timeout is not None:
            extra["timeout"] = timeout
        try:
            response = self._http.request(
                method,
                url,
                params=dict(params) if params else None,
                json=json_body,
                headers=_headers(headers, api_key=self.api_key),
                **extra,
            )
        except httpx.HTTPError as exc:
            raise HttpError(0, f"network: {exc}", url=url) from exc
        _raise_for_status(response, url)
        return response

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", path, **kwargs)

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class BaseAsyncClient:
    """Async HTTP transport. Wraps ``httpx.AsyncClient`` with the same path,
    headers, and error-mapping rules as ``BaseClient``; resources ``await``
    ``self._client.request(...)`` and call ``unwrap()`` on the response."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float | None = None,
        verify: bool | str = True,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self._http = http_client or httpx.AsyncClient(
            timeout=_resolve_timeout(timeout),
            verify=verify,
        )
        self._owns_client = http_client is None

    async def request(
        self,
        method: str,
        path: str,
        *,
        version_prefix: bool = True,
        params: Mapping[str, Any] | None = None,
        json_body: Any = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        url = _url(self.base_url, path, version_prefix=version_prefix)
        extra: dict[str, Any] = {}
        if timeout is not None:
            extra["timeout"] = timeout
        try:
            response = await self._http.request(
                method,
                url,
                params=dict(params) if params else None,
                json=json_body,
                headers=_headers(headers, api_key=self.api_key),
                **extra,
            )
        except httpx.HTTPError as exc:
            raise HttpError(0, f"network: {exc}", url=url) from exc
        _raise_for_status(response, url)
        return response

    async def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return await self.request("POST", path, **kwargs)

    async def close(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
