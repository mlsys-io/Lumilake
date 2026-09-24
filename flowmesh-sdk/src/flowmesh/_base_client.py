"""Base client implementation with shared HTTP transport logic."""

import asyncio
import json
import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, Self

import httpx

from ._constants import API_VERSION_PREFIX, DEFAULT_BASE_URL, STREAM_TIMEOUT, USER_AGENT
from .config import DEFAULT_CONFIG_PATH, FlowMeshConfig
from .exceptions import (
    APIError,
    AuthenticationError,
    ConfigNotFoundError,
    FlowMeshConnectionError,
    NotFoundError,
    ValidationError,
)


def _build_headers(api_key: str | None) -> dict[str, str]:
    headers: dict[str, str] = {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _make_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + API_VERSION_PREFIX + "/" + path.lstrip("/")


def _raise_for_status(response: httpx.Response, method: str) -> None:
    if response.status_code < 400:
        return
    try:
        body = response.json()
    except Exception:
        body = response.text

    message = ""
    if isinstance(body, dict):
        message = body.get("detail", "") or body.get("message", "")
    if not message:
        message = str(body)

    status = response.status_code
    kwargs: dict[str, Any] = dict(
        status_code=status,
        method=method,
        url=str(response.url),
        body=body,
    )
    if status in (401, 403):
        raise AuthenticationError(message, **kwargs)
    if status == 404:
        raise NotFoundError(message, **kwargs)
    if status in (400, 422):
        raise ValidationError(message, **kwargs)
    raise APIError(message, **kwargs)


def _raise_for_stream_status(response: httpx.Response, method: str) -> None:
    if response.status_code < 400:
        return
    response.read()
    _raise_for_status(response, method)


async def _raise_for_stream_status_async(response: httpx.Response, method: str) -> None:
    if response.status_code < 400:
        return
    await response.aread()
    _raise_for_status(response, method)


class BaseClient:
    """Synchronous HTTP transport for the FlowMesh API."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        timeout: float,
        http_client: httpx.Client | None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._timeout = timeout
        self._headers = _build_headers(api_key)
        self._http = http_client or httpx.Client(
            headers=self._headers,
            timeout=httpx.Timeout(timeout),
        )

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | list[tuple[str, str]] | None = None,
        json_body: Any = None,
        data: str | bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        url = _make_url(self.base_url, path)
        kwargs: dict[str, Any] = {}
        if params:
            kwargs["params"] = params
        if json_body is not None:
            kwargs["json"] = json_body
        if data is not None:
            kwargs["content"] = data
        if headers:
            kwargs["headers"] = headers
        try:
            response = self._http.request(method, url, **kwargs)
        except httpx.ConnectError as exc:
            raise FlowMeshConnectionError(f"Failed to connect to {url}: {exc}")
        _raise_for_status(response, method)
        if not response.content:
            return None
        return response.json()

    def _request_raw(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | list[tuple[str, str]] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        url = _make_url(self.base_url, path)
        kwargs: dict[str, Any] = {}
        if params:
            kwargs["params"] = params
        if headers:
            kwargs["headers"] = headers
        try:
            response = self._http.request(method, url, **kwargs)
        except httpx.ConnectError as exc:
            raise FlowMeshConnectionError(f"Failed to connect to {url}: {exc}")
        _raise_for_status(response, method)
        return response

    def _stream_sse(
        self,
        path: str,
        cursor: str | None = None,
        params: dict[str, str] | None = None,
    ) -> Iterator[tuple[str | None, dict[str, Any]]]:
        """Stream SSE log events, yielding cursor/data pairs."""
        url = _make_url(self.base_url, path)
        headers = self._headers.copy()
        headers["Accept"] = "text/event-stream"
        params = params.copy() if params else {}
        if cursor:
            params["cursor"] = cursor
        try:
            with self._http.stream(
                "GET",
                url,
                params=params,
                headers=headers,
                timeout=httpx.Timeout(STREAM_TIMEOUT, connect=self._timeout),
            ) as response:
                _raise_for_stream_status(response, "GET")
                current_data: list[str] = []
                current_event: str | None = None
                current_id: str | None = None
                for raw_line in response.iter_lines():
                    line = raw_line.strip("\r")
                    if not line:
                        if current_data and (
                            current_event is None or current_event == "log"
                        ):
                            blob = "".join(current_data).strip()
                            if blob:
                                try:
                                    yield current_id, json.loads(blob)
                                except json.JSONDecodeError:
                                    yield current_id, {"message": blob}
                        current_data = []
                        current_event = None
                        current_id = None
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("id:"):
                        current_id = line.split(":", 1)[1].strip() or None
                        continue
                    if line.startswith("event:"):
                        current_event = line.split(":", 1)[1].strip() or None
                        continue
                    if line.startswith("data:"):
                        current_data.append(line.split(":", 1)[1].lstrip() + "\n")
                        continue
                # flush remainder
                if current_data and (current_event is None or current_event == "log"):
                    blob = "".join(current_data).strip()
                    if blob:
                        try:
                            yield current_id, json.loads(blob)
                        except json.JSONDecodeError:
                            yield current_id, {"message": blob}
        except httpx.ConnectError as exc:
            raise FlowMeshConnectionError(f"Failed to connect to {url}: {exc}")

    def _download(
        self,
        path: str,
        output_path: Path,
        chunk_size: int = 256 * 1024,
    ) -> None:
        url = _make_url(self.base_url, path)
        try:
            with self._http.stream("GET", url) as response:
                _raise_for_stream_status(response, "GET")
                with output_path.open("wb") as fh:
                    for chunk in response.iter_bytes(chunk_size=chunk_size):
                        fh.write(chunk)
        except httpx.ConnectError as exc:
            raise FlowMeshConnectionError(f"Failed to connect to {url}: {exc}")

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class BaseAsyncClient:
    """Asynchronous HTTP transport for the FlowMesh API."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        timeout: float,
        http_client: httpx.AsyncClient | None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._timeout = timeout
        self._headers = _build_headers(api_key)
        self._http = http_client or httpx.AsyncClient(
            headers=self._headers,
            timeout=httpx.Timeout(timeout),
        )

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | list[tuple[str, str]] | None = None,
        json_body: Any = None,
        data: str | bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        url = _make_url(self.base_url, path)
        kwargs: dict[str, Any] = {}
        if params:
            kwargs["params"] = params
        if json_body is not None:
            kwargs["json"] = json_body
        if data is not None:
            kwargs["content"] = data
        if headers:
            kwargs["headers"] = headers
        try:
            response = await self._http.request(method, url, **kwargs)
        except httpx.ConnectError as exc:
            raise FlowMeshConnectionError(f"Failed to connect to {url}: {exc}")
        _raise_for_status(response, method)
        if not response.content:
            return None
        return response.json()

    async def _request_raw(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | list[tuple[str, str]] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        url = _make_url(self.base_url, path)
        kwargs: dict[str, Any] = {}
        if params:
            kwargs["params"] = params
        if headers:
            kwargs["headers"] = headers
        try:
            response = await self._http.request(method, url, **kwargs)
        except httpx.ConnectError as exc:
            raise FlowMeshConnectionError(f"Failed to connect to {url}: {exc}")
        _raise_for_status(response, method)
        return response

    async def _stream_sse(
        self,
        path: str,
        cursor: str | None = None,
        params: dict[str, str] | None = None,
    ) -> AsyncIterator[tuple[str | None, dict[str, Any]]]:
        """Stream SSE log events, yielding cursor/data pairs."""
        url = _make_url(self.base_url, path)
        headers = dict(self._headers)
        headers["Accept"] = "text/event-stream"
        params = dict(params) if params else {}
        if cursor:
            params["cursor"] = cursor
        try:
            async with self._http.stream(
                "GET",
                url,
                params=params,
                headers=headers,
                timeout=httpx.Timeout(STREAM_TIMEOUT, connect=self._timeout),
            ) as response:
                await _raise_for_stream_status_async(response, "GET")
                current_data: list[str] = []
                current_event: str | None = None
                current_id: str | None = None
                async for raw_line in response.aiter_lines():
                    line = raw_line.strip("\r")
                    if not line:
                        if current_data and (
                            current_event is None or current_event == "log"
                        ):
                            blob = "".join(current_data).strip()
                            if blob:
                                try:
                                    yield current_id, json.loads(blob)
                                except json.JSONDecodeError:
                                    yield current_id, {"message": blob}
                        current_data = []
                        current_event = None
                        current_id = None
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("id:"):
                        current_id = line.split(":", 1)[1].strip() or None
                        continue
                    if line.startswith("event:"):
                        current_event = line.split(":", 1)[1].strip() or None
                        continue
                    if line.startswith("data:"):
                        current_data.append(line.split(":", 1)[1].lstrip() + "\n")
                        continue
                # flush remainder
                if current_data and (current_event is None or current_event == "log"):
                    blob = "".join(current_data).strip()
                    if blob:
                        try:
                            yield current_id, json.loads(blob)
                        except json.JSONDecodeError:
                            yield current_id, {"message": blob}
        except httpx.ConnectError as exc:
            raise FlowMeshConnectionError(f"Failed to connect to {url}: {exc}")

    async def _download(
        self,
        path: str,
        output_path: Path,
        chunk_size: int = 256 * 1024,
    ) -> None:
        url = _make_url(self.base_url, path)
        try:
            async with self._http.stream("GET", url) as response:
                await _raise_for_stream_status_async(response, "GET")
                fh = await asyncio.to_thread(output_path.open, "wb")
                try:
                    async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                        await asyncio.to_thread(fh.write, chunk)
                finally:
                    await asyncio.to_thread(fh.close)
        except httpx.ConnectError as exc:
            raise FlowMeshConnectionError(f"Failed to connect to {url}: {exc}")

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


def resolve_config(
    base_url: str | None = None,
    api_key: str | None = None,
    config_path: Path | None = DEFAULT_CONFIG_PATH,
) -> FlowMeshConfig:
    """Resolve configuration from params → env → config file → defaults."""
    if base_url is None:
        base_url = os.getenv("FLOWMESH_BASE_URL", "").strip() or None
    if api_key is None:
        api_key = os.getenv("FLOWMESH_API_KEY", "").strip() or None

    if not (base_url is None or api_key is None):
        return FlowMeshConfig(base_url=base_url, api_key=api_key)

    try:
        cfg = FlowMeshConfig.from_file(config_path or DEFAULT_CONFIG_PATH)
    except ConfigNotFoundError:
        return FlowMeshConfig(
            base_url=DEFAULT_BASE_URL if base_url is None else base_url, api_key=api_key
        )

    if base_url is not None:
        cfg.base_url = base_url
    if api_key is not None:
        cfg.api_key = api_key
    return cfg
