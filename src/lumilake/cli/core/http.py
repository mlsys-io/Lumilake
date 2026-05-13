"""HTTP client helpers for interacting with Lumilake services."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import requests
import typer

from lumilake import envs

from . import logging
from .config import DEFAULT_CONFIG_PATH, LumilakeConfig, load_config

API_VERSION_PREFIX = "/api/v1"

DEFAULT_TIMEOUT: float = 300.0
"""Default request timeout in seconds."""


def _resolve_timeout() -> float:
    """Read timeout from ``LUMILAKE_TIMEOUT`` env var, falling back to default."""
    return envs.get_lumilake_timeout(default=DEFAULT_TIMEOUT)


class HttpError(RuntimeError):
    """Raised when an HTTP request fails."""


@dataclass
class HttpClient:
    base_url: str
    timeout: float = field(default_factory=_resolve_timeout)

    def _headers(self) -> Mapping[str, str]:
        return {"Accept": "application/json"}

    def get(
        self, path: str, version_prefix: bool = False, **kwargs: Any
    ) -> requests.Response:
        return self._request("GET", path, version_prefix, **kwargs)

    def post(
        self, path: str, version_prefix: bool = False, **kwargs: Any
    ) -> requests.Response:
        return self._request("POST", path, version_prefix, **kwargs)

    def put(
        self, path: str, version_prefix: bool = False, **kwargs: Any
    ) -> requests.Response:
        return self._request("PUT", path, version_prefix, **kwargs)

    def patch(
        self, path: str, version_prefix: bool = False, **kwargs: Any
    ) -> requests.Response:
        return self._request("PATCH", path, version_prefix, **kwargs)

    def delete(
        self, path: str, version_prefix: bool = False, **kwargs: Any
    ) -> requests.Response:
        return self._request("DELETE", path, version_prefix, **kwargs)

    def download(
        self,
        path: str,
        output_path: Path,
        version_prefix: bool = False,
        chunk_size: int = 1024 * 256,
        **kwargs: Any,
    ) -> None:
        url = self._make_url(path, version_prefix)
        headers = dict(self._headers())
        extra_headers = kwargs.pop("headers", None)
        if extra_headers:
            headers.update(extra_headers)
        timeout = kwargs.pop("timeout", self.timeout)
        try:
            response = requests.get(
                url, headers=headers, stream=True, timeout=timeout, **kwargs
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise HttpError(f"Failed to download {url}: {exc}") from exc

        with output_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                fh.write(chunk)

    def _make_url(self, path: str, version_prefix: bool) -> str:
        url = self.base_url.rstrip("/")
        if version_prefix:
            url += API_VERSION_PREFIX
        url += "/" + path.lstrip("/")
        return url

    def _request(
        self, method: str, path: str, version_prefix: bool, **kwargs: Any
    ) -> requests.Response:
        url = self._make_url(path, version_prefix)
        headers = dict(self._headers())
        extra_headers = kwargs.pop("headers", None)
        if extra_headers:
            headers.update(extra_headers)
        kwargs.setdefault("timeout", self.timeout)
        try:
            response = requests.request(method, url, headers=headers, **kwargs)
        except requests.RequestException as exc:
            raise HttpError(f"{method} {url} failed: {exc}") from exc
        if response.status_code >= 400:
            raise HttpError(
                f"{method} {url} failed: {response.status_code} {response.text}"
            )
        return response


def _require_config(path: Path = DEFAULT_CONFIG_PATH) -> LumilakeConfig:
    """Load config or exit with a clear message directing the user to login."""

    def _error(msg: str) -> NoReturn:
        logging.error(msg)
        raise typer.Exit(code=1)

    try:
        config = load_config(path)
    except FileNotFoundError:
        _error("Not logged in. Run `lumilake login <url>` first.")
    except ValueError as exc:
        _error(f"Invalid config file: {exc}. Please re-login.")

    if not config.base_url:
        _error("Missing base_url in config. Please re-login.")
    return config


def client_from_config(config: LumilakeConfig | None = None) -> HttpClient:
    """Build an HttpClient from saved config. Exits if not logged in."""
    if config is None:
        config = _require_config()
    return HttpClient(base_url=config.base_url)
