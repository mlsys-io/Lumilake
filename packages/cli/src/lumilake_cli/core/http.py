"""HTTP client helpers for interacting with Lumilake services."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from lumilake import envs

from .config import DEFAULT_CONFIG_PATH, load_config

API_VERSION_PREFIX = "/api/v1"

DEFAULT_TIMEOUT: float = 300.0
"""Default request timeout in seconds."""

DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:9000"


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


def resolve_base_url(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> tuple[str, str]:
    """Pick a server URL with a clear precedence order.

    Returns ``(base_url, source)`` where ``source`` is one of ``"env"``,
    ``"config"``, or ``"default"``. Resolution order:

    1. ``LUMILAKE_BASE_URL`` environment variable.
    2. ``~/.lumilake/config.toml`` written by ``lumilake deploy up``.
    3. ``http://127.0.0.1:9000`` (the local deploy default).
    """
    env_url = envs.get_lumilake_base_url()
    if env_url:
        return env_url, "env"
    try:
        cfg = load_config(config_path)
    except FileNotFoundError:
        return DEFAULT_LOCAL_BASE_URL, "default"
    except ValueError:
        return DEFAULT_LOCAL_BASE_URL, "default"
    if cfg.base_url:
        return cfg.base_url, "config"
    return DEFAULT_LOCAL_BASE_URL, "default"


def client_from_config() -> HttpClient:
    """Build an HttpClient from the resolved base URL."""
    base_url, _ = resolve_base_url()
    return HttpClient(base_url=base_url)
