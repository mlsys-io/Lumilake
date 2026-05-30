"""Configuration loading for the Lumilake SDK.

Supports loading from the shared CLI config file (~/.lumilake/config.toml),
environment variables, or explicit parameters.
"""

import json
import logging
import os
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

from .errors import ConfigInvalidError, ConfigNotFoundError

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path.home() / ".lumilake"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.toml"


@dataclass
class LumilakeConfig:
    """SDK configuration."""

    base_url: str
    api_key: str | None = None

    @classmethod
    def from_file(cls, path: Path = DEFAULT_CONFIG_PATH) -> Self:
        if not path.exists():
            raise ConfigNotFoundError(f"Config file not found at {path}")
        try:
            data = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as exc:
            raise ConfigInvalidError(
                f"Failed to parse config file {path}: {exc}"
            ) from exc
        return cls.from_mapping(data)

    @classmethod
    def from_env(cls) -> Self:
        """Reads ``LUMILAKE_BASE_URL`` and ``LUMILAKE_API_KEY``."""
        base_url = os.getenv("LUMILAKE_BASE_URL", "").strip()
        if not base_url:
            raise ConfigInvalidError("LUMILAKE_BASE_URL environment variable not set")
        return cls(
            base_url=base_url,
            api_key=os.getenv("LUMILAKE_API_KEY", "").strip() or None,
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> Self:
        base_url = data.get("base_url")
        if base_url is None:
            raise ConfigInvalidError("Missing 'base_url' in config")
        if not isinstance(base_url, str):
            raise ConfigInvalidError(
                "Invalid type for 'base_url' in config, expected string"
            )
        base_url = base_url.strip()
        if not base_url:
            raise ConfigInvalidError("Config 'base_url' cannot be empty")

        api_key = data.get("api_key")
        if isinstance(api_key, str):
            api_key = api_key.strip() or None
        elif api_key is not None:
            raise ConfigInvalidError(
                "Invalid type for 'api_key' in config, expected string or None"
            )

        return cls(base_url=base_url, api_key=api_key)

    def to_mapping(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def save(self, path: Path = DEFAULT_CONFIG_PATH) -> None:
        parent = path.parent
        parent_was_owned = parent == DEFAULT_CONFIG_DIR
        parent_existed = parent.exists()
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not parent_existed or parent_was_owned:
            try:
                os.chmod(parent, 0o700)
            except OSError as exc:
                logger.warning("could not chmod config dir %s: %s", parent, exc)

        data = self.to_mapping()
        # json.dumps handles TOML-compatible escaping of " and \ in values.
        body = "\n".join(f"{key} = {json.dumps(value)}" for key, value in data.items())
        if body:
            body += "\n"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, body.encode("utf-8"))
        finally:
            os.close(fd)
        os.chmod(path, 0o600)
        logger.info("saved lumilake config to %s", path)
