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
        except Exception as exc:
            raise ConfigInvalidError(f"Failed to parse config file {path}: {exc}")
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
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_mapping()
        # json.dumps handles TOML-compatible escaping of " and \ in values.
        lines = [f"{key} = {json.dumps(value)}" for key, value in data.items()]
        path.write_text("\n".join(lines) + ("\n" if lines else ""))
        logger.info("saved lumilake config to %s", path)
