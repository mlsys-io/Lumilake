"""Configuration loading for the FlowMesh SDK.

Supports loading from the shared CLI config file (~/.flowmesh/config.toml),
environment variables, or explicit parameters.
"""

import os
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

from .exceptions import ConfigInvalidError, ConfigNotFoundError

DEFAULT_CONFIG_DIR = Path.home() / ".flowmesh"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.toml"


@dataclass
class FlowMeshConfig:
    """SDK configuration."""

    base_url: str
    api_key: str | None = None

    @classmethod
    def from_file(cls, path: Path = DEFAULT_CONFIG_PATH) -> Self:
        """Load configuration from a TOML config file."""
        if not path.exists():
            raise ConfigNotFoundError(f"Config file not found at {path}")
        try:
            data = tomllib.loads(path.read_text())
        except Exception as exc:
            raise ConfigInvalidError(f"Failed to parse config file {path}: {exc}")
        return cls.from_mapping(data)

    @classmethod
    def from_env(cls) -> Self:
        """Load configuration from environment variables.

        Reads ``FLOWMESH_BASE_URL`` and ``FLOWMESH_API_KEY``
        """
        base_url = os.getenv("FLOWMESH_BASE_URL", "").strip()
        if not base_url:
            raise ConfigInvalidError("FLOWMESH_BASE_URL environment variable not set")
        return cls(
            base_url=base_url, api_key=os.getenv("FLOWMESH_API_KEY", "").strip() or None
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> Self:
        """Create from a dict (e.g. parsed TOML)."""
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
            api_key = api_key.strip()
        elif api_key is not None:
            raise ConfigInvalidError(
                "Invalid type for 'api_key' in config, expected string or None"
            )

        return cls(base_url=base_url, api_key=api_key)

    def to_mapping(self) -> dict[str, Any]:
        """Serialize to a dict suitable for TOML output."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def save(self, path: Path = DEFAULT_CONFIG_PATH) -> None:
        """Persist config to disk as TOML, creating the directory if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_mapping()
        lines = [f'{key} = "{value}"' for key, value in data.items()]
        path.write_text("\n".join(lines) + ("\n" if lines else ""))
