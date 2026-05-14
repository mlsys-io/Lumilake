"""Configuration handling for the Lumilake CLI."""

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_DIR = Path.home() / ".lumilake"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.toml"


@dataclass
class LumilakeConfig:
    base_url: str

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "LumilakeConfig":
        if "base_url" not in data:
            raise ValueError("Missing config key: base_url")
        return cls(base_url=data["base_url"])

    def to_mapping(self) -> dict[str, str]:
        return asdict(self)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> LumilakeConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    content = tomllib.loads(path.read_text())
    return LumilakeConfig.from_mapping(content)


def save_config(config: LumilakeConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Persist config to disk, creating the directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.to_mapping()
    lines = [f'{key} = "{value}"' for key, value in data.items()]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
