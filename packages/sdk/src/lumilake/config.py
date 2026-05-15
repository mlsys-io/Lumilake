"""Saved connection state at ``~/.lumilake/config.toml``.

Holds the server ``base_url`` written by ``lumilake deploy up``.
"""

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".lumilake" / "config.toml"


@dataclass
class LumilakeConfig:
    """Saved server URL. Matches the CLI's TOML schema."""

    base_url: str

    @classmethod
    def load(cls, path: Path | str | None = None) -> "LumilakeConfig":
        """Read ``path`` (default ``~/.lumilake/config.toml``)."""
        target = Path(path) if path else DEFAULT_CONFIG_PATH
        if not target.exists():
            raise FileNotFoundError(
                f"lumilake config not found at {target}. Run "
                f"`lumilake deploy up` to create it, or set LUMILAKE_BASE_URL."
            )
        with open(target, "rb") as f:
            data = tomllib.load(f)
        return cls(base_url=data["base_url"])

    def save(self, path: Path | str | None = None) -> None:
        target = Path(path) if path else DEFAULT_CONFIG_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        body = f'base_url = "{self.base_url}"\n'
        target.write_text(body, encoding="utf-8")
        logger.info("saved lumilake config to %s", target)
