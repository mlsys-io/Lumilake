"""Packaged deploy assets (compose file, env template).

Resolved through :mod:`importlib.resources` so the deploy CLI works the
same from a workspace checkout or a PyPI install — operators don't have
to be in (or even have) a Lumilake source tree.
"""

from importlib import resources
from pathlib import Path


class AssetNotFoundError(FileNotFoundError):
    """Raised when a packaged asset cannot be resolved."""


def asset_path(*parts: str) -> Path:
    """Return a usable filesystem path for an asset inside this package."""
    resource = resources.files(__name__)
    for part in parts:
        resource /= part
    try:
        with resources.as_file(resource) as path:
            return Path(path)
    except FileNotFoundError as exc:
        raise AssetNotFoundError(str(resource)) from exc


def env_example_path() -> Path:
    """Path to the bundled ``.env.example`` template."""
    return asset_path(".env.example")


def compose_path() -> Path:
    """Path to the bundled ``compose.yml`` (docker-compose) file."""
    return asset_path("compose.yml")
