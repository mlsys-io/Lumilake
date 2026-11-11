"""Configuration handling for the Lumilake CLI."""

from pathlib import Path

from dotenv import dotenv_values

from lumilake.config import DEFAULT_CONFIG_DIR, DEFAULT_CONFIG_PATH, LumilakeConfig

__all__ = [
    "DEFAULT_CONFIG_DIR",
    "DEFAULT_CONFIG_PATH",
    "LumilakeConfig",
    "deployment_url",
]


def deployment_url(deploy_dir: Path) -> str:
    """Local server URL a deployment's ``.env`` serves, or the default port."""
    env_path = deploy_dir / ".env"
    port = ""
    if env_path.is_file():
        port = dotenv_values(env_path).get("LUMILAKE_SERVER_PORT") or ""
    return f"http://127.0.0.1:{port or '9000'}"
