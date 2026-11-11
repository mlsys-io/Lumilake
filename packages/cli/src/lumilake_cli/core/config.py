"""Configuration handling for the Lumilake CLI."""

from pathlib import Path

from lumilake.config import DEFAULT_CONFIG_DIR, DEFAULT_CONFIG_PATH, LumilakeConfig

__all__ = [
    "DEFAULT_CONFIG_DIR",
    "DEFAULT_CONFIG_PATH",
    "LumilakeConfig",
    "deployment_url",
]


def _env_file_value(path: Path, key: str) -> str:
    """Read one ``KEY=value`` line from a ``.env`` file (quotes stripped)."""
    if not path.is_file():
        return ""
    prefix = f"{key}="
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line.startswith(prefix):
            continue
        value = line[len(prefix) :]
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        return value
    return ""


def deployment_url(deploy_dir: Path) -> str:
    """Local server URL a deployment's ``.env`` serves, or the default port."""
    port = _env_file_value(deploy_dir / ".env", "LUMILAKE_SERVER_PORT") or "9000"
    return f"http://127.0.0.1:{port}"
