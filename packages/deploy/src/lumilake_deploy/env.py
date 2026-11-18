"""``.env``-file helpers for the deploy CLI."""

from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import dotenv_values

ENV_FILE_NAME = ".env"
ENV_TEMPLATE_NAME = ".env.example"
FLOWMESH_ENV_FILE_NAME = ".env.flowmesh"


def _quote(value: str) -> str:
    """Shell-quote a value for the env file (simple double-quote wrapping)."""
    escaped = value.replace('"', '\\"')
    return f'"{escaped}"'


def _line(key: str, value: str) -> str:
    return f"{key}={_quote(value)}\n"


def read_env_file(path: Path) -> dict[str, str]:
    """Parse a ``.env``-style file the way docker compose reads it.

    Inline ``# comment`` is not part of a value, quotes are stripped, and a
    missing file reads as empty.
    """
    if not path.is_file():
        return {}
    return {key: value or "" for key, value in dotenv_values(path).items()}


def read_env_value(path: Path, key: str) -> str:
    """Read a single env var value from a ``.env``-style file.

    Returns an empty string if the key is not found. Parsing matches what
    docker compose reads from the same file, so an inline ``# comment``
    is not part of the value and quoting is stripped.
    """
    return read_env_file(path).get(key, "")


def patch_env_value(path: Path, key: str, value: str, *, quote: bool = True) -> None:
    """Replace ``key=...`` in ``path`` if present, append otherwise. Pass
    ``quote=False`` for ``.env.flowmesh``, whose values FlowMesh's stack
    loader hands to compose verbatim; writing the same value is a no-op.
    """
    new_line = _line(key, value) if quote else f"{key}={value}\n"
    if not path.is_file():
        path.write_text(new_line)
        return
    prefix = f"{key}="
    out: list[str] = []
    replaced = False
    for raw in path.read_text().splitlines(keepends=True):
        if raw.lstrip().startswith(prefix):
            out.append(new_line)
            replaced = True
        else:
            out.append(raw)
    if not replaced:
        if out and not out[-1].endswith("\n"):
            out[-1] = out[-1] + "\n"
        out.append(new_line)
    path.write_text("".join(out))


# Hosts that reach this host's own FlowMesh server from a host-network
# container. Anything else is a deliberate remote endpoint.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

ORCHESTRATOR_URL_KEY = "LUMILAKE_RUNTIME_ORCHESTRATOR_URL"
FLOWMESH_BASE_URL_KEY = "FLOWMESH_BASE_URL"


def _local_url(value: str, host: str, port: int) -> str | None:
    """``value`` on ``port``, or None to keep it as it is."""
    if not value:
        return f"http://{host}:{port}"
    parts = urlsplit(value)
    if parts.hostname not in _LOOPBACK_HOSTS:
        return None
    netloc = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    return urlunsplit(parts._replace(netloc=f"{netloc}:{port}"))


def sync_orchestrator_urls(env_path: Path, flowmesh_env_path: Path) -> list[str]:
    """Point a missing or loopback FlowMesh URL at ``SERVER_HTTP_PORT``,
    keep one naming another host, and return the rewritten ``KEY=value``
    pairs. Does nothing when ``.env.flowmesh`` sets no port."""
    raw_port = read_env_value(flowmesh_env_path, "SERVER_HTTP_PORT")
    if not raw_port:
        return []
    port = int(raw_port)
    changed: list[str] = []
    for path, key, host in (
        (env_path, ORCHESTRATOR_URL_KEY, "127.0.0.1"),
        (flowmesh_env_path, FLOWMESH_BASE_URL_KEY, "localhost"),
    ):
        url = _local_url(read_env_value(path, key), host, port)
        if url is None:
            continue
        bare = path == flowmesh_env_path
        line = f"{key}={url}" if bare else _line(key, url).rstrip("\n")
        if line in path.read_text().splitlines():
            continue
        patch_env_value(path, key, url, quote=not bare)
        changed.append(f"{key}={url}")
    return changed
