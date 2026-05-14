"""``.env``-file helpers for the deploy CLI."""

from pathlib import Path

ENV_FILE_NAME = ".env"
ENV_TEMPLATE_NAME = ".env.example"
FLOWMESH_ENV_FILE_NAME = ".env.flowmesh"


def _quote(value: str) -> str:
    """Shell-quote a value for the env file (simple double-quote wrapping)."""
    escaped = value.replace('"', '\\"')
    return f'"{escaped}"'


def _line(key: str, value: str) -> str:
    return f"{key}={_quote(value)}\n"


def read_env_value(path: Path, key: str) -> str:
    """Read a single env var value from a ``.env``-style file.

    Returns an empty string if the key is not found. Strips surrounding
    double-quotes on the value.
    """
    if not path.is_file():
        return ""
    prefix = f"{key}="
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or not line.startswith(prefix):
            continue
        value = line[len(prefix) :]
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        return value
    return ""


def patch_env_value(path: Path, key: str, value: str) -> None:
    """Replace ``key=...`` in ``path`` if present, append otherwise.

    Used by ``deploy init`` only — never by lifecycle commands.
    Idempotent: writing the same value is a no-op.
    """
    quoted_line = _line(key, value)
    if not path.is_file():
        path.write_text(quoted_line)
        return
    prefix = f"{key}="
    out: list[str] = []
    replaced = False
    for raw in path.read_text().splitlines(keepends=True):
        if raw.lstrip().startswith(prefix):
            out.append(quoted_line)
            replaced = True
        else:
            out.append(raw)
    if not replaced:
        if out and not out[-1].endswith("\n"):
            out[-1] = out[-1] + "\n"
        out.append(quoted_line)
    path.write_text("".join(out))
