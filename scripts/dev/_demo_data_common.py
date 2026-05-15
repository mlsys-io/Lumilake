import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


@dataclass(frozen=True)
class S3Config:
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    secure: bool
    cert_file: str | None


def load_env_file(env_file: Path | None) -> dict[str, str]:
    if env_file is None:
        return {}
    if not env_file.is_file():
        raise FileNotFoundError(f"env file not found: {env_file}")
    pat = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
    out: dict[str, str] = {}
    for raw in env_file.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = pat.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        out[key] = value
    return out


def resolve_env(
    env_file: Path | None,
    overrides: dict[str, str | None] | None = None,
) -> dict[str, str]:
    merged: dict[str, str] = {}
    merged.update(load_env_file(env_file))
    for key, value in os.environ.items():
        if value:
            merged[key] = value
    if overrides:
        for key, override in overrides.items():
            if override is not None:
                merged[key] = override
    return merged


def require_env(env: dict[str, str], keys: Iterable[str]) -> None:
    missing = [k for k in keys if not env.get(k)]
    if missing:
        raise SystemExit(
            f"missing required env variable(s): {', '.join(missing)} "
            "(set in --env-file or process env)"
        )


def parse_s3_url(raw: str, cert_file: str | None = None) -> S3Config:
    parsed = urlparse(raw)
    if parsed.scheme != "s3":
        raise SystemExit(f"S3_URL must use s3:// scheme: {raw!r}")
    if not parsed.hostname or not parsed.username or not parsed.password:
        raise SystemExit(
            "S3_URL must include credentials and host, e.g. "
            "s3://access:secret@host:port/bucket"
        )
    bucket = parsed.path.lstrip("/").split("/", 1)[0]
    if not bucket:
        raise SystemExit("S3_URL must include a bucket in the path")
    endpoint = parsed.hostname
    if parsed.port:
        endpoint = f"{endpoint}:{parsed.port}"
    return S3Config(
        endpoint=endpoint,
        access_key=unquote(parsed.username),
        secret_key=unquote(parsed.password),
        bucket=bucket,
        secure=bool(cert_file),
        cert_file=cert_file,
    )


def make_minio_client(cfg: S3Config) -> Any:
    try:
        from minio import Minio
    except ImportError as exc:
        raise SystemExit(
            "minio package is required. Install via "
            "`uv sync --all-packages` or `pip install minio`."
        ) from exc

    http_client = None
    if cfg.cert_file:
        import certifi
        import urllib3

        http_client = urllib3.PoolManager(
            cert_reqs="CERT_REQUIRED",
            ca_certs=cfg.cert_file or certifi.where(),
        )
    return Minio(
        endpoint=cfg.endpoint,
        access_key=cfg.access_key,
        secret_key=cfg.secret_key,
        secure=cfg.secure,
        http_client=http_client,
    )


def find_default_env_file(start: Path | None = None) -> Path | None:
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


def info(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    val = float(n)
    for unit in units:
        if val < 1024.0:
            return f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{val:.1f} PiB"
