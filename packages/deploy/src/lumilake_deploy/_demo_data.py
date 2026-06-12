import json
import os
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import urllib3


@dataclass(frozen=True)
class LumidDataConfig:
    base_url: str
    token: str | None


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


def lumid_config_from_env(env: dict[str, str]) -> LumidDataConfig:
    base = env.get("LUMID_DATA_URL", "").strip().rstrip("/")
    if not base:
        raise SystemExit(
            "LUMID_DATA_URL is required "
            "(fix: set LUMID_DATA_URL=http://host:port in --env-file or pass "
            "--lumid-data-url)"
        )
    token = env.get("LUMID_DATA_TOKEN") or env.get("LUMILAKE_RUNTIME_TOKEN") or None
    return LumidDataConfig(base_url=base, token=token)


def _headers(
    cfg: LumidDataConfig, extra: dict[str, str] | None = None
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if cfg.token:
        headers["Authorization"] = f"Bearer {cfg.token}"
    if extra:
        headers.update(extra)
    return headers


class LumidBlobClient:
    _LIST_PAGE_LIMIT = 10000

    def __init__(self, cfg: LumidDataConfig, *, timeout_seconds: float = 60.0) -> None:
        self._cfg = cfg
        self._http = urllib3.PoolManager()
        self._timeout = timeout_seconds

    def put_blob(self, key: str, body: bytes, content_type: str) -> None:
        url = f"{self._cfg.base_url}/blobs/{_encode_key(key)}"
        resp = self._http.request(
            "PUT",
            url,
            body=body,
            headers=_headers(self._cfg, {"Content-Type": content_type}),
            timeout=urllib3.Timeout(total=self._timeout),
        )
        if resp.status >= 400:
            raise SystemExit(
                f"PUT {url} failed: HTTP {resp.status} {resp.data[:200]!r}"
            )

    def iter_blob_keys(self, prefix: str) -> Iterator[str]:
        url = f"{self._cfg.base_url}/blobs"
        norm = prefix.strip("/")
        resp = self._http.request(
            "GET",
            url,
            fields={"prefix": norm, "limit": str(self._LIST_PAGE_LIMIT)},
            headers=_headers(self._cfg),
            timeout=urllib3.Timeout(total=self._timeout),
        )
        if resp.status >= 400:
            raise SystemExit(
                f"GET {url}?prefix={norm!r} failed: HTTP {resp.status} "
                f"{resp.data[:200]!r}"
            )
        try:
            payload = json.loads(resp.data.decode("utf-8"))
        except ValueError as exc:
            raise SystemExit(f"GET {url} returned non-JSON: {exc}") from exc
        if payload.get("truncated"):
            raise SystemExit(
                f"listing for prefix {norm!r} exceeds the "
                f"{self._LIST_PAGE_LIMIT}-key server cap; narrow the prefix"
            )
        for obj in payload.get("objects", []):
            key = obj.get("key") if isinstance(obj, dict) else None
            if isinstance(key, str) and key and not key.endswith("/"):
                yield key

    def download_blob(self, key: str, dest: Path) -> None:
        url = f"{self._cfg.base_url}/blobs/{_encode_key(key)}"
        resp = self._http.request(
            "GET",
            url,
            headers=_headers(self._cfg),
            preload_content=False,
            timeout=urllib3.Timeout(total=self._timeout),
        )
        try:
            if resp.status >= 400:
                raise SystemExit(
                    f"GET {url} failed: HTTP {resp.status} {resp.read()[:200]!r}"
                )
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.stream(64 * 1024):
                    f.write(chunk)
        finally:
            resp.release_conn()


def _encode_key(key: str) -> str:
    # Keep '/' literal so the path hierarchy survives encoding.
    from urllib.parse import quote

    return quote(key, safe="/")


def find_default_env_file(start: Path | None = None) -> Path | None:
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


def compose_key_prefix(base_prefix: str, sub_prefix: str) -> str:
    """Return the full key prefix by joining ``base_prefix`` and ``sub_prefix``.

    Either segment may be empty; no leading or trailing slash is produced.
    """
    base = base_prefix.strip("/")
    sub = sub_prefix.strip("/")
    return "/".join(seg for seg in (base, sub) if seg)


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
