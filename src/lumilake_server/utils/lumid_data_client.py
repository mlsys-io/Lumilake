"""Thin client for lumid-data-app HTTP endpoints used by data profiling and archive I/O.

Covers:
  - ``profile(sql, plans)``            — sync, for use in asyncio.to_thread
  - ``list_blobs(prefix)``             — async
  - ``retrieve_sample(sql)``           — sync, for use in asyncio.to_thread
  - ``put_blob(key, body, ct)``        — sync, upload a blob
  - ``get_blob(key)``                  — sync, download a blob
  - ``alist_blob_keys(prefix, ...)``   — async, list all blob keys under a prefix
  - ``acatalog_column_exists(...)``    — async, check catalog column existence

All calls read ``LUMID_DATA_URL`` and ``LUMID_DATA_TOKEN`` from ``envs``
and set ``Authorization: Bearer <token>``.  Timeout is taken from
``LUMID_DATA_TIMEOUT_SECONDS``.
"""

import json
import logging
import re
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

import requests
from lumilake import envs
from lumilake.log import trace_id_var

from lumilake_server.utils.http_client import aget_json, arequest, get, post_json

logger = logging.getLogger(__name__)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_sql_identifier(name: str, kind: str) -> None:
    """Reject identifiers that wouldn't fit a SQL ``[A-Za-z_][A-Za-z0-9_]*``."""
    if not _IDENTIFIER_RE.fullmatch(name):
        raise ValueError(f"invalid {kind} identifier: {name!r}")


def _encode_blob_key(key: str) -> str:
    """URL-encode a blob key, preserving ``/`` separators."""
    return quote(key.lstrip("/"), safe="/")


class BlobNotFound(Exception):
    """Raised when a blob key is not found in lumid-data-app."""


def _base_url() -> str:
    url = envs.LUMID_DATA_URL
    if not url:
        raise RuntimeError(
            "LUMID_DATA_URL is required for data profiling via lumid-data-app "
            "(see .env.example)"
        )
    return url.rstrip("/")


def _auth_headers() -> dict[str, str]:
    token = envs.LUMID_DATA_TOKEN
    if not token:
        raise RuntimeError(
            "LUMID_DATA_TOKEN is required for data profiling via lumid-data-app "
            "(see .env.example)"
        )
    return {"Authorization": f"Bearer {token}"}


def _default_headers() -> dict[str, str]:
    headers = _auth_headers()
    trace_id = trace_id_var.get()
    if trace_id:
        headers["X-Request-ID"] = trace_id
    return headers


def profile(
    sql: str,
    plans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """POST /profile and return the ``variants`` list.

    Each returned dict has keys ``plan_id``, ``raw_cost``,
    ``estimated_rows``, ``footprints``, and ``explain_json`` —
    matching ``DataProfileCostEstimate`` field names.
    """
    body: dict[str, Any] = {"sql": sql}
    if plans:
        body["plans"] = plans
    url = f"{_base_url()}/profile"
    body_bytes = len(json.dumps(body).encode("utf-8"))
    logger.info("POST %s body_bytes=%d", url, body_bytes)
    start = time.monotonic()
    result = post_json(
        url,
        json=body,
        headers=_default_headers(),
        timeout=envs.LUMID_DATA_TIMEOUT_SECONDS,
    )
    logger.debug("POST %s elapsed=%.3fs", url, time.monotonic() - start)
    if not isinstance(result, dict):
        raise RuntimeError(
            f"lumid-data-app /profile returned unexpected shape: {type(result)}"
        )
    variants = result.get("variants")
    if not isinstance(variants, list):
        raise RuntimeError(
            f"lumid-data-app /profile response missing 'variants' list: {result!r}"
        )
    return variants


async def list_blobs(prefix: str) -> tuple[dict[str, int | None], list[str]]:
    """GET /blobs?prefix=<prefix> and return ``(file_sizes, folder_paths)``.

    ``file_sizes`` maps relative paths under the prefix to byte sizes.
    ``folder_paths`` lists the unique folder prefixes implied by those paths.
    """
    url = f"{_base_url()}/blobs"
    logger.info("GET %s prefix=%r", url, prefix)
    start = time.monotonic()
    result = await aget_json(
        url,
        params={"prefix": prefix, "limit": "10000"},
        headers=_default_headers(),
        timeout=envs.LUMID_DATA_TIMEOUT_SECONDS,
    )
    logger.debug("GET %s elapsed=%.3fs", url, time.monotonic() - start)
    if not isinstance(result, dict):
        raise RuntimeError(
            f"lumid-data-app /blobs returned unexpected shape: {type(result)}"
        )
    objects = result.get("objects")
    if not isinstance(objects, list):
        raise RuntimeError(
            f"lumid-data-app /blobs response missing 'objects' list: {result!r}"
        )
    if result.get("truncated"):
        raise RuntimeError(
            f"lumid-data-app /blobs listing for prefix {prefix!r} exceeds the "
            "10000-key server cap; cannot enumerate this prefix"
        )
    sizes: dict[str, int | None] = {}
    folders: set[str] = set()
    prefix_norm = prefix.rstrip("/")
    strip_len = len(prefix_norm) + 1 if prefix_norm else 0
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        key = obj.get("key")
        if not isinstance(key, str) or not key:
            continue
        if prefix_norm:
            if not key.startswith(prefix_norm + "/"):
                continue
            rel = key[strip_len:]
        else:
            rel = key
        if not rel:
            continue
        raw_size = obj.get("size")
        sizes[rel] = int(raw_size) if isinstance(raw_size, (int, float)) else None
        parts = rel.split("/")
        for idx in range(1, len(parts)):
            folders.add(f"{'/'.join(parts[:idx])}/")
    return sizes, sorted(folders)


def put_blob(key: str, body: bytes, content_type: str) -> None:
    """PUT /blobs/<key> — upload ``body`` as ``content_type``.

    Raises ``RuntimeError`` on HTTP 413 (quota exceeded).
    """
    url = f"{_base_url()}/blobs/{_encode_blob_key(key)}"
    logger.info("PUT %s body_bytes=%d", url, len(body))
    start = time.monotonic()
    resp = requests.put(
        url,
        data=body,
        headers={**_default_headers(), "Content-Type": content_type},
        timeout=envs.LUMID_DATA_TIMEOUT_SECONDS,
        allow_redirects=False,
    )
    logger.debug(
        "PUT %s status=%d elapsed=%.3fs",
        url,
        resp.status_code,
        time.monotonic() - start,
    )
    if resp.status_code == 413:
        raise RuntimeError(
            f"lumid-data-app blob upload rejected (413): key={key!r} "
            f"exceeds blob_max_bytes quota"
        )
    resp.raise_for_status()


def get_blob(key: str) -> tuple[bytes, str]:
    """GET /blobs/<key> — return ``(body, content_type)``.

    Raises ``BlobNotFound`` when the server returns 404.
    """
    url = f"{_base_url()}/blobs/{_encode_blob_key(key)}"
    logger.info("GET %s", url)
    start = time.monotonic()
    resp = requests.get(
        url,
        headers=_default_headers(),
        timeout=envs.LUMID_DATA_TIMEOUT_SECONDS,
        allow_redirects=False,
    )
    logger.debug(
        "GET %s status=%d elapsed=%.3fs",
        url,
        resp.status_code,
        time.monotonic() - start,
    )
    if resp.status_code == 404:
        raise BlobNotFound(key)
    resp.raise_for_status()
    ct = resp.headers.get("Content-Type", "application/octet-stream")
    return resp.content, ct


async def alist_blob_keys(prefix: str, *, recursive: bool) -> list[str]:
    """Return absolute blob keys under ``prefix``.

    ``recursive=True`` lists all keys (no delimiter); ``recursive=False`` uses
    delimiter ``"/"`` for a single level. The ``/blobs`` endpoint has no cursor,
    so this requests the server's max page and raises if the listing is still
    truncated rather than silently dropping keys.
    """
    delimiter = "" if recursive else "/"
    url = f"{_base_url()}/blobs"
    logger.info("GET %s prefix=%r recursive=%s", url, prefix, recursive)
    start = time.monotonic()
    result = await aget_json(
        url,
        params={"prefix": prefix, "delimiter": delimiter, "limit": "10000"},
        headers=_default_headers(),
        timeout=envs.LUMID_DATA_TIMEOUT_SECONDS,
    )
    logger.debug("GET %s elapsed=%.3fs", url, time.monotonic() - start)
    if not isinstance(result, dict):
        raise RuntimeError(
            f"lumid-data-app /blobs returned unexpected shape: {type(result)}"
        )
    objects = result.get("objects")
    if not isinstance(objects, list):
        raise RuntimeError(
            f"lumid-data-app /blobs response missing 'objects' list: {result!r}"
        )
    if result.get("truncated"):
        raise RuntimeError(
            f"lumid-data-app /blobs listing for prefix {prefix!r} exceeds the "
            "10000-key server cap; cannot enumerate this prefix"
        )
    keys: list[str] = []
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        k = obj.get("key")
        if isinstance(k, str) and k:
            keys.append(k)
    return keys


async def acatalog_column_exists(schema: str, table: str, column: str) -> bool:
    """Return ``True`` if ``column`` exists on ``schema.table`` in the catalog.

    Calls ``GET /catalog/tables/{schema}/{table}``.
    Returns ``False`` on 404 (table not found).
    Raises on other HTTP errors.
    """
    _validate_sql_identifier(schema, "schema")
    _validate_sql_identifier(table, "table")
    url = (
        f"{_base_url()}/catalog/tables/"
        f"{quote(schema, safe='')}/{quote(table, safe='')}"
    )
    logger.info("GET %s", url)
    start = time.monotonic()
    resp = await arequest(
        "GET",
        url,
        headers=_default_headers(),
        timeout=envs.LUMID_DATA_TIMEOUT_SECONDS,
    )
    logger.debug(
        "GET %s status=%d elapsed=%.3fs",
        url,
        resp.status,
        time.monotonic() - start,
    )
    if resp.status == 404:
        return False
    resp.raise_for_status()
    result = await resp.json()
    if not isinstance(result, dict):
        return False
    columns = result.get("columns")
    if not isinstance(columns, list):
        return False
    return any(
        isinstance(col, Mapping) and col.get("name") == column for col in columns
    )


def retrieve_sample(sql: str) -> list[dict[str, Any]]:
    """POST /retrieve and fetch the materialized JSONL rows.

    Posts ``sql`` (which already includes any LIMIT clause) to
    ``/retrieve``, fetches the ``materialized_uri`` from the response,
    and parses the newline-delimited JSON rows.
    """
    body = {"sql": sql}
    url = f"{_base_url()}/retrieve"
    body_bytes = len(json.dumps(body).encode("utf-8"))
    logger.info("POST %s body_bytes=%d", url, body_bytes)
    start = time.monotonic()
    result = post_json(
        url,
        json=body,
        headers=_default_headers(),
        timeout=envs.LUMID_DATA_TIMEOUT_SECONDS,
    )
    logger.debug("POST %s elapsed=%.3fs", url, time.monotonic() - start)
    if not isinstance(result, dict):
        raise RuntimeError(
            f"lumid-data-app /retrieve returned unexpected shape: {type(result)}"
        )
    materialized_uri = result.get("materialized_uri")
    if not isinstance(materialized_uri, str) or not materialized_uri:
        raise RuntimeError(
            "lumid-data-app /retrieve response missing 'materialized_uri': "
            f"{result!r}"
        )
    # Reject absolute URLs — a compromised upstream could redirect the
    # bearer-attached fetch to an attacker-controlled host.
    if not materialized_uri.startswith("/"):
        raise RuntimeError(
            "lumid-data-app /retrieve materialized_uri must be an "
            f"app-relative path starting with '/'; got {materialized_uri!r}"
        )
    fetch_url = f"{_base_url()}{materialized_uri}"
    logger.info("GET %s", fetch_url)
    fetch_start = time.monotonic()
    fetch_resp = get(
        fetch_url,
        headers=_default_headers(),
        timeout=envs.LUMID_DATA_TIMEOUT_SECONDS,
    )
    logger.debug(
        "GET %s status=%d elapsed=%.3fs",
        fetch_url,
        fetch_resp.status_code,
        time.monotonic() - fetch_start,
    )
    fetch_resp.raise_for_status()
    rows: list[dict[str, Any]] = []
    for raw_line in fetch_resp.text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows
