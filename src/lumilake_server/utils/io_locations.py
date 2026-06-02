import json
import os
from dataclasses import dataclass
from io import BytesIO
from urllib.parse import urlparse

from lumilake import envs
from minio import Minio
from minio.error import S3Error

from lumilake_server.schemas.io import S3Location
from lumilake_server.utils.parsing import join_prefix
from lumilake_server.utils.s3 import create_minio_client


def normalize_s3_literal(path: str) -> str:
    """Trim leading slashes and reject empty paths.

    The path is whatever the operator passed; there is no per-user prefix
    scoping (the server is single-tenant).
    """
    cleaned = path.strip()
    if not cleaned:
        raise ValueError("s3 path is required")
    has_trailing_slash = cleaned.endswith("/")
    resolved = cleaned.lstrip("/")
    if has_trailing_slash and not resolved.endswith("/"):
        resolved += "/"
    return resolved


@dataclass(frozen=True)
class S3Connection:
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool
    cert_path: str | None


def _normalize_s3_uri(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return uri
    if not uri:
        raise ValueError("s3 prefix must not be empty")
    if not envs.S3_DATA_PREFIX:
        raise ValueError("S3_DATA_PREFIX is not configured")
    normalized = envs.S3_DATA_PREFIX.strip("/")
    bucket, _, base_prefix = normalized.partition("/")
    if not bucket:
        raise ValueError("S3_DATA_PREFIX must include a bucket (e.g. bucket/prefix)")
    key = join_prefix(base_prefix, uri)
    return f"s3://{bucket}/{key}"


def _build_s3_uri(location: S3Location, obj: str | None = None) -> str:
    target = obj or location.prefix
    if not target:
        raise ValueError("s3 prefix must not be empty")
    if location.connection_string and not target.startswith("s3://"):
        return f"{location.connection_string.rstrip('/')}/{target.lstrip('/')}"
    return _normalize_s3_uri(target)


def _split_bucket_object(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError("s3 uri must start with s3://")
    has_conn = bool(
        parsed.username or parsed.password or parsed.port or "@" in parsed.netloc
    )
    if has_conn:
        path = parsed.path.lstrip("/")
        bucket, _, obj = path.partition("/")
    else:
        bucket = parsed.netloc or ""
        path = parsed.path.lstrip("/")
        if not bucket and path:
            bucket, _, obj = path.partition("/")
        else:
            obj = path
    if not bucket or not obj:
        raise ValueError("s3 uri must include bucket and object path")
    return bucket, obj


def _parse_connection(conn_uri: str) -> S3Connection:
    parsed = urlparse(conn_uri)
    if parsed.scheme not in {"s3", "http", "https"}:
        raise ValueError("invalid s3 connection string")
    endpoint = parsed.hostname or ""
    if parsed.port:
        endpoint = f"{endpoint}:{parsed.port}"
    access_key = parsed.username or ""
    secret_key = parsed.password or ""
    secure = parsed.scheme == "https" or bool(envs.S3_CERT_FILE)
    cert_path = envs.S3_CERT_FILE or None
    if not endpoint or not access_key or not secret_key:
        raise ValueError("s3 connection string must include credentials and host")
    return S3Connection(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
        cert_path=cert_path,
    )


def _s3_client_for_uri(uri: str) -> tuple[Minio, str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        uri = _normalize_s3_uri(uri)
        parsed = urlparse(uri)
    has_conn = bool(
        parsed.username or parsed.password or parsed.port or "@" in parsed.netloc
    )
    if has_conn:
        conn = _parse_connection(uri)
        bucket, obj = _split_bucket_object(uri)
    else:
        if not envs.S3_URL:
            raise ValueError("S3_URL is not configured")
        conn = _parse_connection(envs.S3_URL)
        bucket, obj = _split_bucket_object(uri)
    client = create_minio_client(
        endpoint=conn.endpoint,
        access_key=conn.access_key,
        secret_key=conn.secret_key,
        cert_file=conn.cert_path,
        secure=conn.secure,
    )
    return client, bucket, obj


_DEFAULT_CONTENT_TYPE_BY_SUFFIX = {
    ".json": "application/json",
    ".parquet": "application/vnd.apache.parquet",
    ".npz": "application/octet-stream",
    ".txt": "text/plain",
}


def write_sharded_index(
    location: S3Location,
    shards: dict[str, bytes | str],
    content_types: dict[str, str] | None = None,
) -> None:
    # Non-atomic: partial failure can leave the index half-written.
    # Callers should write the manifest last and use it as the completion signal.
    if not location.prefix:
        raise ValueError("write_sharded_index requires a non-empty S3 prefix")
    resolved_content_types = dict(content_types or {})
    base_uri = _build_s3_uri(location, location.prefix.rstrip("/") + "/")
    for relative_key, body in shards.items():
        data = body.encode("utf-8") if isinstance(body, str) else bytes(body)
        content_type = resolved_content_types.get(
            relative_key
        ) or _DEFAULT_CONTENT_TYPE_BY_SUFFIX.get(
            os.path.splitext(relative_key)[1].lower(), "application/octet-stream"
        )
        shard_uri = base_uri.rstrip("/") + "/" + relative_key.lstrip("/")
        client, bucket, obj = _s3_client_for_uri(shard_uri)
        client.put_object(
            bucket_name=bucket,
            object_name=obj,
            data=BytesIO(data),
            length=len(data),
            content_type=content_type,
        )


def read_s3_bytes(location: S3Location, relative_key: str) -> bytes | None:
    # Returns None on NoSuchKey so callers can distinguish missing from empty.
    base_uri = _build_s3_uri(location, location.prefix.rstrip("/") + "/")
    shard_uri = base_uri.rstrip("/") + "/" + relative_key.lstrip("/")
    client, bucket, obj = _s3_client_for_uri(shard_uri)
    try:
        resp = client.get_object(bucket, obj)
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject"}:
            return None
        raise
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()


def read_s3_json(location: S3Location, relative_key: str) -> dict | list | None:
    raw = read_s3_bytes(location, relative_key)
    if raw is None:
        return None
    return json.loads(raw.decode("utf-8"))


def list_sharded_index(location: S3Location, subprefix: str = "") -> list[str]:
    base_prefix = location.prefix.rstrip("/") + "/"
    full_prefix = base_prefix + subprefix.lstrip("/")
    base_uri = _build_s3_uri(location, full_prefix)
    client, bucket, _obj = _s3_client_for_uri(base_uri)
    _, prefix_in_bucket = _split_bucket_object(base_uri)
    # Strip the connection-string's embedded base so returned keys are
    # relative to location.prefix (usable by read_s3_bytes(rel)).
    subprefix_norm = subprefix.lstrip("/")
    if subprefix_norm:
        base_prefix_in_bucket = prefix_in_bucket[: -len(subprefix_norm)]
    else:
        base_prefix_in_bucket = prefix_in_bucket
    strip_len = max(0, len(base_prefix_in_bucket) - len(base_prefix))
    conn_base = base_prefix_in_bucket[:strip_len]
    results: list[str] = []
    for obj_info in client.list_objects(
        bucket, prefix=prefix_in_bucket, recursive=True
    ):
        name = obj_info.object_name
        if not name:
            continue
        if conn_base and name.startswith(conn_base):
            name = name[len(conn_base) :]
        if name.startswith(base_prefix):
            rel = name[len(base_prefix) :]
        else:
            rel = name
        results.append(rel)
    return results
