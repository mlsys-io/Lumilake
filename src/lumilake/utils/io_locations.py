import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from urllib.parse import urlparse

import psycopg
from minio import Minio
from minio.error import S3Error
from psycopg import sql

from lumilake import envs
from lumilake.schemas.io import DBLocation, IOLocation, S3Location
from lumilake.utils.parsing import join_prefix
from lumilake.utils.s3 import create_minio_client


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


def normalize_location_key(location: IOLocation) -> str:
    if isinstance(location, DBLocation):
        schema, table = _split_table(location.table)
        return f"db://{schema}.{table}.{location.column}"
    if isinstance(location, S3Location):
        bucket, obj = _split_bucket_object(_normalize_s3_uri(location.prefix))
        return f"s3://{bucket}/{obj}"


def load_input_location(location: IOLocation) -> list[str]:
    if isinstance(location, DBLocation):
        return _read_db_column(location)
    if isinstance(location, S3Location):
        if _is_s3_folder_prefix(location.prefix):
            objects = _list_s3_objects(location)
            return [_build_s3_uri(location, obj) for obj in objects]
        if _is_s3_image_object(location.prefix):
            return [_build_s3_uri(location, location.prefix)]
        return _read_s3_object(location)


def ensure_input_location_exists(location: IOLocation) -> None:
    if isinstance(location, DBLocation):
        _ensure_db_column_exists(location)
        return
    if isinstance(location, S3Location):
        if _is_s3_folder_prefix(location.prefix):
            if not _list_s3_objects(location):
                raise ValueError(
                    f"input s3 folder {location.prefix} is empty or missing"
                )
        else:
            _ensure_s3_object_exists(location)
        return


def ensure_output_location_available(location: IOLocation) -> None:
    if isinstance(location, DBLocation):
        _ensure_db_table_absent(location)
        return
    if isinstance(location, S3Location):
        if _is_s3_folder_prefix(location.prefix):
            if _list_s3_objects(location):
                raise ValueError(f"output s3 folder {location.prefix} already exists")
        else:
            _ensure_s3_object_absent(location)
        return


def write_output_location(
    location: IOLocation,
    values: Iterable[str],
    source_items: Iterable[str] | None = None,
    output_extension: str | None = None,
) -> None:
    if isinstance(location, DBLocation):
        _write_db_column(location, values)
        return
    if isinstance(location, S3Location):
        if _is_s3_folder_prefix(location.prefix):
            _write_s3_folder(location, values, source_items, output_extension)
        else:
            _write_s3_object(location, values)
        return


def _split_table(table: str) -> tuple[str, str]:
    parts = table.split(".", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "public", parts[0]


def _db_conn() -> psycopg.Connection:
    assert envs.DATABASE_URL, "DATABASE_URL is not set"
    return psycopg.connect(envs.DATABASE_URL)


def _ensure_db_column_exists(location: DBLocation) -> None:
    schema, table = _split_table(location.table)
    with _db_conn() as conn:
        cur = conn.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s AND column_name = %s
            """,
            (schema, table, location.column),
        )
        if cur.fetchone() is None:
            raise ValueError("input database column does not exist")


def _ensure_db_table_absent(location: DBLocation) -> None:
    schema, table = _split_table(location.table)
    with _db_conn() as conn:
        cur = conn.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema, table),
        )
        if cur.fetchone() is not None:
            raise ValueError("output table already exists")


def _read_db_column(location: DBLocation) -> list[str]:
    schema, table = _split_table(location.table)
    with _db_conn() as conn:
        query = sql.SQL("SELECT {col} FROM {schema}.{table}").format(
            col=sql.Identifier(location.column),
            schema=sql.Identifier(schema),
            table=sql.Identifier(table),
        )
        rows = conn.execute(query).fetchall()
    return [str(row[0]) for row in rows if row and row[0] is not None]


def _write_db_column(location: DBLocation, values: Iterable[str]) -> None:
    schema, table = _split_table(location.table)
    with _db_conn() as conn:
        conn.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {schema}").format(
                schema=sql.Identifier(schema)
            )
        )
        create_query = sql.SQL("CREATE TABLE {schema}.{table} ({col} TEXT)").format(
            schema=sql.Identifier(schema),
            table=sql.Identifier(table),
            col=sql.Identifier(location.column),
        )
        conn.execute(create_query)
        insert_query = sql.SQL(
            "INSERT INTO {schema}.{table} ({col}) VALUES (%s)"
        ).format(
            schema=sql.Identifier(schema),
            table=sql.Identifier(table),
            col=sql.Identifier(location.column),
        )
        with conn.cursor() as cur:
            cur.executemany(insert_query, [(str(v),) for v in values])
        conn.commit()


def _normalize_s3_uri(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return uri
    if not uri:
        raise ValueError("s3 prefix must not be empty")
    if not envs.S3_URL:
        raise ValueError("S3_URL is not configured")
    bucket, base_prefix = _default_bucket_prefix(envs.S3_URL)
    key = join_prefix(base_prefix, uri)
    return f"s3://{bucket}/{key}"


def _build_s3_uri(location: S3Location, obj: str | None = None) -> str:
    target = obj or location.prefix
    if not target:
        raise ValueError("s3 prefix must not be empty")
    if location.connection_string and not target.startswith("s3://"):
        return f"{location.connection_string.rstrip('/')}/{target.lstrip('/')}"
    return _normalize_s3_uri(target)


def _is_s3_folder_prefix(prefix: str) -> bool:
    return bool(prefix) and prefix.endswith("/")


def _is_s3_image_object(prefix: str) -> bool:
    _, ext = os.path.splitext(prefix.lower())
    return ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}


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


def _default_bucket_prefix(conn_uri: str) -> tuple[str, str]:
    parsed = urlparse(conn_uri)
    if parsed.scheme != "s3":
        raise ValueError("S3_URL must start with s3://")
    has_conn = bool(
        parsed.username or parsed.password or parsed.port or "@" in parsed.netloc
    )
    if has_conn:
        path = parsed.path.lstrip("/")
        bucket, _, prefix = path.partition("/")
    else:
        bucket = parsed.netloc or ""
        prefix = parsed.path.lstrip("/")
    if not bucket:
        raise ValueError("S3_URL must include bucket")
    return bucket, prefix


def _parse_connection(conn_uri: str) -> S3Connection:
    parsed = urlparse(conn_uri)
    if parsed.scheme not in {"s3", "http", "https"}:
        raise ValueError("invalid s3 connection string")
    endpoint = parsed.hostname or ""
    if parsed.port:
        endpoint = f"{endpoint}:{parsed.port}"
    access_key = parsed.username or ""
    secret_key = parsed.password or ""
    secure = parsed.scheme == "https" or bool(envs.S3_CERT_LOCATION)
    cert_path = envs.S3_CERT_LOCATION or None
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


def _ensure_s3_object_exists(location: S3Location) -> None:
    client, bucket, obj = _s3_client_for_uri(_build_s3_uri(location))
    try:
        client.stat_object(bucket, obj)
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject"}:
            raise ValueError("input s3 object not found") from exc
        raise


def _ensure_s3_object_absent(location: S3Location) -> None:
    client, bucket, obj = _s3_client_for_uri(_build_s3_uri(location))
    try:
        client.stat_object(bucket, obj)
        raise ValueError(
            f"output s3 object {location.prefix} already exists at {bucket}/{obj}"
        )
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject"}:
            return
        raise


def _read_s3_object(location: S3Location) -> list[str]:
    client, bucket, obj = _s3_client_for_uri(_build_s3_uri(location))
    resp = client.get_object(bucket, obj)
    try:
        raw = resp.read()
    finally:
        resp.close()
        resp.release_conn()
    return raw.decode("utf-8").splitlines()


def _write_s3_object(location: S3Location, values: Iterable[str]) -> None:
    client, bucket, obj = _s3_client_for_uri(_build_s3_uri(location))
    body = "\n".join(str(v) for v in values).encode("utf-8")
    client.put_object(
        bucket_name=bucket,
        object_name=obj,
        data=BytesIO(body),
        length=len(body),
        content_type="text/plain",
    )


def _write_s3_folder(
    location: S3Location,
    values: Iterable[str],
    source_items: Iterable[str] | None,
    output_extension: str | None,
) -> None:
    if source_items is None:
        raise ValueError("folder outputs require source_items")
    output_ext = output_extension or ".txt"
    if not output_ext.startswith("."):
        output_ext = f".{output_ext}"
    source_list = list(source_items)
    value_list = list(values)
    if len(source_list) != len(value_list):
        raise ValueError("output length must match input folder length")
    client, bucket, obj_prefix = _s3_client_for_uri(_build_s3_uri(location))
    prefix = obj_prefix if obj_prefix.endswith("/") else f"{obj_prefix}/"
    for source, value in zip(source_list, value_list, strict=True):
        stem = _derive_source_stem(source)
        object_name = f"{prefix}{stem}{output_ext}"
        body = str(value).encode("utf-8")
        client.put_object(
            bucket_name=bucket,
            object_name=object_name,
            data=BytesIO(body),
            length=len(body),
            content_type="text/plain",
        )


def _list_s3_objects(location: S3Location) -> list[str]:
    client, bucket, obj_prefix = _s3_client_for_uri(
        _build_s3_uri(location, location.prefix)
    )
    prefix = obj_prefix if obj_prefix.endswith("/") else f"{obj_prefix}/"
    objects = client.list_objects(bucket, prefix=prefix, recursive=True)
    names = [
        obj.object_name
        for obj in objects
        if obj.object_name and not obj.object_name.endswith("/")
    ]
    return sorted(names)


def _derive_source_stem(source: str) -> str:
    parsed = urlparse(source)
    path = parsed.path or source
    base = os.path.basename(path.rstrip("/"))
    if not base:
        base = "item"
    stem, _ = os.path.splitext(base)
    return stem or base


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
