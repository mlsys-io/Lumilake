from urllib.parse import quote_plus


def parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def build_database_url(
    *,
    db_type: str | None,
    host: str | None,
    port: str | None,
    name: str | None,
    user: str | None,
    password: str | None,
) -> str | None:
    if not (host and port and name and user and password is not None):
        return None
    resolved_type = (db_type or "postgresql").lower()
    if resolved_type == "postgres":
        resolved_type = "postgresql"
    encoded_user = quote_plus(user)
    encoded_password = quote_plus(password)
    return f"{resolved_type}://{encoded_user}:{encoded_password}@{host}:{port}/{name}"


def split_bucket_prefix(value: str) -> tuple[str, str]:
    cleaned = value.strip().strip("/")
    bucket, _, prefix = cleaned.partition("/")
    if not bucket:
        raise ValueError("s3 prefix must include a bucket")
    return bucket, prefix


def join_prefix(base: str, suffix: str) -> str:
    if not base:
        return suffix.lstrip("/")
    if not suffix:
        return base.rstrip("/")
    return f"{base.rstrip('/')}/{suffix.lstrip('/')}"
