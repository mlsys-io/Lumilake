_FORBIDDEN_CHARS = ("?", "#", "\x00")


def normalize_s3_literal(path: str) -> str:
    """Trim leading slashes and reject empty or unsafe paths.

    The path is whatever the operator passed; there is no per-user prefix
    scoping (the server is single-tenant). Reject path-traversal and
    URL-control characters so a caller cannot escape ``S3_DATA_PREFIX``
    when the value is concatenated into a blob key.
    """
    cleaned = path.strip()
    if not cleaned:
        raise ValueError("s3 path is required")
    for ch in _FORBIDDEN_CHARS:
        if ch in cleaned:
            raise ValueError(f"s3 path contains forbidden character: {ch!r}")
    has_trailing_slash = cleaned.endswith("/")
    resolved = cleaned.lstrip("/")
    if has_trailing_slash and not resolved.endswith("/"):
        resolved += "/"
    inner = resolved[:-1] if resolved.endswith("/") else resolved
    for segment in inner.split("/"):
        if segment == "" or segment == "." or segment == "..":
            raise ValueError(
                f"s3 path contains empty or traversal segment: {segment!r}"
            )
    return resolved
