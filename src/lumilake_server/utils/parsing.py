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
