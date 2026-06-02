"""Unit tests for _demo_data helpers: parse_s3_url and compose_key_prefix."""

from lumilake_deploy._demo_data import compose_key_prefix, parse_s3_url

# ---------------------------------------------------------------------------
# parse_s3_url
# ---------------------------------------------------------------------------


def test_parse_s3_url_populates_bucket_and_base_prefix() -> None:
    """data_prefix='mybucket/data/v1' → bucket='mybucket', base_prefix='data/v1'."""
    cfg = parse_s3_url("s3://access:secret@host:9000", "mybucket/data/v1")
    assert cfg.bucket == "mybucket"
    assert cfg.base_prefix == "data/v1"


def test_parse_s3_url_bucket_only_gives_empty_base_prefix() -> None:
    """data_prefix='mybucket' (no slash) → base_prefix is empty string."""
    cfg = parse_s3_url("s3://access:secret@host:9000", "mybucket")
    assert cfg.bucket == "mybucket"
    assert cfg.base_prefix == ""


def test_parse_s3_url_leading_slash_stripped() -> None:
    """Leading slash in data_prefix is stripped before splitting."""
    cfg = parse_s3_url("s3://access:secret@host:9000", "/mybucket/prefix")
    assert cfg.bucket == "mybucket"
    assert cfg.base_prefix == "prefix"


# ---------------------------------------------------------------------------
# compose_key_prefix
# ---------------------------------------------------------------------------


def test_compose_key_prefix_joins_base_and_s3_prefix() -> None:
    """base_prefix='data/v1' + s3_prefix='example-data' → 'data/v1/example-data'."""
    assert compose_key_prefix("data/v1", "example-data") == "data/v1/example-data"


def test_compose_key_prefix_empty_base_returns_s3_prefix_only() -> None:
    """base_prefix='' + s3_prefix='example-data' → 'example-data' (no leading slash)."""
    assert compose_key_prefix("", "example-data") == "example-data"


def test_compose_key_prefix_strips_leading_trailing_slash_from_s3_prefix() -> None:
    """Leading/trailing slashes in s3_prefix are stripped before joining."""
    assert compose_key_prefix("data/v1", "/example-data/") == "data/v1/example-data"


# ---------------------------------------------------------------------------
# End-to-end composition: parse_s3_url + compose_key_prefix
# ---------------------------------------------------------------------------


def test_full_key_prefix_with_base_prefix_and_s3_prefix() -> None:
    """S3_DATA_PREFIX='mybucket/data/v1' + --s3-prefix='example-data' → correct key."""
    cfg = parse_s3_url("s3://access:secret@host:9000", "mybucket/data/v1")
    full_key_prefix = compose_key_prefix(cfg.base_prefix, "example-data")
    assert full_key_prefix == "data/v1/example-data"


def test_full_key_prefix_bucket_only_data_prefix() -> None:
    """S3_DATA_PREFIX='mybucket' + --s3-prefix='example-data' → 'example-data'."""
    cfg = parse_s3_url("s3://access:secret@host:9000", "mybucket")
    full_key_prefix = compose_key_prefix(cfg.base_prefix, "example-data")
    assert full_key_prefix == "example-data"
