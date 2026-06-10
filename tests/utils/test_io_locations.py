import pytest

from lumilake_server.utils.io_locations import normalize_s3_literal


def test_normalize_s3_literal_strips_leading_slashes() -> None:
    assert normalize_s3_literal("/foo/bar") == "foo/bar"


def test_normalize_s3_literal_preserves_trailing_slash() -> None:
    assert normalize_s3_literal("/foo/bar/") == "foo/bar/"


def test_normalize_s3_literal_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="s3 path is required"):
        normalize_s3_literal("")


def test_normalize_s3_literal_raises_on_whitespace_only() -> None:
    with pytest.raises(ValueError, match="s3 path is required"):
        normalize_s3_literal("   ")


def test_normalize_s3_literal_no_leading_slash() -> None:
    assert normalize_s3_literal("bucket/prefix") == "bucket/prefix"


def test_normalize_s3_literal_adds_trailing_slash_when_input_ends_with_slash() -> None:
    assert normalize_s3_literal("/foo/").endswith("/")


def test_normalize_s3_literal_rejects_parent_traversal() -> None:
    with pytest.raises(ValueError, match="traversal"):
        normalize_s3_literal("data/../private/secret.txt")


def test_normalize_s3_literal_rejects_dot_segment() -> None:
    with pytest.raises(ValueError, match="traversal"):
        normalize_s3_literal("data/./private")


def test_normalize_s3_literal_rejects_empty_inner_segment() -> None:
    with pytest.raises(ValueError, match="empty"):
        normalize_s3_literal("foo//bar")


def test_normalize_s3_literal_rejects_question_mark() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        normalize_s3_literal("foo/bar?extra")


def test_normalize_s3_literal_rejects_fragment() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        normalize_s3_literal("foo/bar#frag")


def test_normalize_s3_literal_rejects_nul_byte() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        normalize_s3_literal("foo/bar\x00baz")


def test_normalize_s3_literal_rejects_leading_dotdot() -> None:
    with pytest.raises(ValueError, match="traversal"):
        normalize_s3_literal("../etc/passwd")
