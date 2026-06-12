"""Unit tests for _demo_data helpers: lumid_config_from_env, compose_key_prefix."""

import pytest
from lumilake_deploy._demo_data import (
    compose_key_prefix,
    lumid_config_from_env,
)

# ---------------------------------------------------------------------------
# lumid_config_from_env
# ---------------------------------------------------------------------------


def test_lumid_config_explicit_token_wins() -> None:
    cfg = lumid_config_from_env(
        {
            "LUMID_DATA_URL": "http://lumid:5101",
            "LUMID_DATA_TOKEN": "explicit",
            "LUMILAKE_RUNTIME_TOKEN": "fallback",
        }
    )
    assert cfg.base_url == "http://lumid:5101"
    assert cfg.token == "explicit"


def test_lumid_config_falls_back_to_runtime_token() -> None:
    cfg = lumid_config_from_env(
        {
            "LUMID_DATA_URL": "http://lumid:5101/",
            "LUMILAKE_RUNTIME_TOKEN": "fallback",
        }
    )
    # Trailing slash on the base URL is stripped so callers can always join
    # with ``/blobs/...`` without producing a double slash.
    assert cfg.base_url == "http://lumid:5101"
    assert cfg.token == "fallback"


def test_lumid_config_token_optional() -> None:
    """When neither token nor runtime-token is set, cfg.token is None."""
    cfg = lumid_config_from_env({"LUMID_DATA_URL": "http://lumid:5101"})
    assert cfg.token is None


def test_lumid_config_missing_url_raises() -> None:
    with pytest.raises(SystemExit, match="LUMID_DATA_URL"):
        lumid_config_from_env({})


# ---------------------------------------------------------------------------
# compose_key_prefix
# ---------------------------------------------------------------------------


def test_compose_key_prefix_joins_base_and_sub_prefix() -> None:
    assert compose_key_prefix("data/v1", "example-data") == "data/v1/example-data"


def test_compose_key_prefix_empty_base_returns_sub_prefix_only() -> None:
    """Empty base + 'example-data' → 'example-data' (no leading slash)."""
    assert compose_key_prefix("", "example-data") == "example-data"


def test_compose_key_prefix_strips_leading_trailing_slash() -> None:
    """Leading/trailing slashes in either segment are stripped before joining."""
    assert compose_key_prefix("/data/v1/", "/example-data/") == "data/v1/example-data"


def test_compose_key_prefix_both_empty() -> None:
    assert compose_key_prefix("", "") == ""
