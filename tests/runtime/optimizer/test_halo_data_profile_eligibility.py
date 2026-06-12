"""Eligibility check accepts both retrieval shapes: bare ``type=sql|s3``
(profile-graph nodes) and ``type=lumid, mode=sql|s3`` (runtime ops). If it
misses either, /profile rows never reach scheduling."""

from lumilake_server.runtime.optimizer.halo import _retrieval_is_data_profile_eligible


def test_eligible_bare_sql() -> None:
    assert _retrieval_is_data_profile_eligible({"type": "sql", "template": "x"}) is True


def test_eligible_bare_s3() -> None:
    assert _retrieval_is_data_profile_eligible({"type": "s3", "template": "x"}) is True


def test_eligible_lumid_sql() -> None:
    assert (
        _retrieval_is_data_profile_eligible(
            {"type": "lumid", "mode": "sql", "template": "..."}
        )
        is True
    )


def test_eligible_lumid_s3() -> None:
    assert (
        _retrieval_is_data_profile_eligible(
            {"type": "lumid", "mode": "s3", "template": "..."}
        )
        is True
    )


def test_ineligible_lumid_agent() -> None:
    """Agent mode doesn't go through HALO's plan-choice path."""
    assert (
        _retrieval_is_data_profile_eligible({"type": "lumid", "mode": "agent"}) is False
    )


def test_ineligible_lumid_unknown_mode() -> None:
    assert (
        _retrieval_is_data_profile_eligible({"type": "lumid", "mode": "something"})
        is False
    )


def test_ineligible_lumid_missing_mode() -> None:
    assert _retrieval_is_data_profile_eligible({"type": "lumid"}) is False


def test_ineligible_unrelated_type() -> None:
    assert _retrieval_is_data_profile_eligible({"type": "list"}) is False
