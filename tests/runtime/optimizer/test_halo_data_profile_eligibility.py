"""Eligibility covers both retrieval shapes: a SQL or S3 profile node, and a
lumid retrieval in sql or s3 mode. A missed shape starves /profile rows."""

from lumilake_server.data_specs import (
    EmptyData,
    ListData,
    RetrievalData,
    S3ProfileData,
    SqlProfileData,
)
from lumilake_server.runtime.optimizer.halo import _retrieval_is_data_profile_eligible


def test_eligible_sql_profile() -> None:
    assert _retrieval_is_data_profile_eligible(SqlProfileData(type="sql", template="x"))


def test_eligible_s3_profile() -> None:
    assert _retrieval_is_data_profile_eligible(S3ProfileData(type="s3", template="x"))


def test_eligible_lumid_sql() -> None:
    assert _retrieval_is_data_profile_eligible(
        RetrievalData(type="lumid", mode="sql", template="...")
    )


def test_eligible_lumid_s3() -> None:
    assert _retrieval_is_data_profile_eligible(
        RetrievalData(type="lumid", mode="s3", template="...")
    )


def test_ineligible_lumid_agent() -> None:
    """Agent mode skips HALO's plan-choice path."""
    assert not _retrieval_is_data_profile_eligible(
        RetrievalData(type="lumid", mode="agent")
    )


def test_ineligible_unrelated_type() -> None:
    assert not _retrieval_is_data_profile_eligible(ListData(items=[]))
    assert not _retrieval_is_data_profile_eligible(EmptyData())
