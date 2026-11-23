"""A lumid retrieval in sql or s3 mode reaches HALO's plan-choice path."""

import pytest

from lumilake_server.runtime.optimizer.halo import HaloOptimizer
from tests.support.data_specs import runtime_op


@pytest.mark.parametrize(
    ("mode", "eligible"), [("sql", True), ("s3", True), ("agent", False)]
)
def test_lumid_retrieval_eligibility_follows_its_mode(
    mode: str, eligible: bool
) -> None:
    spec = {"type": "lumid", "mode": mode, "template": "..."}
    op = runtime_op("n", "data_retrieval", "data_retrieval", "", spec, {}, {})
    halo = HaloOptimizer.__new__(HaloOptimizer)

    assert halo._is_data_profile_candidate(op) is eligible
