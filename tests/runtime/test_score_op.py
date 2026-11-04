"""Tests for the Lumilake score op."""

import pytest

from lumilake_server.ops import ScoreOp


def test_score_op_requires_config_with_named_error() -> None:
    with pytest.raises(ValueError, match="ScoreOp requires config"):
        ScoreOp("question", ["candidate"])
