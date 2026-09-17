from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_RUNNER_LANE_WORKFLOWS = (
    "env-examples.yml",
    "package-build.yml",
    "unit-tests.yml",
    "lint-typecheck.yml",
    "security.yml",
)

_OLD_DUPLICATED_TEXT = (
    "Lumilake is a PUBLIC repo, so the split is by TRUST OF THE TRIGGER"
)


def test_runner_lane_comment_is_not_duplicated_across_workflows() -> None:
    """Each of the five runner-lane workflows must carry a short pointer to
    CONTRIBUTING.md#runner-lanes instead of an inline copy of the policy
    comment."""
    for name in _RUNNER_LANE_WORKFLOWS:
        text = (REPO_ROOT / ".github" / "workflows" / name).read_text()
        assert (
            _OLD_DUPLICATED_TEXT not in text
        ), f"{name} still carries the old inline runner-lane policy comment"
        assert (
            "CONTRIBUTING.md#runner-lanes" in text
        ), f"{name} is missing the runner-lane pointer to CONTRIBUTING.md"


def test_contributing_md_owns_the_runner_lane_rationale() -> None:
    """The single owning explanation for the runner-lane split must live in
    CONTRIBUTING.md, not be re-derived from any one workflow file."""
    text = (REPO_ROOT / "CONTRIBUTING.md").read_text()
    assert "## Runner Lanes" in text
    assert "NUS_RUNNERS" in text
