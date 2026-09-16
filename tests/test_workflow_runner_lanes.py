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
    """These five workflows used to each carry their own copy of an
    ~11-line runner-lane policy comment. Pins the replacement of that
    duplicated block (in each of the five ``.github/workflows/*.yml`` files)
    with a short pointer to CONTRIBUTING.md#runner-lanes: reverting any
    single file back to the old inline block makes that file's checks fail
    below, even though the other four files and CONTRIBUTING.md are
    unchanged."""
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
