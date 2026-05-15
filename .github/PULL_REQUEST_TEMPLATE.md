<!-- markdownlint-disable -->

<!--
PR title format (enforced by CI):
  type(scope): description
  e.g. feat(deploy): add --since option to lumilake deploy logs
       fix(server): handle 404 from MinIO user_info probe
       [BREAKING] refactor: rename client.get_job to client.fetch_job

Types: feat, fix, refactor, chore, test, perf, build, ci, docs
Scope (optional): deploy, server, cli, runtime, docker, ...
-->

## Purpose

<!-- What does this PR do? Reference related issues with "Fixes #123" or "Relates to #123". -->

## Changes

<!-- List modified files or groups of files with a brief explanation. -->
<!--
- `packages/cli/src/lumilake_cli/commands/deploy.py` — added `--since` flag to `deploy logs`
- `packages/deploy/src/lumilake_deploy/docker_client.py` — thread `since` through to docker-py
- `tests/cli/test_deploy_impl.py` — cover the new flag
-->

## Design

<!-- For non-trivial PRs: the high-level approach and alternatives considered. -->

## Test Plan

<!-- How were these changes validated? Commands, sample workflows, or screenshots. -->

## Test Result

<!-- Paste relevant test output, logs, or before/after comparisons. -->

---

<details>
<summary>Pre-submission Checklist</summary>

- [ ] I have read `CONTRIBUTING.md`.
- [ ] I have run `uv run pre-commit run --all-files` and fixed any issues.
- [ ] I have added or updated tests covering my changes (if applicable).
- [ ] I have verified that `uv run pytest tests/` passes locally.
- [ ] If I changed the SDK or CLI, I have verified the affected interface works locally.
- [ ] If this is a breaking change, I have prefixed the PR title with `[BREAKING]` and described migration steps above.
- [ ] I have updated documentation or config examples if user-facing behavior changed.

</details>
