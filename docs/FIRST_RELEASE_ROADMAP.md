# First Release Roadmap

This document tracks the remaining work before publishing Lumilake's first OSS release. The migration cleanup has already removed the private repository history, dead legacy surfaces, and non-OSS module references. The remaining work is now release hardening rather than source migration.

## 1. Documentation and Package Polish

Clean up public-facing docs so they describe the current OSS project only.

- Update `CONTRIBUTING.md` to remove stale references to `requirements.txt`, `sync_requirements.py`, the `requirements-sync` pre-commit hook, and the `requirements-sync` CI workflow.
- Update workflow documentation to match the current CI set: lint/typecheck, tests, security, package build, env examples, PR title, and DCO sign-off.
- Remove the permanent forbidden legacy env-key list from `scripts/dev/check_env_examples.py`; the env example check should validate the current contract without carrying a historical denylist.
- Add package metadata in `pyproject.toml`: license expression, authors/maintainers, project URLs, keywords, and classifiers.
- Add concise public reference docs for the surfaces users need after install: environment variables, CLI commands, API overview, architecture, and plugin/hooks model.
- Ensure README stays as the short quick start and points to the deeper docs instead of duplicating them.
- Update the issue templates to match the structure of Flowmesh

## 2. E2E Release Check

Run the project through a real user flow on the self-hosted environment and fix any issues found.

- Start from a clean checkout and install with the documented commands.
- Run `lumilake deploy init` and inspect the generated `.env`.
- Fill the required local test values and run `lumilake deploy up`.
- Verify server health and API docs are reachable.
- Submit at least one representative workflow through the CLI or SDK.
- Watch the job to completion and verify result retrieval.
- Confirm runtime artifacts and intermediate outputs go to `S3_ARCHIVE_PREFIX` or lumid.data, not container-local storage.
- Tear the deployment down with the documented command and confirm no unexpected local state is left behind.

The e2e check should be recorded as a release checklist item, not committed as a secret-bearing test. If a reusable script is added, it must accept all credentials and endpoints from the environment and avoid embedding local paths or private data.

## 3. Release Machinery

Add the workflows and docs needed to publish repeatably.

- Add a release documentation page covering version bump, lock refresh, tag policy, TestPyPI validation, PyPI publishing, and rollback/yank guidance.
- Add release metadata validation, including tag-to-version consistency.
- Add a GitHub release workflow using pinned actions and trusted publishing where possible.
- Decide whether the first release publishes only the `lumilake` package or also any split helper packages.
- Run the package build workflow before tagging and verify the built wheel/sdist import cleanly in a fresh environment.