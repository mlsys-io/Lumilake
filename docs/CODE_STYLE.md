# Code Style

This guide collects rules agents and contributors should apply before editing source code. Use `CONTRIBUTING.md` for setup, validation commands, commits, and PR conventions.

## General

- Use Python 3.12+ and type hints throughout.
- Use `uv run ...` for project commands. The system Python may not match the repository environment.
- Match existing architecture and local helper APIs before adding new patterns.
- Prefer the simpler implementation when two designs have the same behavior.
- Do not add extension points for modules or services that are not part of this repository.
- Do not keep compatibility layers for unreleased behavior. Replace old paths outright instead of adding aliases, re-exports, stubs, or dual contracts.
- Do not commit throwaway scripts, scratch files, local experiments, or generated debug output.

## Python

- Import at the top of the file. Inline imports are only for genuine circular imports or optional heavy dependencies.
- Do not use `importlib` when a normal `import` or `from ... import ...` works.
- Prefer `X | Y` and `X | None` over `typing.Union` and `typing.Optional`.
- Prefer `typing.Any` over `object` unless `Any` would be semantically wrong.
- Use Pydantic v2 models for API payloads and profile schemas when structured data crosses a boundary.
- Avoid `.get()` and `getattr()` for keys or attributes known to exist. Narrow broad types with `isinstance` instead.
- Declare object state as fields. Do not dynamically `setattr` a field and later `getattr` it on the same object.
- Avoid `hasattr()` for normal control flow; it bypasses useful type checking.
- Use a specific `# type: ignore[code]` only after exhausting ordinary typing fixes.
- Missing-dependency type errors should be fixed by adding the dependency or type stub, then refreshing `uv.lock`.
- Do not use `del arg` to silence unused-parameter lint. Remove the parameter and update callers.
- Do not enforce keyword-only arguments by default. Use `*` only when it prevents a concrete misuse.
- Do not add `from __future__ import annotations` unless it is necessary.

## Logging and Errors

- Do not use `print()` in source code. Use the project logger.
- In `except` blocks, only translate documented expected cases. Re-raise unexpected library errors.
- Do not return `None` or `False` to hide an error unless the function contract explicitly models a not-found probe.
- Defaults belong in the env or profile layer. Do not duplicate profile-layer defaults at call sites.

## Comments and Docs

- Minimize comments. Add one only when the reason cannot be inferred locally.
- Docstrings and comments describe current behavior, not what the code replaced.
- Do not leave source-history breadcrumbs such as "old path", "replacement for", or "previously".
- Do not add hardcoded file-path examples that will drift; point to the owning doc or command instead.

## Deploy and Runtime

- Prefer the Lumilake CLI, SDK, and deploy helpers over raw HTTP or direct Docker commands.
- `docker compose up/down` remains subprocess-based because the Docker Python SDK has no compose equivalent.
- One-shot container operations should go through the Docker SDK wrapper already used by `packages/deploy/src/lumilake_deploy/`.
- After runtime dependency changes, update `pyproject.toml` and refresh `uv.lock`.
- Job records, runtime artifacts, and FlowMesh intermediate outputs are stored through the archive layer. Do not introduce local result directories for runtime data.
- Cancelling a job must cancel the underlying runtime request as well.

## Tests

- Default unit validation is `uv run pytest tests/`. CI also collects coverage via `pytest-cov` against the SDK, CLI, deploy, hook, and server packages.
- When changing runtime behavior, add or update a focused test in the same PR.
- Do not ignore failing checks. A red test on `main` is a CI gap to close, not a reason to skip validation.

## Git Hygiene

- Stage only files related to the commit. Do not use `git add -A` or `git add .`.
- Sign off every commit with `git commit -s`.
- Add an AI assistance trailer when applicable: `Co-Authored-By: <agent name> <email>`.
- Before opening a PR, check for duplicate open PRs or issues in the same area.
- Avoid amending pushed commits unless explicitly requested. If a hook fails, the commit did not happen; fix, re-stage, and commit again.
