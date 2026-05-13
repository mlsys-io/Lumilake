# AGENTS.md

Guidance for agentic tools (Claude Code, Cursor, Aider, etc.) working in this repo. This file is the single source of truth; `CLAUDE.md` points at it. For contributor-facing setup details, see `CONTRIBUTING.md`.

## Setup

- Environment is managed by `uv`. The system may not have Python installed directly. Always use `uv run ...` to execute commands.
- `uv sync --group lint --group test --extra cli` installs everything a development-focused agent needs.
- `uv run pre-commit run --all-files` runs every formatting / lint / type-check / spell-check the CI runs. This must pass before any PR lands. The exact tool set lives in `.pre-commit-config.yaml`.
- **Scratch.** Never commit throwaway scripts / experiments.

## Design Discipline

- **Match existing code style and architectural patterns.** Don't invent a new pattern for a problem the codebase already has a solution for.
- **When uncertain, pick the simpler, more concise implementation.**
- **Prefer clear abstractions; state should be explicit.** If you're managing state in a Python class, declare the fields — don't dynamically `setattr` a field and later `getattr` it on the same object.
- **Don't create trivial 1-2 LOC helpers used only once** unless they significantly improve readability. Inline the expression.
- **Minimize comments.** Code should be self-documenting. A good comment reminds the reader of non-obvious global context that can't be inferred locally ("why" never "what"). If removing the comment doesn't confuse a future reader, don't write it.
- **Keep the public surface self-contained.** Do not add extension points for modules or services that are not part of this repository.

## General Code Style

- Python 3.12+, type hints throughout.
- **No `print()`** — use the project logger.
- **No back-compat shims.** When updating code, replace the old path outright — don't keep both. Old paths will never be used again. This includes re-exports, aliases, no-op stubs, and source-history breadcrumb comments.
- **Defaults live in the env / profile layer.** Don't duplicate a profile-layer default at the call site (e.g. `or "postgres"` after `profile.pg_user`).
- **In `except`, only raise.** Don't swallow library errors into booleans or `None`-returns. The narrow exception is a documented not-found probe where the library has no list-based alternative — translate only the documented missing-resource case and re-raise everything else.
- **No `del arg` to silence unused-param lint.** Remove the parameter entirely and update callers.
- **Prefer Pydantic v2 models** over untyped dicts for API payloads and profile schemas. Avoid `.get()` / `getattr` when a key is known to exist.
- **Prefer the CLI** (`uv run lumilake ...`) over shell scripts and raw HTTP calls.

## Python Typing and Imports

- **Always import on top.** Use inline imports only when strictly necessary — breaking a genuine circular import (prefer extracting a leaf module first), or gating an optional heavy dependency behind try/except.
- **Never use `importlib`.** Always `from xxx import xxx` or `import xxx`.
- **Prefer `X | Y`** over `typing.Union[X, Y]`, and **`X | None`** over `typing.Optional[X]`.
- **Prefer `typing.Any`** over `object` in annotations. Only use `object` when `Any` is semantically wrong (e.g. framework override signatures).
- **`# type: ignore` is a last resort.** Exhaust alternatives first: fix the type, add an `isinstance` guard, add the missing dependency or type stub. When unavoidable, always use the specific error code (e.g. `# type: ignore[arg-type]`) — never bare `# type: ignore`.
- **Missing-dep type errors → add the dep** to `pyproject.toml`, then `uv lock` to update `uv.lock`. Don't suppress with `# type: ignore`.
- **Avoid `hasattr` / `getattr` that bypasses type checking.** When a variable has a broad type, use `isinstance` to guard attribute access. Acceptable uses of `getattr`: dynamic dispatch, defaults with `getattr(obj, attr, default)`, or reaching untyped third-party libraries.
- **Don't enforce `*` (keyword-only separator) by default.** Use it conservatively, only when it prevents a specific misuse.
- **Don't write `from __future__ import annotations`** unless strictly necessary. For forward references, use `typing.Self` when applicable, or quote the type as a string literal.

## Docker and Deploy

- The Docker image installs the server extra via `pip install ".[server]"` — `pyproject.toml` is canonical. `uv.lock` exists only for reproducible dev environments; the Docker build does not consume it. After a runtime-dep change, run `uv lock` to refresh the dev lock so `uv sync` keeps working. See `Dockerfile` for the build itself.
- `docker compose up/down` stays on subprocess — the Docker Python SDK has no compose equivalent. One-shot container ops use the SDK wrapper. Don't reintroduce raw `subprocess.run(["docker", ...])` for those. See `src/lumilake/deploy/` for current call sites.

## Testing

- `uv run pytest tests/ --ignore=tests/server` is the default. `tests/server` spins up a live server and is exercised manually before release.
- When you change a runtime path, add or update a test under `tests/` in the same PR.

## Logs and Results

- Job records, runtime artifacts, and FlowMesh intermediate outputs are stored through `S3_ARCHIVE_PREFIX` (or lumid.data when configured).
- Container logs: `uv run lumilake deploy logs` (streams the server by default). `--tail N` / `--since 10m` / `-t` for common filters.

## Execution

- When cancelling a job, the request must be cancelled as well.

## Pull Requests

- **Before opening a PR, check for duplicate work:**
  ```bash
  gh pr list --state open --search "<short area keywords>"
  gh issue view <issue_number> --comments
  ```
  If an open PR already addresses the same thing, don't open another.
- Only stage files directly related to the commit. Never use `git add -A` or `git add .` — they pick up scratch files.
- PR title format: `type(scope): description`. Allowed types: `feat, fix, refactor, chore, test, perf, build, ci, docs`. Prefix with `[BREAKING]` for breaking changes.
- **Sign off every commit** under the Developer Certificate of Origin. Use `git commit -s`, or install the pre-commit hooks to auto-append:
  ```bash
  uv run pre-commit install --install-hooks -t pre-commit -t prepare-commit-msg -t commit-msg
  ```
- **Disclose AI assistance** via a commit trailer: `Co-Authored-By: <agent name> <email>`.
- Avoid `--amend` on pushed commits; create a new commit instead. When a pre-commit hook fails, the commit didn't happen — fix and re-stage, then commit fresh.

## Editing This File

Every new rule here adds to the token budget every agent pays on load. Keep it lean.

- **AGENTS.md stays under 200 lines.** When it's close, split a section into a domain guide before adding.
- **Test before codifying.** If agents already do the right thing without the rule, don't add it. If it's a one-off incident, prefer a lint/CI/test fence instead.
- **Don't duplicate other docs.** Link to or point at the source of truth (`Dockerfile`, `pyproject.toml`, `CONTRIBUTING.md`, upstream tool docs); don't re-paste their details here. Specifics drift; pointers don't.
- **No code-specific examples.** Rules describe principles, not particular call sites. The example will rot when the code moves.
- **No hardcoded paths that drift.** Prefer "search for X" patterns over specific file paths that may move.
- **Consolidate conflicts.** If two rules touch the same topic, merge them instead of stacking.
