# Contributing to Lumilake

Thanks for your interest in contributing. We welcome bug fixes, new features, documentation improvements, and feedback of all kinds.

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose v2 (for the deploy flow)

### Setup

```bash
uv sync --group lint --group test --extra cli
```

Typical dependency groups / extras:

| Group or extra | Purpose |
|----------------|---------|
| `--extra cli` | Installs the `lumilake` command |
| `--group lint` | `black`, `isort`, `ruff`, `mypy`, `codespell` (pinned to CI versions) |
| `--group test` | `pytest`, `pytest-asyncio`, `pytest-timeout` |

For full development across all deps:

```bash
uv sync --all-groups --all-extras
```

### Install Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to enforce formatting, linting, type-checking, spell-checking, and DCO sign-off on every commit.

```bash
uv run pre-commit install --install-hooks -t pre-commit -t prepare-commit-msg -t commit-msg
```

This installs three hook stages:

- **pre-commit** — runs `isort`, `black`, `ruff`, `mypy`, and `codespell` on staged files.
- **prepare-commit-msg** — automatically appends a [DCO sign-off](#signing-off-commits-dco) line to your commit message.
- **commit-msg** — verifies the sign-off is present (safety net).

## Code Style

| Tool | Purpose | Config |
|------|---------|--------|
| [isort](https://pycqa.github.io/isort/) | Import sorting | `pyproject.toml` `[tool.isort]` |
| [Black](https://black.readthedocs.io/) | Code formatting | `pyproject.toml` `[tool.black]` |
| [Ruff](https://docs.astral.sh/ruff/) | Linting | `pyproject.toml` `[tool.ruff]` |
| [mypy](https://mypy.readthedocs.io/) | Type checking | `pyproject.toml` `[tool.mypy]` |
| [codespell](https://github.com/codespell-project/codespell) | Spell checking | `pyproject.toml` `[tool.codespell]` |

Run all checks manually:

```bash
uv run pre-commit run --all-files
```

## Testing

```bash
uv run pytest tests/ --ignore=tests/server  # Default unit suite
uv run pytest tests/cli/                     # CLI-only
```

## Dependency Management

`pyproject.toml` is the source of truth for dependency ranges and optional extras. `uv.lock` is committed so local development and CI resolve consistently. After changing dependencies:

```bash
uv lock
```

Do not add generated requirements files unless a release workflow explicitly needs them.

## Signing Off Commits (DCO)

All contributions must be signed off under the [Developer Certificate of Origin](https://developercertificate.org/). Append a `Signed-off-by: Your Name <your.email@example.com>` line to every commit message.

Easiest: use `git commit -s`, or install the pre-commit hooks above to have it auto-appended.

The `Check DCO Sign-off` CI job verifies every non-merge commit in a PR carries the line. If you forget on an existing branch:

```bash
git rebase --signoff HEAD~N   # N = number of commits to sign off
```

## Pull Request Guidelines

- **Title format** (enforced by CI): `type(scope): description`. Allowed types: `feat, fix, refactor, chore, test, perf, build, ci, docs`. Scope is optional. Prefix with `[BREAKING]` for breaking changes.
- Keep PRs focused. Split unrelated changes into separate PRs.
- Fill in the PR template's Purpose / Changes / Design / Test Plan sections.
- Run `uv run pre-commit run --all-files` and `uv run pytest tests/ --ignore=tests/server` locally before opening the PR.
- If you changed a dependency, update `uv.lock`.

## CI Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `lint-typecheck` | PR / main push | `pre-commit run --all-files` |
| `unit-tests` | PR / main push | `pytest tests/ --ignore=tests/server` |
| `env-examples` | PR / main push | Validates `.env.example` against the deploy-time env contract |
| `package-build` | PR / main push | Builds wheel/sdist and smoke-tests the package |
| `security` | PR / main push | Runs workflow audit, secret scan, Bandit, and dependency audit |
| `check-signoff` | PR | Every non-merge commit must carry `Signed-off-by:` |
| `check-pr-title` | PR | Validates the PR title format |

## Running Locally

```bash
# First deploy
uv run lumilake deploy init          # copies .env.example -> .env
# edit .env to taste
uv run lumilake deploy up

# Iterate
uv run lumilake deploy restart server
uv run lumilake deploy logs server --tail 200

# Tear down
uv run lumilake deploy down          # keeps data volumes
uv run lumilake deploy reset         # drops everything
```

The server is at http://127.0.0.1:9000; API docs at `/docs`.

## Code of Conduct

Be kind, be specific, share context. Assume good intent, and if a review feels harsh, ask for a rephrase.
