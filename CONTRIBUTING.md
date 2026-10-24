# Contributing to Lumilake

Thanks for your interest in contributing. We welcome bug fixes, new features, documentation improvements, and feedback of all kinds.

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose v2 (for the deploy flow)

### Setup

Use the same install command CI uses so mypy/pytest can resolve the full workspace:

```bash
uv sync --all-packages --all-extras --all-groups
```

Typical dependency groups / extras:

| Group or extra | Purpose |
|----------------|---------|
| `--extra cli` | Installs the `lumilake` command |
| `--extra sdk` | Installs the Python SDK clients |
| `--extra deploy` | Installs Docker / FlowMesh deploy helpers |
| `--extra hook` | Installs the `lumilake_hook` resource-kind helpers |
| `--group lint` | `black`, `isort`, `ruff`, `mypy`, `codespell` (pinned to CI versions) |
| `--group test` | `pytest`, `pytest-asyncio`, `pytest-timeout` |
| `--group server-runtime` | Server image runtime deps — needed for mypy to type-check `src/lumilake_server` |

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
uv run pytest tests/                         # Full unit + parser/hook suite
uv run pytest tests/cli/                     # CLI-only
```

Coverage matches CI when run with `--cov`:

```bash
uv run pytest tests/ \
  --cov=lumilake --cov=lumilake_cli --cov=lumilake_deploy \
  --cov=lumilake_hook --cov=lumilake_server \
  --cov-report=term-missing
```

## Dependency Management

`pyproject.toml` is the source of truth for dependency ranges and optional extras. `uv.lock` is committed so local development and CI resolve consistently. After changing dependencies:

```bash
uv lock
```

Do not add generated requirements files unless a release workflow explicitly needs them.

### Pinned build tools

The server `Dockerfile` copies `uv`/`uvx` from the official uv image pinned by
its multi-platform index digest, and installs a hash-pinned `pip` wheel through
uv. CI pins the same uv release in every `astral-sh/setup-uv` step, and
`[tool.uv].required-version` enforces it locally. When changing uv:

1. Verify the upstream version and the uv image's index digest (covering both
   Linux amd64 and arm64) with
   `docker buildx imagetools inspect ghcr.io/astral-sh/uv:<version>`.
2. Update the digest in the `Dockerfile`, the `version:` in every
   `setup-uv` step, and `[tool.uv].required-version` together.

When changing the pinned `pip` wheel in the `Dockerfile`, take its SHA-256 from
the [official PyPI release metadata](https://pypi.org/pypi/pip/26.2.2/json) and
update the hash in the `Dockerfile`.

The CI secrets scan downloads a versioned Gitleaks archive and verifies its
repository-pinned SHA-256 before extraction. When updating Gitleaks, check the
archive hash against the upstream release asset checksums file, and update the
workflow pin and hash together. A checksum fetched alongside an archive during
CI is not a substitute for the reviewed repository pin.

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
- Run `uv run pre-commit run --all-files` and `uv run pytest tests/` locally before opening the PR.
- If you changed a dependency, update `uv.lock`.

## CI Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `lint-typecheck` | PR / main push | `pre-commit run --all-files` |
| `unit-tests` | PR / main push | `pytest tests/` with `pytest-cov` |
| `env-examples` | PR / main push | Validates `.env.example` against the deploy-time env contract |
| `package-build` | PR / main push | Builds wheels/sdists and smoke-tests the metapackage + four interface wheels |
| `image-publish` | PR / main / tag push | Builds PR images without publishing; pushes multi-arch `:sha`/`:dev`/`:vX.Y.Z` images to GHCR (not `:latest`) |
| `release` | GitHub release published | Rebuilds + verifies wheels, publishes to PyPI via OIDC after `pypi` environment approval |
| `release-images` | GitHub release published | Verifies the pre-pushed server image and retags it as `:latest` after `ghcr` environment approval |
| `security` | PR / main push | Runs workflow audit, secret scan, Bandit, and dependency audit |
| `check-signoff` | PR | Every non-merge commit must carry `Signed-off-by:` |
| `check-pr-title` | PR | Validates the PR title format |

See [`docs/RELEASE.md`](docs/RELEASE.md) for the full release runbook, including PyPI Trusted Publisher setup, GHCR visibility setup, and rollback options.

## Running Locally

`lumilake deploy` reads `.env` (and optionally `.env.flowmesh`) from `--project-dir` (`-C <path>`) or the current working directory. From a workspace checkout, `uv run lumilake deploy ...` works; from a PyPI install just use `lumilake deploy ...` directly. Either way, point the CLI at a deployment directory that holds your env files.

```bash
mkdir -p ~/lumilake-deploy

# First deploy (uv run from the workspace, or plain lumilake from PyPI)
uv run lumilake deploy -C ~/lumilake-deploy init       # writes ~/lumilake-deploy/.env
$EDITOR ~/lumilake-deploy/.env
uv run lumilake deploy -C ~/lumilake-deploy pull       # fetch the published server image
uv run lumilake deploy -C ~/lumilake-deploy up

# Iterate
uv run lumilake deploy -C ~/lumilake-deploy restart server
uv run lumilake deploy -C ~/lumilake-deploy logs server --tail 200

# Tear down
uv run lumilake deploy -C ~/lumilake-deploy down       # keeps data volumes
uv run lumilake deploy -C ~/lumilake-deploy reset      # drops everything
```

`LUMILAKE_DEPLOY_DIR=~/lumilake-deploy` lets you drop the `-C` flag on every call. The server is at http://127.0.0.1:9000; API docs at `/docs`.

## Code of Conduct

Be kind, be specific, share context. Assume good intent, and if a review feels harsh, ask for a rephrase.
