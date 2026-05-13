# Lumilake

Lumilake is a data analytics engine for agentic workflows. It accepts workflow specs (native graph JSON, YAML, or n8n JSON), builds execution plans, and submits runtime tasks through the configured runtime manager (FlowMesh).

## Goals

- Optimize multi-step AI/data workflows before execution.
- Ship the HALO scheduler in-tree.
- Expose an HTTP API for job submission, progress tracking, result retrieval, and trace analysis.

## Quick Start

```bash
# Install CLI + deps
uv sync --extra cli

# Copy the env template
uv run lumilake deploy init

# Edit .env (connection strings, bucket names, etc.)
$EDITOR .env

# Start the server. Add `--flowmesh` to init if you want the bundled
# FlowMesh stack managed by `lumilake deploy`.
uv run lumilake deploy up
```

The server comes up at `http://127.0.0.1:9000` (`/docs` for the API browser).

## Repository Layout

```text
.
├── src/lumilake/
│   ├── server/                # FastAPI app + API routes
│   ├── runtime/               # scheduler, optimizer, runtime manager (FlowMesh)
│   ├── ops/                   # logical operators
│   ├── schemas/               # API/runtime schemas
│   ├── sdk/                   # Python SDK
│   └── cli/                   # Typer CLI
├── scripts/                   # CI helpers
├── tests/                     # pytest suite
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── uv.lock                    # dev-only lockfile for `uv sync` reproducibility
└── pyproject.toml
```

## Deployment (recommended)

`.env` drives deployment. Copy the template:

```bash
uv run lumilake deploy init
```

For the bundled FlowMesh stack, generate the FlowMesh env as well:

```bash
uv run lumilake deploy init --flowmesh
```

Data access has two modes:

- **lumid.data mode** - set `LUMID_DATA_URL`; Lumilake sends SQL and storage operations through that service.
- **Direct mode** - set `DATABASE_URL`, `S3_URL`, and `S3_USER_DATA_PREFIX` to existing PostgreSQL and S3-compatible services.

Common commands:

```bash
uv run lumilake deploy up              # bring up server, and FlowMesh if .env.flowmesh exists
uv run lumilake deploy down            # stop services (keep data)
uv run lumilake deploy reset           # stop, purge data volumes, bring up again
uv run lumilake deploy restart server  # restart one service in place
uv run lumilake deploy logs             # stream server logs (server is the default service)
uv run lumilake deploy logs server --tail 200 --since 10m
```

The Docker image installs from `pyproject.toml` directly via `pip install .` on `python:3.12-slim`. After changing a dependency, run `uv lock` so the dev environment lockfile stays in sync.

## Run Server Directly (no Docker)

```bash
# Install everything
uv sync --group test --group lint

# Environment for a local run — can be exported or placed in .env
cp .env.example .env
$EDITOR .env

uv run python -m lumilake.server.main
```

Required env vars:

- Server: `LUMILAKE_SERVER_HOST`, `LUMILAKE_SERVER_PORT`
- Runtime: `LUMILAKE_RUNTIME_ORCHESTRATOR_URL`
- Data plane: `LUMID_DATA_URL`, or direct-mode `DATABASE_URL`, `S3_URL`, and `S3_USER_DATA_PREFIX`
- Archive: `S3_ARCHIVE_PREFIX`

Health check:

```bash
curl http://127.0.0.1:<LUMILAKE_SERVER_PORT>/healthz
```

API docs:

```text
http://127.0.0.1:<LUMILAKE_SERVER_PORT>/docs
```

## Submit and Monitor Jobs

Use the CLI:

```bash
uv run lumilake login http://127.0.0.1:9000
uv run lumilake job submit <workflow.yaml>
uv run lumilake job list
uv run lumilake job watch <job_id>
```

`uv run lumilake --help` lists every command group. Tune scheduling
behavior via `LUMILAKE_BATCH_ACCUMULATION_SECONDS` (wait window before
forming the first optimization batch) and `LUMILAKE_STARVATION_LIMIT`
(`0` = immediate starvation override) in `.env`.

## Scheduler

Lumilake ships the HALO scheduler: cost-aware scheduling via a DP solver over the
runtime graph and a multimodal cost model.

## Development

```bash
# Sync dev deps
uv sync --group lint --group test --extra cli

# Install pre-commit hooks (formatting + DCO sign-off)
uv run pre-commit install --install-hooks -t pre-commit -t prepare-commit-msg -t commit-msg

# Run all checks
uv run pre-commit run --all-files

# Run tests
uv run pytest tests/ --ignore=tests/server
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide: PR
title format, CI workflows, DCO sign-off, dependency pinning.

## License

See the repo root for license details.
