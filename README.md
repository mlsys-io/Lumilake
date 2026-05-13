# Lumilake

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Lint](https://github.com/mlsys-io/lumilake_OSS/actions/workflows/lint-typecheck.yml/badge.svg)](https://github.com/mlsys-io/lumilake_OSS/actions/workflows/lint-typecheck.yml)
[![Tests](https://github.com/mlsys-io/lumilake_OSS/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/mlsys-io/lumilake_OSS/actions/workflows/unit-tests.yml)

Lumilake is a data analytics engine for agentic workflows. It accepts workflow specs (native graph JSON, YAML, or n8n JSON), optimizes the runtime graph with HALO, and dispatches tasks through FlowMesh.

## What Lumilake Provides

- Workflow parsing for native graph specs, YAML workflows, and n8n exports.
- HALO scheduling for multi-step AI and data workflows.
- A FastAPI server for job submission, status, cancellation, results, workers, and traces.
- A CLI and Python SDK for local deployment and server API access.
- Data access through either lumid.data forwarding or direct PostgreSQL and S3-compatible storage.
- Shared hook integration through `lumid-hooks`, plus Lumilake-owned optimizer plugins.

## Install

From PyPI:

```bash
pip install "lumilake[cli]"
```

From a source checkout:

```bash
uv sync --group lint --group test --extra cli
```

The package is split into extras so lightweight imports stay small:

| Extra | Includes |
|-------|----------|
| `server` | FastAPI server, runtime, HALO scheduler, FlowMesh and lumid.data clients |
| `deploy` | Local Docker/FlowMesh deployment helpers |
| `cli` | `lumilake` command line interface |
| `sdk` | Python SDK HTTP clients |

## Quick Start

```bash
uv run lumilake deploy init --flowmesh
$EDITOR .env
uv run lumilake deploy up
```

The server listens on `http://127.0.0.1:9000` by default. Open `/docs` for the API browser.

Submit and inspect a workflow:

```bash
uv run lumilake login http://127.0.0.1:9000
uv run lumilake job submit examples/templates/yaml/image-generation.yaml --format yaml --input Stock=AAPL --output-prefix demo/image-generation
uv run lumilake job list
uv run lumilake job watch <job_id>
```

## Data Access

Lumilake supports two data-plane modes:

- **lumid.data mode**: set `LUMID_DATA_URL`; Lumilake forwards SQL and storage operations to lumid.data.
- **Direct mode**: set `DATABASE_URL`, `S3_URL`, and `S3_USER_DATA_PREFIX` to use existing PostgreSQL and S3-compatible services directly.

Job records and runtime artifacts are written under `S3_ARCHIVE_PREFIX`. In lumid.data mode, the archive operations are also forwarded through lumid.data.

## Deployment

Generate `.env` from the checked-in template:

```bash
uv run lumilake deploy init
```

Generate both Lumilake and bundled FlowMesh env files:

```bash
uv run lumilake deploy init --flowmesh
```

Common deployment commands:

```bash
uv run lumilake deploy doctor
uv run lumilake deploy build
uv run lumilake deploy up
uv run lumilake deploy status
uv run lumilake deploy logs server --tail 200
uv run lumilake deploy restart server
uv run lumilake deploy down
uv run lumilake deploy clean
```

Use `deploy down` to stop services while keeping data volumes. Use `deploy clean` or `deploy reset` only when you want to remove local stack state.

## Python SDK

```python
from lumilake.sdk import LumilakeClient

with LumilakeClient(base_url="http://127.0.0.1:9000") as client:
    print(client.health())
    print(client.jobs.list())
```

Install the SDK extra for HTTP clients:

```bash
pip install "lumilake[sdk]"
```

Install deploy support as well if you want `client.deploy.*` methods:

```bash
pip install "lumilake[sdk,deploy]"
```

See `docs/SDK.md` for the SDK resource map.

## Documentation

- `docs/ENV.md` - environment variables and data-plane modes.
- `docs/CLI.md` - command groups and common CLI usage.
- `docs/WORKFLOWS.md` - workflow input formats and YAML structure.
- `docs/OPS.md` - built-in operation classes.
- `docs/SDK.md` - sync and async Python client usage.
- `docs/API.md` - server route overview and response shape.
- `docs/ARCHITECTURE.md` - module layout and runtime flow.
- `docs/PLUGINS.md` - shared hooks and Lumilake plugin model.
- `docs/CODE_STYLE.md` - coding rules for contributors and agents.

## Plugins

Lumilake wires shared hook protocols from `lumid-hooks` for identity, permissions, resource registration, submission guards, and usage sinks. Optimizer registration remains Lumilake-specific.

A minimal in-memory plugin is available under `examples/plugins/simple_plugin/`.

## Repository Layout

```text
.
├── src/lumilake/              # server, runtime, CLI, SDK, deploy helpers
├── src/lumilake_hook/         # Lumilake resource-kind helpers for hooks
├── examples/                  # workflow templates and sample plugins
├── tests/                     # pytest suite
├── scripts/                   # CI and developer helpers
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── uv.lock
└── pyproject.toml
```

## Development

```bash
uv sync --group lint --group test --extra cli
uv run pre-commit install --install-hooks -t pre-commit -t prepare-commit-msg -t commit-msg
uv run pre-commit run --all-files
uv run pytest tests/
```

After changing dependencies, run:

```bash
uv lock
```

See `CONTRIBUTING.md` for PR title format, CI workflows, DCO sign-off, dependency guidance, and local testing notes.

## License

Apache-2.0. See `LICENSE`.
