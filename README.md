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
- Data access through direct PostgreSQL and S3-compatible storage; agent-style retrievals additionally route through lumid.data when `LUMID_DATA_URL` is set.
- Shared hook integration through `lumid-hooks`, plus Lumilake-owned optimizer plugins.

## Install

From PyPI:

```bash
pip install "lumilake[cli]"
```

From a source checkout:

```bash
uv sync --all-packages --all-extras --all-groups
```

The PyPI `lumilake` distribution is a code-free metapackage; install one of the extras below to get a working set. The server runtime is published as a Docker image only and is intentionally not on PyPI.

| Extra | Includes |
|-------|----------|
| `sdk` | Python SDK HTTP clients (`lumilake-sdk` → module `lumilake`) |
| `cli` | `lumilake` command line interface plus deploy lifecycle (`lumilake-cli` + `lumilake-deploy`) |
| `deploy` | Local Docker / FlowMesh deployment helpers (`lumilake-deploy`) |
| `hook` | Resource-kind helpers for shared hook integrations (`lumilake-hook`) |
| `all` | Everything above. |

## Quick Start

The server runs as the published Docker image. `lumilake deploy` reads its env files from `--project-dir` (or the current working directory). Either point at a deployment directory with `-C` / `--project-dir`, or `cd` to it first.

```bash
mkdir -p ~/lumilake-deploy
lumilake deploy -C ~/lumilake-deploy init --flowmesh   # ~/lumilake-deploy/.env + .env.flowmesh
$EDITOR ~/lumilake-deploy/.env                          # fill in DATABASE_URL / S3 / model keys
lumilake deploy -C ~/lumilake-deploy pull               # fetch ghcr.io/mlsys-io/lumilake_server:<tag>
lumilake deploy -C ~/lumilake-deploy up                 # bring the stack up via docker compose
```

`LUMILAKE_DEPLOY_DIR=~/lumilake-deploy` is an equivalent override. The deployment directory only needs to hold your `.env` files (and any local state docker compose creates) — the compose file and server image are resolved from the installed `lumilake-deploy` package and GHCR. The server listens on `http://127.0.0.1:9000` by default — open `/docs` for the API browser.

Note: a real workflow run also requires running PostgreSQL and S3-compatible storage; agent-style retrievals (`DataRetrievalOp` with `type: agent`) additionally require `LUMID_DATA_URL`. See `docs/ENV.md` for the env contract. If you don't have your own data plane, the repo ships a bundled Postgres + MinIO at `scripts/dev/compose.data-plane.yml` — see `docs/E2E_DEMO.md` for the full three-step demo flow (data plane → load demo data → run a workflow).

### Hello world

The repo ships a `hello-world.yaml` template — `FormatOp` →
`LambdaOp` → `LLMChatOp` — that is the smallest copy-paste starting
point for a Lumilake YAML workflow. Submit it once the stack is up:

```bash
# From a source checkout:
uv run lumilake job submit examples/templates/yaml/hello-world.yaml \
    --format yaml --input 'Name=world' --output-prefix demo/hello-world

# From a PyPI install (download the template alongside lumilake):
curl -O https://raw.githubusercontent.com/mlsys-io/lumilake_OSS/main/examples/templates/yaml/hello-world.yaml
lumilake job submit hello-world.yaml \
    --format yaml --input 'Name=world' --output-prefix demo/hello-world
lumilake job watch <job_id>
lumilake job result <job_id>
```

Edit `config.model` at the bottom of the YAML to match a model your
FlowMesh workers actually serve. If the job stays in `running`, run
`lumilake worker list` to see which models are available and edit
`config.model` accordingly.

### Real workflows

Submit and inspect a workflow. From a source checkout the example
workflow file is at `examples/templates/yaml/trading-agent.yaml`;
PyPI installs do not ship the templates, so pass an absolute path to a
workflow file you have locally:

```bash
# From a source checkout:
uv run lumilake login http://127.0.0.1:9000
uv run lumilake job submit examples/templates/yaml/trading-agent.yaml \
    --format yaml --input 'Stock=NVDA,AAPL,MSFT' --output-prefix demo/trading-agent

# From a PyPI install (lumilake on PATH; supply your own workflow file):
lumilake login http://127.0.0.1:9000
lumilake job submit /path/to/your/workflow.yaml \
    --format yaml --input 'Stock=NVDA,AAPL,MSFT' --output-prefix demo/trading-agent

lumilake job list
lumilake job watch <job_id>
```

See `docs/E2E_DEMO.md` for a full reproduction using the bundled demo
workflows and dataset.

## Data Access

- **SQL retrievals** connect directly to `DATABASE_URL`.
- **S3 retrievals** connect directly to `S3_URL`.
- **Agent retrievals** (`DataRetrievalOp` with `type: agent`) require `LUMID_DATA_URL` and route through lumid.data's `/agent/v1` endpoint.

Job records and runtime artifacts are written under `S3_ARCHIVE_PREFIX` using the same `S3_URL` connection.

## Deployment

Examples below assume you're set up with `LUMILAKE_DEPLOY_DIR=~/lumilake-deploy` (or pass `-C ~/lumilake-deploy` explicitly). Workspace-checkout users can prefix the commands with `uv run`; PyPI-install users invoke `lumilake` directly.

Generate `.env` from the bundled template:

```bash
lumilake deploy init
```

Generate both Lumilake and bundled FlowMesh env files:

```bash
lumilake deploy init --flowmesh
```

Common deployment commands:

```bash
lumilake deploy doctor
lumilake deploy pull         # or `build` to compile from source
lumilake deploy up
lumilake deploy status
lumilake deploy logs server --tail 200
lumilake deploy restart server
lumilake deploy down
lumilake deploy clean
```

Use `deploy down` to stop services while keeping data volumes (non-destructive). Use `deploy clean` or `deploy reset` only when you want to remove local stack state; both delete every Lumilake-managed volume, and `reset` prompts for confirmation by default (`--yes` skips the prompt).

## Python SDK

```python
from lumilake import LumilakeClient

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
├── src/lumilake_server/       # server runtime — image-only, not on PyPI
├── packages/sdk/              # `lumilake-sdk` → module `lumilake` (Client, envs, log)
├── packages/cli/              # `lumilake-cli` → `lumilake_cli` (Typer entry point)
├── packages/deploy/           # `lumilake-deploy` — packaged compose + .env.example assets
├── packages/hook/             # `lumilake-hook` → `lumilake_hook` (resource-kind helpers)
├── examples/                  # workflow templates and sample plugins
├── tests/                     # pytest suite
├── scripts/                   # CI and developer helpers
├── Dockerfile                 # builds ghcr.io/mlsys-io/lumilake_server
├── .env.example -> packages/deploy/.../assets/.env.example   # symlink for editors
├── uv.lock
└── pyproject.toml             # metapackage (`lumilake`) with [sdk]/[cli]/[deploy]/[hook]/[all] extras
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
