# CLI Reference

Install the CLI extra or use `uv run` from a source checkout:

```bash
pip install "lumilake[cli]"     # full CLI + deploy lifecycle
lumilake --help
```

```bash
uv run lumilake --help
```

`pip install lumilake-cli` (no extra) installs a thin CLI that omits the
deploy lifecycle — useful for users that only need `lumilake login` /
`job` / `worker` / `trace`. Running `lumilake deploy` from a thin install
prints an install hint pointing back at `lumilake[cli]` or
`lumilake-cli[deploy]`.

## Configuration

| Command | Purpose |
|---------|---------|
| `lumilake login <url>` | Save the server URL to `~/.lumilake/config.toml`. |
| `lumilake logout` | Remove saved CLI configuration. |
| `lumilake config` | Show saved CLI configuration. |
| `lumilake info` | Query server health metadata. |
| `lumilake health` | Query server health. |

## Deploy

Every `lumilake deploy` command reads `.env` (and optionally `.env.flowmesh`) from `--project-dir` (`-C <path>`) or, when not given, the current working directory. `LUMILAKE_DEPLOY_DIR` is an equivalent override.

```bash
mkdir -p ~/lumilake-deploy
lumilake deploy -C ~/lumilake-deploy init   # writes ~/lumilake-deploy/.env
# or:
LUMILAKE_DEPLOY_DIR=~/lumilake-deploy lumilake deploy init
```

The compose file and image are resolved from the installed `lumilake-deploy` package and GHCR; the deployment directory only needs to hold your `.env` files (and any local state docker compose creates).

If another FlowMesh stack is already running on the host, resolve port
collisions before `deploy up`. Common co-tenant defaults are FlowMesh
HTTP `8000`, gRPC `50051`, Redis control `6379`, and Redis telemetry
`6380`; the bundled stack reads the override keys documented in
`docs/ENV.md` from `.env.flowmesh`.

| Command | Purpose |
|---------|---------|
| `lumilake deploy init [--flowmesh]` | Create `.env`; optionally create `.env.flowmesh`. |
| `lumilake deploy doctor [--flowmesh]` | Validate deployment env files. |
| `lumilake deploy build` | Build the Lumilake server Docker image from source. |
| `lumilake deploy pull` | Pull the published server image from the registry (`$LUMILAKE_REGISTRY`). |
| `lumilake deploy up` | Start the local stack (image must already be present — run `pull` or `build` first). |
| `lumilake deploy down [--wipe-archive]` | Stop the stack but keep data volumes. Non-destructive. |
| `lumilake deploy clean` | Stop the stack and delete volumes. Destructive. |
| `lumilake deploy reset [--yes]` | Wipe every volume, then start the stack again. Destructive. |
| `lumilake deploy status` | Show known stack container state. |
| `lumilake deploy restart [service]` | Restart one service or the full stack. |
| `lumilake deploy logs [service]` | Stream or tail service logs. |
| `lumilake deploy update-flowmesh` | Re-lock and install the latest FlowMesh packages. |

### `deploy down` vs `deploy reset`

| Command | Stops services | Removes data volumes |
|---------|----------------|----------------------|
| `lumilake deploy down` | yes | no — archive (job records, run artifacts), compute Postgres, and MinIO corpus data survive, so `deploy up` resumes against the same state. `--wipe-archive` additionally wipes compute Postgres and FlowMesh runtime-state volumes while preserving MinIO corpus data. |
| `lumilake deploy clean` | yes | yes (every Lumilake-managed volume) |
| `lumilake deploy reset` | yes | yes (every Lumilake-managed volume), then re-runs `deploy up`. Prompts for confirmation; pass `--yes` to skip the prompt in scripts. |

Use `deploy down` between sessions. Reach for `clean` / `reset` only when you intentionally want to drop the local stack state.

## Jobs

| Command | Purpose |
|---------|---------|
| `lumilake job submit <workflow>` | Submit a workflow spec. |
| `lumilake job preview <workflow>` | Compile and schedule a workflow without dispatching runtime work. |
| `lumilake job list` | List jobs. |
| `lumilake job info <job_id>` | Show one job. |
| `lumilake job progress <job_id>` | Show detailed progress. |
| `lumilake job result <job_id>` | Show the final result. |
| `lumilake job inputs <job_id>` | Show resolved job inputs. |
| `lumilake job watch <job_id>` | Watch status until completion or failure. |
| `lumilake job cancel <job_id>` | Cancel a job. |
| `lumilake job artifact <job_id> --path <path> --output <file>` | Download a stored job artifact. |

Run `uv run lumilake job submit --help` for format, input, and output flags.

## Workers and Traces

| Command | Purpose |
|---------|---------|
| `lumilake worker list` | List runtime workers. |
| `lumilake worker get <worker_id>` | Show one worker. |
| `lumilake trace list` | List execution traces. |
| `lumilake trace get <exec_id>` | Show a trace summary, JSON payload, or Mermaid graph. |
