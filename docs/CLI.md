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
deploy lifecycle — useful for users that only need
`job` / `worker` / `trace` against a remote server. Running
`lumilake deploy` from a thin install prints an install hint pointing
back at `lumilake[cli]` or `lumilake-cli[deploy]`.

## Configuration

The CLI resolves the target server URL in this order:

1. `LUMILAKE_BASE_URL` environment variable.
2. `~/.lumilake/config.toml` (written by `lumilake deploy up`).
3. `http://127.0.0.1:9000` (the local deploy default).

| Command | Purpose |
|---------|---------|
| `lumilake config` | Show the resolved base URL and where it came from. |
| `lumilake info` | Query server health metadata. |
| `lumilake health` | Query server health. |

`lumilake deploy up` writes the resolved local server URL to `~/.lumilake/config.toml` after the stack is healthy (preserving any saved `api_key`), so subsequent CLI and SDK calls find it automatically. Remote / hosted users should set `LUMILAKE_BASE_URL` instead.

### Authentication

When the server runs an `IdentityProvider` plugin (e.g. `lumid_lumilake_plugin`), the CLI presents a bearer token via `Authorization: Bearer <api_key>`. Three commands manage it:

| Command | Purpose |
|---------|---------|
| `lumilake init <url> --api-key <key> [--config <path>] [--force]` | Save `base_url` + `api_key` to `~/.lumilake/config.toml`. Confirms overwrite unless `--force`. |
| `lumilake deinit [--config <path>]` | Delete the config file. |
| `lumilake config [--source auto\|file\|env] [--show-api-key]` | Show the resolved config. `api_key` is redacted (first 4 + last 4) unless `--show-api-key`. |

Resolution order matches the SDK: explicit args > `LUMILAKE_BASE_URL` / `LUMILAKE_API_KEY` env > `~/.lumilake/config.toml` > local defaults. The CLI sends no `Authorization` header when `api_key` resolves to nothing — fine for local-mode deploys, rejected by deploys running with `LUMILAKE_REQUIRE_IDENTITY_PROVIDER=1`.

## Deploy

Every `lumilake deploy` command reads `.env` (and optionally `.env.flowmesh`) from `--project-dir` (`-C <path>`) or, when not given, the current working directory. `LUMILAKE_DEPLOY_DIR` is an equivalent override.

```bash
mkdir -p ~/lumilake-deploy
lumilake deploy -C ~/lumilake-deploy init   # writes ~/lumilake-deploy/.env
# or:
LUMILAKE_DEPLOY_DIR=~/lumilake-deploy lumilake deploy init
```

`init` previews the file (or a unified diff against an existing one) before asking for confirmation; pass `--force` to skip the preview and prompt.

The compose file and image are resolved from the installed `lumilake-deploy` package and GHCR; the deployment directory only needs to hold your `.env` files (and any local state docker compose creates).

If another FlowMesh stack is already running on the host, resolve port
collisions before `deploy up`. Common co-tenant defaults are FlowMesh
HTTP `8000`, gRPC `50051`, Redis control `6379`, and Redis telemetry
`6380`; the bundled stack reads the override keys documented in
`docs/ENV.md` from `.env.flowmesh`.

| Command | Purpose |
|---------|---------|
| `lumilake deploy init [--flowmesh] [--force]` | Create `.env` (preview + confirm); optionally also `.env.flowmesh`. |
| `lumilake deploy doctor [--flowmesh]` | Validate deployment env files. |
| `lumilake deploy build` | Build the Lumilake server Docker image from source. |
| `lumilake deploy pull` | Pull the published server image from the registry (`$LUMILAKE_REGISTRY`). |
| `lumilake deploy up` | Start the local stack and persist `base_url` to `~/.lumilake/config.toml` (image must already be present — run `pull` or `build` first). |
| `lumilake deploy down [--wipe-archive]` | Stop the stack but keep data volumes. Non-destructive. |
| `lumilake deploy clean` | Stop the stack and delete volumes. Destructive. |
| `lumilake deploy purge <image_tag> [--dry-run] [--force]` | Remove one local Lumilake server image tag for the configured registry. |
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
Use `deploy purge <image_tag> --dry-run` to preview image cleanup; it removes
local server image tags only, not containers or volumes.

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

## `--json` flag

Most query commands (`lumilake job list/info/progress/result/inputs`, `lumilake worker list/get`, `lumilake trace list`) emit the server's JSON envelope by default, so scripts can parse them directly. `lumilake trace get` defaults to a human-readable summary table; pass `--json` (a shortcut for `--format json`) for parse-friendly output.
