# CLI Reference

Install the CLI extra or use `uv run` from a source checkout:

```bash
pip install "lumilake[cli]"
lumilake --help
```

```bash
uv run lumilake --help
```

## Configuration

| Command | Purpose |
|---------|---------|
| `lumilake login <url>` | Save the server URL to `~/.lumilake/config.toml`. |
| `lumilake logout` | Remove saved CLI configuration. |
| `lumilake config` | Show saved CLI configuration. |
| `lumilake info` | Query server health metadata. |
| `lumilake health` | Query server health. |

## Deploy

| Command | Purpose |
|---------|---------|
| `lumilake deploy init [--flowmesh]` | Create `.env`; optionally create `.env.flowmesh`. |
| `lumilake deploy doctor [--flowmesh]` | Validate deployment env files. |
| `lumilake deploy build` | Build the Lumilake server Docker image. |
| `lumilake deploy up` | Start the local stack. |
| `lumilake deploy down [--wipe-archive]` | Stop the stack while keeping data volumes. |
| `lumilake deploy clean` | Stop the stack and delete volumes. |
| `lumilake deploy reset` | Clean reset, then start the stack again. |
| `lumilake deploy status` | Show known stack container state. |
| `lumilake deploy restart [service]` | Restart one service or the full stack. |
| `lumilake deploy logs [service]` | Stream or tail service logs. |
| `lumilake deploy update-flowmesh` | Re-lock and install the latest FlowMesh packages. |

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
