# `lumilake.sdk`

Python SDK for Lumilake. Programmatic equivalent of the `lumilake` CLI — every CLI subcommand has a corresponding method on `LumilakeClient` (sync) and `AsyncLumilakeClient` (async). Two parallel client classes; each resource module ships both sync and async variants. Lives inside the `lumilake` distribution and ships behind the `[sdk]` extra so basic installs stay slim.

## Install

Server-API + filesystem `init` (small surface; httpx only):

```bash
pip install 'lumilake[sdk]'
```

Add deploy lifecycle (`client.deploy.up/down/...`). Pulls in `[deploy]` (Docker SDK, psycopg, FlowMesh stack helpers); `client.deploy.*` calls into `lumilake.deploy` directly:

```bash
pip install 'lumilake[sdk,deploy]'
```

In the workspace:

```bash
uv sync --group lint --group test --extra sdk --extra deploy
```

Without `[deploy]`, every `client.deploy.*` method except `init` raises `DeployError` with an install hint on first call. Server-API resources work in either configuration.

## Sync usage

```python
from lumilake.sdk import LumilakeClient

with LumilakeClient.from_config() as client:        # reads ~/.lumilake/config.toml
    client.health()
    client.jobs.list()
    client.jobs.submit({"workflow": ...})
    client.deploy.up()
```

## Async usage

```python
from lumilake.sdk import AsyncLumilakeClient

async with AsyncLumilakeClient.from_config() as client:
    await client.health()
    await client.jobs.list()
    await client.jobs.submit({"workflow": ...})
    await client.deploy.up()
```

Both clients accept `base_url=` directly, or pull it from `~/.lumilake/config.toml` via `.from_config()`. Resolution order: explicit arg > `LUMILAKE_BASE_URL` env > saved config.

## CLI ↔ SDK mapping

| CLI command | Sync | Async |
|---|---|---|
| `lumilake login <url>` | (writes `~/.lumilake/config.toml`) | — |
| `lumilake info` / `health` | `client.info.status()` / `client.health()` | same, await |
| `lumilake deploy {init,up,down,clean,restart,reset,logs,update-flowmesh}` | `client.deploy.<verb>(...)` | `await client.deploy.<verb>(...)` |
| `lumilake deploy {doctor,build,status}` | CLI only | CLI only |
| `lumilake job {submit,list,get,cancel,logs,wait}` | `client.jobs.<verb>(...)` | `await client.jobs.<verb>(...)` |
| `lumilake worker {list,get}` | `client.workers.<verb>(...)` | `await client.workers.<verb>(...)` |
| `lumilake trace {list,get}` | `client.traces.<verb>(...)` | `await client.traces.<verb>(...)` |

## Layout

```
src/lumilake/sdk/
├── __init__.py                         # exports LumilakeClient, AsyncLumilakeClient
├── client.py                           # LumilakeClient (sync)
├── async_client.py                     # AsyncLumilakeClient
├── _base_client.py                     # BaseClient + BaseAsyncClient (shared)
├── config.py                           # LumilakeConfig — TOML round-trip
├── errors.py                           # LumilakeError + NotFound/Http/DeployError
└── resources/                          # one module per CLI command group
    ├── _base.py                        # SyncResource + AsyncResource
    ├── info.py                         # Info + AsyncInfo
    ├── deploy.py                       # Deploy + AsyncDeploy
    ├── jobs.py                         # Jobs + AsyncJobs
    ├── workers.py                      # Workers + AsyncWorkers
    └── traces.py                       # Traces + AsyncTraces
```

## Design notes

- **Sync + async parity.** One sync class, one async class, every resource exposes both. Shared base classes (`BaseClient` / `BaseAsyncClient`) handle version-prefix, envelope unwrapping, and error → exception mapping. Async resources `await` the same code paths.
- **Two transport modes.** Server-API resources speak HTTP (`httpx.Client` / `httpx.AsyncClient`). Deploy lifecycle calls `lumilake.deploy` directly; async dispatches the same Python calls through `asyncio.to_thread` so the event loop stays responsive across Docker and FlowMesh setup work.
- **Config shared with the CLI.** `from_config()` reads the same `~/.lumilake/config.toml` the CLI writes via `lumilake login`.
- **Never raw `docker compose up`.** Deploy resource always uses the `lumilake.deploy` machinery — the project rule applies inside the SDK too.
- **Context-manager support.** `with LumilakeClient(...) as client:` (sync) and `async with AsyncLumilakeClient(...) as client:` close the underlying httpx client cleanly.
