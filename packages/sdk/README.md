# `lumilake`

Python SDK for Lumilake server APIs and local deploy helpers. It provides sync and async resource clients for common server surfaces; the CLI remains the complete command surface for operational helpers. The SDK lives inside the `lumilake` distribution and ships behind the `[sdk]` extra so basic installs stay slim.

## Install

Server-API + filesystem `init` (small surface; httpx only):

```bash
pip install 'lumilake[sdk]'
```

Add deploy lifecycle (`client.deploy.up/down/...`). Pulls in `[deploy]` (Docker SDK, FlowMesh stack helpers); `client.deploy.*` calls into `lumilake_deploy` directly:

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
from lumilake import LumilakeClient

with LumilakeClient.from_config() as client:        # reads ~/.lumilake/config.toml
    client.health()
    client.jobs.list()
    client.jobs.submit({"workflow": ...})
    client.deploy.up()
```

## Async usage

```python
from lumilake import AsyncLumilakeClient

async with AsyncLumilakeClient.from_config() as client:
    await client.health()
    await client.jobs.list()
    await client.jobs.submit({"workflow": ...})
    await client.deploy.up()
```

Constructing either client without `.from_config()` resolves the server URL in this order: explicit `base_url=`, `LUMILAKE_BASE_URL`, then `~/.lumilake/config.toml`. `.from_config(path=...)` reads a specific TOML file directly.

## CLI and SDK surfaces

| Surface | Sync | Async |
|---|---|---|
| `lumilake deploy up` | writes `~/.lumilake/config.toml` | - |
| `lumilake info` / `health` | `client.health()` | same, await |
| `lumilake deploy {init,up,down,clean,restart,reset,logs,update-flowmesh}` | `client.deploy.<verb>(...)` | `await client.deploy.<verb>(...)` |
| `lumilake deploy {doctor,build,status}` | CLI only | CLI only |
| Job submit/preview/list/list_all/get/progress/result/inputs/artifact/cancel/wait/watch | `client.jobs.<verb>(...)` | `await client.jobs.<verb>(...)` |
| `lumilake worker {list,get}` (+ `list_all`) | `client.workers.<verb>(...)` | `await client.workers.<verb>(...)` |
| `lumilake trace {list,get}` (+ `list_all`) | `client.traces.<verb>(...)` | `await client.traces.<verb>(...)` |

## Design notes

- **Sync + async parity.** One sync class, one async class, and each SDK resource exposes both. Shared base classes (`BaseClient` / `BaseAsyncClient`) handle version-prefix, envelope unwrapping, and error-to-exception mapping. Async resources `await` the same code paths.
- **Two transport modes.** Server-API resources speak HTTP (`httpx.Client` / `httpx.AsyncClient`). Deploy lifecycle calls `lumilake_deploy` directly; async dispatches the same Python calls through `asyncio.to_thread` so the event loop stays responsive across Docker and FlowMesh setup work.
- **Config shared with the CLI.** `from_config()` reads the same `~/.lumilake/config.toml` the CLI writes from `lumilake deploy up`.
- **Never raw `docker compose up`.** Deploy resource always uses the `lumilake_deploy` machinery — the project rule applies inside the SDK too.
- **Context-manager support.** `with LumilakeClient(...) as client:` (sync) and `async with AsyncLumilakeClient(...) as client:` close the underlying httpx client cleanly.
