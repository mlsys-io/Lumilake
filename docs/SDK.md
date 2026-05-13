# SDK Reference

Lumilake ships sync and async Python clients behind the `sdk` extra. The SDK covers common server API surfaces and local deploy helpers. The CLI remains the complete operational command surface.

## Install

```bash
pip install "lumilake[sdk]"
```

Add deploy lifecycle helpers with:

```bash
pip install "lumilake[sdk,deploy]"
```

From a source checkout:

```bash
uv sync --group lint --group test --extra sdk --extra deploy
```

## Clients

```python
from lumilake.sdk import LumilakeClient

with LumilakeClient.from_config() as client:
    client.health()
    client.jobs.list()
```

```python
from lumilake.sdk import AsyncLumilakeClient

async with AsyncLumilakeClient.from_config() as client:
    await client.health()
    await client.jobs.list()
```

Both clients accept `base_url=` directly or load the server URL with `.from_config()`. Resolution order is explicit argument, `LUMILAKE_BASE_URL`, then `~/.lumilake/config.toml`.

## Resources

| Surface | Sync | Async |
|---------|------|-------|
| Health | `client.health()` | `await client.health()` |
| Info | `client.info.status()` | `await client.info.status()` |
| Deploy | `client.deploy.<verb>(...)` | `await client.deploy.<verb>(...)` |
| Jobs | `client.jobs.submit/list/get/cancel(...)` | same, await |
| Workers | `client.workers.list/get(...)` | same, await |
| Traces | `client.traces.list/get(...)` | same, await |

Job preview, progress, result, input, artifact, and watch helpers are available through the CLI and HTTP API.

## Deploy Extra

Without the `deploy` extra, deploy methods except `init` raise `DeployError` with an install hint. Server API resources work without the deploy extra.

Deploy methods call `lumilake.deploy` directly. Async deploy methods dispatch the same Python calls through `asyncio.to_thread` so Docker and FlowMesh setup work does not block the event loop.
