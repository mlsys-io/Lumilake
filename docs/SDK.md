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
from lumilake import LumilakeClient

with LumilakeClient.from_config() as client:
    client.health()
    client.jobs.list()
```

```python
from lumilake import AsyncLumilakeClient

async with AsyncLumilakeClient.from_config() as client:
    await client.health()
    await client.jobs.list()
```

Both clients accept `base_url=` directly or load the server URL with `.from_config()`. See [Configuration](#configuration) for resolution order.

## Configuration

The SDK and CLI share a small TOML file that records the server URL:

```
~/.lumilake/config.toml
```

Schema:

```toml
base_url = "http://127.0.0.1:9000"
```

### What writes it

`lumilake login <url>` writes the file with the supplied base URL. On
builds where `lumilake deploy up` also writes the local stack URL, it
updates the same file. If the file is missing after bringing up a stack,
run `lumilake login http://127.0.0.1:9000` once so
`LumilakeClient.from_config()` and every `lumilake <cmd>` invocation can
find the server.

### How `from_config()` resolves

`LumilakeClient.from_config()` (and the async equivalent) resolves the
base URL in this priority order:

1. `base_url=` argument passed to the constructor.
2. `LUMILAKE_BASE_URL` environment variable.
3. `~/.lumilake/config.toml`.

If none of the three yields a URL, the call raises `RuntimeError` with
the message:

```
no base_url provided and no saved config. Pass base_url= explicitly, set LUMILAKE_BASE_URL, or run `lumilake login`.
```

`from_config(path=...)` accepts a custom path for tests or non-default
installs.

The CLI currently reads the saved `~/.lumilake/config.toml`; run
`lumilake login <url>` or update that file for CLI calls.
`LUMILAKE_BASE_URL` is an SDK override.

## Resources

| Surface | Sync | Async |
|---------|------|-------|
| Health | `client.health()` | `await client.health()` |
| Deploy | `client.deploy.<verb>(...)` | `await client.deploy.<verb>(...)` |
| Jobs | `client.jobs.submit/list/get/cancel/wait(...)` | same, await |
| Workers | `client.workers.list/get(...)` | same, await |
| Traces | `client.traces.list/get(...)` | same, await |

Job preview, progress, result, input, artifact, and watch helpers are available through the CLI and HTTP API.

## Deploy Extra

Without the `deploy` extra, deploy methods except `init` raise `DeployError` with an install hint. Server API resources work without the deploy extra.

Deploy methods call `lumilake_deploy` directly. Async deploy methods dispatch the same Python calls through `asyncio.to_thread` so Docker and FlowMesh setup work does not block the event loop.
