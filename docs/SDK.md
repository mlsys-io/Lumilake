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

`lumilake deploy up` writes the file with the local stack URL once the
stack starts. Remote / hosted users should pass `base_url=` explicitly
or set `LUMILAKE_BASE_URL`.

### How clients resolve

`LumilakeClient(...)` (and the async equivalent) resolves the base URL
in this priority order:

1. `base_url=` argument passed to the constructor.
2. `LUMILAKE_BASE_URL` environment variable.
3. `~/.lumilake/config.toml`.

If none of the three yields a URL, the call raises `RuntimeError` with
the message:

```
no base_url provided and no saved config. Pass base_url= explicitly, set LUMILAKE_BASE_URL, or run `lumilake deploy up`.
```

`from_config(path=...)` reads a specific config file directly; use it
for tests or non-default installs.

## Resources

| Surface | Sync | Async |
|---------|------|-------|
| Health | `client.health()` | `await client.health()` |
| Deploy | `client.deploy.<verb>(...)` | `await client.deploy.<verb>(...)` |
| Jobs | `client.jobs.submit / preview / list / list_all / get / progress / result / inputs / artifact / cancel / wait / watch(...)` | same, await |
| Workers | `client.workers.list / list_all / get(...)` | same, await |
| Traces | `client.traces.list / list_all / get(...)` | same, await |

### Jobs

All `Jobs` / `AsyncJobs` methods mirror the CLI surface and the server's HTTP routes one-to-one:

```python
client.jobs.submit({"data": [...]}, workflow_format="yaml")
client.jobs.preview({"data": [...]}, workflow_format="yaml")
client.jobs.list(status="completed", limit=20)
client.jobs.get(job_id)
client.jobs.progress(job_id)
client.jobs.result(job_id)
client.jobs.inputs(job_id)
client.jobs.cancel(job_id)
client.jobs.artifact(job_id, path="s3://...", output="result.json")

# Block until a terminal state and return the final job record.
client.jobs.wait(job_id, timeout=900.0)

# Yield one snapshot per poll until the job is terminal.
for snapshot in client.jobs.watch(job_id):
    print(snapshot["status"], snapshot["progress"])
```

### Pagination — `list_all`

`Jobs`, `Workers`, and `Traces` all expose `list_all(...)`; the async versions return async iterators. The iterator handles cursor traversal internally; pass `page_size=` to bound the per-request `limit`, and the iterator stops when the server stops returning a `next_cursor`.

```python
for job in client.jobs.list_all(status="completed", page_size=50):
    process(job)
```

```python
async for job in async_client.jobs.list_all():
    await process(job)
```

## Timeouts

The default HTTP timeout is **300 seconds**, set in `lumilake._base_client.DEFAULT_TIMEOUT`. Override it three ways:

1. **Client default** — pass `timeout=` to `LumilakeClient(...)` or `AsyncLumilakeClient(...)`. Applies to every call the client makes.
2. **Environment** — set `LUMILAKE_TIMEOUT=<seconds>` before constructing the client.
3. **Per call** — every resource method accepts `timeout=<seconds>` (or `request_timeout=` for poll-driven helpers like `wait` and `watch`). Long-running calls like `wait`, `watch`, `result`, and `artifact` are the usual reasons to bump it on a single request.

```python
client.jobs.result(job_id, timeout=900.0)
client.jobs.wait(job_id, timeout=1800.0, request_timeout=60.0)
client.jobs.artifact(job_id, path=..., output=..., timeout=600.0)
```

## Deploy Extra

Without the `deploy` extra, deploy methods except `init` raise `DeployError` with an install hint. Server API resources work without the deploy extra.

Deploy methods call `lumilake_deploy` directly. Async deploy methods dispatch the same Python calls through `asyncio.to_thread` so Docker and FlowMesh setup work does not block the event loop.
