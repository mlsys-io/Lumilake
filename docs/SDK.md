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
base_url = "https://lumilake.example.com"
api_key  = "lm_pat_..."  # optional; falls back to LUMILAKE_API_KEY env or empty
```

### What writes it

`lumilake deploy up` writes the file with the local stack URL once the
stack starts. Remote / hosted users should pass `base_url=` explicitly
or set `LUMILAKE_BASE_URL`.

### How clients resolve

`from_config(path=...)` reads a specific config file directly; use it
for tests or non-default installs. The default constructor resolves
`base_url` (and `api_key`) through the precedence below.

### Authentication

Both clients accept `api_key=` and resolve it together with `base_url`
through a single `resolve_config()` call. Precedence:

1. Explicit `base_url=` / `api_key=` constructor args.
2. `LUMILAKE_BASE_URL` / `LUMILAKE_API_KEY` environment variables.
3. Saved `~/.lumilake/config.toml` (written by `lumilake init`).
4. Defaults: `base_url=http://127.0.0.1:9000`, `api_key=None`.

When `api_key` resolves to a value, the SDK sends it as
`Authorization: Bearer <api_key>` on every request. Deployments running
with `LUMILAKE_REQUIRE_IDENTITY_PROVIDER=1` reject requests without it.

```python
from lumilake import LumilakeClient

with LumilakeClient(api_key="lm_pat_live_…") as client:
    client.jobs.list()
```

## Resources

| Surface | Sync | Async |
|---------|------|-------|
| Health | `client.health()` | `await client.health()` |
| Deploy | `client.deploy.<verb>(...)` | `await client.deploy.<verb>(...)` |
| Jobs | `client.jobs.submit / preview / list / list_all / get / progress / result / inputs / artifact / list_workflows / get_logs / stream_logs / download_logs / cancel / wait / watch(...)` | same, await |
| Workers | `client.workers.list / list_all / get(...)` | same, await |
| Traces | `client.traces.list / list_all / get(...)` | same, await |

### Jobs

All `Jobs` / `AsyncJobs` methods mirror the CLI surface and the server's HTTP routes one-to-one:

```python
client.jobs.submit({"data": [...]}, workflow_format="yaml")
# Override the server default optimizer for one job (see GET /api/v1/optimizer):
client.jobs.submit({"data": [...], "optimizer": "halo"}, workflow_format="yaml")
client.jobs.preview({"data": [...]}, workflow_format="yaml")
# Preview with a specific optimizer — or omit "optimizer" to use the server default:
client.jobs.preview({"data": [...], "optimizer": "topological-sort"}, workflow_format="yaml")
client.jobs.list(status="completed", limit=20)
client.jobs.get(job_id)
client.jobs.progress(job_id)
client.jobs.result(job_id)
client.jobs.inputs(job_id)
client.jobs.cancel(job_id)
client.jobs.artifact(job_id, path="s3://...", output="result.json")

# Per-workflow FlowMesh logs (mirrors `lumilake job logs show/stream/download`).
workflows = client.jobs.list_workflows(job_id)
page = client.jobs.get_logs(job_id, workflows[0].workflow_id, limit=200)  # LogQueryResponse
for entry in client.jobs.stream_logs(job_id, workflows[0].workflow_id):  # Iterator[LogEntry]
    print(entry.event.message)
paths = client.jobs.download_logs(job_id, workflows[0].workflow_id, Path("./logs"))  # list[Path]

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
