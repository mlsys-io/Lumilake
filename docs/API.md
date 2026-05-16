# API Overview

The Lumilake server is a FastAPI app. A local deployment listens on `http://127.0.0.1:9000` by default.

| Endpoint | Purpose |
|----------|---------|
| `GET /healthz` | Server health check. |
| `GET /docs` | Interactive API browser. |

Versioned API routes live under `/api/v1`.

## Jobs

| Route | Purpose |
|-------|---------|
| `POST /api/v1/jobs/preview` | Compile workflow specs and generate an optimizer schedule without dispatching runtime work. |
| `POST /api/v1/jobs` | Submit one or more workflows. |
| `GET /api/v1/jobs` | List jobs visible to the caller. |
| `GET /api/v1/jobs/{job_id}` | Fetch one job. |
| `POST /api/v1/jobs/{job_id}/cancel` | Cancel a job and its runtime request. |
| `GET /api/v1/jobs/{job_id}/progress` | Fetch job progress details. |
| `GET /api/v1/jobs/{job_id}/result` | Fetch the stored result for a completed job. |
| `GET /api/v1/jobs/{job_id}/inputs` | Fetch the resolved job inputs. |
| `GET /api/v1/jobs/{job_id}/artifact?path=...` | Download a stored artifact referenced by the job result. |

## Workers

| Route | Purpose |
|-------|---------|
| `GET /api/v1/workers` | List runtime workers. |
| `GET /api/v1/workers/{worker_id}` | Fetch one runtime worker. |

## Traces

| Route | Purpose |
|-------|---------|
| `GET /api/v1/trace` | List execution traces. |
| `GET /api/v1/trace/{exec_id}` | Fetch one execution trace. |

## Response Shape

Most API responses use an envelope:

```json
{
  "ok": true,
  "data": {}
}
```

Errors use normal HTTP status codes with a JSON error body. The Python SDK unwraps successful envelopes and raises typed exceptions for HTTP failures.

### Error Status Codes

- `400 Bad Request` — malformed request syntax (the JSON or YAML body could not be parsed).
- `404 Not Found` — the resource is missing.
- `409 Conflict` — the resource is in a state that cannot satisfy the request; the cancel endpoint includes the job's terminal `status` (`completed` / `failed` / `cancelled`) in `detail`.
- `422 Unprocessable Entity` — the payload was syntactically valid but failed validation (schema, header value, batch-size, input shape, unresolved location, duplicate graph name, graph compilation, ...).
- `5xx` — server-side faults (DB unavailable, runtime backend error).

The `X-Request-ID` response header echoes the request trace id (either the inbound `X-Request-ID` header or a freshly minted `req-<uid>` token); the same id is attached to every server log record produced under that request.
