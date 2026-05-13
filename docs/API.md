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
