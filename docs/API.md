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
| `POST /api/v1/jobs/preview` | Compile workflow specs and generate an optimizer schedule without dispatching runtime work. Accepts an optional `optimizer` field to select the optimizer for this preview; omitted falls back to `LUMILAKE_DEFAULT_OPTIMIZER`. |
| `POST /api/v1/jobs` | Submit one or more workflows. Accepts an optional `optimizer` field to select the optimizer for this job; omitted falls back to `LUMILAKE_DEFAULT_OPTIMIZER`. Use `GET /api/v1/optimizer` to enumerate valid optimizer names. |
| `GET /api/v1/jobs` | List jobs visible to the caller. |
| `GET /api/v1/jobs/{job_id}` | Fetch one job. |
| `POST /api/v1/jobs/{job_id}/cancel` | Cancel a job and its runtime request. |
| `GET /api/v1/jobs/{job_id}/progress` | Fetch job progress details. |
| `GET /api/v1/jobs/{job_id}/result` | Fetch the stored result for a completed job. |
| `GET /api/v1/jobs/{job_id}/inputs` | Fetch the resolved job inputs. |
| `GET /api/v1/jobs/{job_id}/artifact?path=...` | Download a stored artifact referenced by the job result. |
| `GET /api/v1/jobs/{job_id}/workflows` | List FlowMesh workflows associated with the job (one per execution batch). |
| `GET /api/v1/jobs/{job_id}/workflows/{workflow_id}/logs?limit&before&after` | Fetch one page of logs for a job's FlowMesh workflow. |
| `GET /api/v1/jobs/{job_id}/workflows/{workflow_id}/logs/stream?cursor` | Stream logs for a job's FlowMesh workflow as SSE. |
| `GET /api/v1/jobs/{job_id}/workflows/{workflow_id}/logs/download` | Download per-task archived logs as a tar archive (`application/x-tar`). |

## Workers

| Route | Purpose |
|-------|---------|
| `GET /api/v1/workers` | List runtime workers. |
| `GET /api/v1/workers/{worker_id}` | Fetch one runtime worker. |

## Optimizer

| Route | Auth / Permission | Request body | Response body | Purpose |
|-------|------------------|--------------|---------------|---------|
| `GET /api/v1/optimizer` | Bearer token | none | `{"types": ["halo", ...]}` (`OptimizerListResponse`) | List all optimizer type names available on this server — both locally registered and advertised by installed `OptimizerProvider` plugins. Names are lowercased. Use this to determine valid values for the `optimizer` field on job or preview submission. |
| `POST /api/v1/optimizer/schedule` | Bearer token + JOB:WRITE + submission guards | `ScheduleRequest` (`graph`, `worker_names`, `worker_profiles`, `data_profile_results?`, `optimizer_type`) | `ScheduleResponse` (`worker_assignment`) | Generate an optimizer schedule for a supplied runtime graph. Used by lumilake-as-remote-optimizer scenarios where an external Lumilake instance acts as the scheduling backend. |

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
