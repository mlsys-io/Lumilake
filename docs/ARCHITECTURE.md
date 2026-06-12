# Architecture Overview

Lumilake is organized as a small control plane around workflow parsing, scheduling, runtime dispatch, and result archival.

```text
                 +---------------------+
                 | CLI / SDK / HTTP API |
                 +----------+----------+
                            |
                            v
                 +---------------------+
                 | Lumilake FastAPI app |
                 +----------+----------+
                            |
          +-----------------+-----------------+
          |                                   |
          v                                   v
+-------------------+              +--------------------+
| Parser + graph IR  |              | Hooks / plugins     |
| native / YAML / n8n|              | identity, guards,   |
+---------+---------+              | permissions, usage, |
          |                        | registrar, optimizer|
          v                        +--------------------+
+-------------------+
| HALO scheduler    |
| runtime graph plan|
+---------+---------+
          |
          v
+-------------------+        +---------------------------+
| FlowMesh runtime  |<------>| Workers and executors      |
| manager           |        +---------------------------+
+---------+---------+
          |
          v
+-------------------+
| Archive + results |
| lumid-data-app    |
| (S3_ARCHIVE_      |
|     PREFIX)       |
+-------------------+
```

## Main Packages

| Distribution | Module(s) | Role |
|--------------|-----------|------|
| (image only) | `lumilake_server` | FastAPI application, routes, runtime graph, ops, optimizer, job manager, FlowMesh manager. Ships in `ghcr.io/mlsys-io/lumilake_server`; never published to PyPI. |
| `lumilake-sdk` | `lumilake` | Sync and async clients, shared `envs` / `log`. |
| `lumilake-cli` | `lumilake_cli` | Typer-based command line interface. |
| `lumilake-deploy` | `lumilake_deploy` | Local Docker / FlowMesh deployment helpers. |
| `lumilake-hook` | `lumilake_hook` | Lumilake resource-kind helpers for shared hook integrations. |
| `lumilake` | (no code) | Metapackage providing the `[sdk]` / `[cli]` / `[deploy]` / `[hook]` / `[all]` extras. |

## Runtime Flow

1. A client submits a native graph, YAML workflow, or n8n workflow.
2. The server resolves inputs from inline values, SQL queries, or S3 prefixes.
3. The parser builds Lumilake runtime graph specs.
4. HALO optimizes the runtime graph and assigns work to available FlowMesh workers.
5. The runtime manager dispatches the plan to FlowMesh.
6. Progress, results, traces, and artifacts are stored through the archive layer.

## Storage Model

Lumilake separates compute data from job archive data.

- **`DataRetrievalOp`** (all modes — `sql`, `s3`, `agent`) routes through lumid-data-app using the `type: lumid` FlowMesh connector. `LUMID_DATA_URL` is required, plus an effective lumid-data bearer token (`LUMID_DATA_TOKEN` overrides the fallback to `LUMILAKE_RUNTIME_TOKEN`).
- **Data profiling** (EXPLAIN cost estimation via `POST /profile`, S3 object listing via `GET /blobs`, and live sampling via `POST /retrieve`) also routes through lumid-data-app, gated by the same URL + effective-token contract.
- **Archive** (job records, runtime artifacts) is written under `S3_ARCHIVE_PREFIX`, a logical blob-key prefix in lumid-data-app's store. All archive reads and writes route through lumid-data-app using `LUMID_DATA_URL` / `LUMID_DATA_TOKEN`.
- **Server-side input resolution** (resolving `InputOp` SQL/S3 inputs) routes through lumid-data-app. `S3_DATA_PREFIX` is a logical blob-key prefix in lumid-data-app's store used as the base for S3-input key expansion. `DBLocation` output is not supported; submitting a job with a `DBLocation` output returns 422.
