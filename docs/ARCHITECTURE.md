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
| S3 or lumid.data  |
+-------------------+
```

## Main Packages

| Package | Role |
|---------|------|
| `lumilake.server` | FastAPI application and route wiring. |
| `lumilake.runtime` | Graph compilation, optimizer loop, job manager, and FlowMesh runtime manager. |
| `lumilake.ops` | Logical operators used by native and YAML workflows. |
| `lumilake.deploy` | Local Docker/FlowMesh deployment helpers. |
| `lumilake.cli` | Typer-based command line interface. |
| `lumilake.sdk` | Sync and async Python clients. |
| `lumilake_hook` | Lumilake resource-kind helpers for shared hook integrations. |

## Runtime Flow

1. A client submits a native graph, YAML workflow, or n8n workflow.
2. The server resolves inputs from inline values, SQL queries, or S3 prefixes.
3. The parser builds Lumilake runtime graph specs.
4. HALO optimizes the runtime graph and assigns work to available FlowMesh workers.
5. The runtime manager dispatches the plan to FlowMesh.
6. Progress, results, traces, and artifacts are stored through the archive layer.

## Storage Model

Lumilake separates compute data from job archive data. Direct mode reads and writes compute data through configured PostgreSQL and S3-compatible services. lumid.data mode forwards SQL and storage operations to lumid.data instead. Job records and runtime artifacts use `S3_ARCHIVE_PREFIX` in both modes, with lumid.data handling the storage calls when configured.
