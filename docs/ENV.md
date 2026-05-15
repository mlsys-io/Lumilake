# Environment Reference

Lumilake reads configuration from `.env` for local deployment and from process environment variables for direct server runs. Generate the template with:

```bash
lumilake deploy -C <deploy-dir> init
```

`-C / --project-dir` (or `LUMILAKE_DEPLOY_DIR`) selects the directory the env files are written to and later read from. Workspace-checkout users may prefix every command with `uv run`; from a PyPI install (`pip install 'lumilake[cli]'`) invoke `lumilake` directly.

To also generate a bundled FlowMesh stack env file:

```bash
lumilake deploy -C <deploy-dir> init --flowmesh
```

Run `lumilake deploy -C <deploy-dir> doctor` after editing `.env`.

## Required Keys

| Key | Purpose |
|-----|---------|
| `LUMILAKE_SERVER_HOST` | Host the FastAPI server binds to. |
| `LUMILAKE_SERVER_PORT` | Port the FastAPI server binds to. |
| `LUMILAKE_RUNTIME_ORCHESTRATOR_URL` | FlowMesh server URL used for workflow dispatch. |
| `S3_ARCHIVE_PREFIX` | `bucket/prefix` where job records and runtime artifacts are archived. |
| `LUMILAKE_IMAGE_TAG` | Docker image tag used by local deployment. Defaults to `dev` (the rolling main-built image). Pin to a `vX.Y.Z` semver tag for production. |
| `LUMILAKE_REGISTRY` | Container registry the deploy CLI pulls the server image from. Defaults to `ghcr.io/mlsys-io`. **Trust-bearing**: setting this points `lumilake deploy pull` at a different host, so only override it to a registry you control. |

Compute data plane (required for every direct SQL/S3 op):

| Key | Purpose |
|-----|---------|
| `DATABASE_URL` | PostgreSQL connection string used by every SQL `DataRetrievalOp`. |
| `S3_URL` | S3-compatible endpoint and credentials used by every S3 `DataRetrievalOp` and the archive (`S3_ARCHIVE_PREFIX`). |
| `S3_USER_DATA_PREFIX` | `bucket/prefix` for user data objects (used by data-profile listing). |

## Lumid.data routing

Only `DataRetrievalOp` with `type: agent` routes through lumid.data — SQL and S3 retrievals always go direct against `DATABASE_URL` / `S3_URL` regardless of `LUMID_DATA_URL`.

Agent-retrieval keys:

| Key | Purpose |
|-----|---------|
| `LUMID_DATA_URL` | Base URL for lumid.data's `/agent/v1` endpoint. Required when a workflow contains agent retrievals. |
| `LUMID_DATA_TOKEN` | Bearer token sent to lumid.data. |
| `LUMID_DATA_TIMEOUT_SECONDS` | HTTP timeout for lumid.data calls. |

## Scheduler and Runtime Tuning

| Key | Purpose |
|-----|---------|
| `LUMILAKE_OPTIMIZER_TYPE` | Optimizer implementation. Defaults to HALO. |
| `LUMILAKE_OPTIMIZER_BATCH_SIZE` | Maximum batch size used by the optimizer loop. |
| `LUMILAKE_BATCH_ACCUMULATION_SECONDS` | Wait window before forming the first optimization batch. |
| `LUMILAKE_STARVATION_LIMIT` | Starvation threshold; `0` means immediate override. |
| `LUMILAKE_OPTIMIZER_SUBPROCESS_TIMEOUT_SECONDS` | Timeout for optimizer subprocess execution. |
| `LUMILAKE_FLOWMESH_OUTPUT_DESTINATION` | FlowMesh result delivery mode. |

See `.env.example` for the full template and defaults.
