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

Direct data-plane mode also requires:

| Key | Purpose |
|-----|---------|
| `DATABASE_URL` | PostgreSQL connection string for SQL/data operations. |
| `S3_URL` | S3-compatible endpoint and credentials. |
| `S3_USER_DATA_PREFIX` | Prefix for user data objects. |

## Data-Plane Modes

Set `LUMID_DATA_URL` to route SQL and storage operations through lumid.data. In that mode, Lumilake does not need direct `DATABASE_URL`, `S3_URL`, or `S3_USER_DATA_PREFIX` values.

Leave `LUMID_DATA_URL` empty to use direct PostgreSQL and S3-compatible services.

Optional lumid.data keys:

| Key | Purpose |
|-----|---------|
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
