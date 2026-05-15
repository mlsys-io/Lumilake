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
| `S3_CERT_FILE` | Path to an S3 TLS cert bundle. Optional; the bundled compose mounts the file when set. |

## Lumid.data routing

Only `DataRetrievalOp` with `type: agent` routes through lumid.data — SQL and S3 retrievals always go direct against `DATABASE_URL` / `S3_URL` regardless of `LUMID_DATA_URL`.

Agent-retrieval keys:

| Key | Purpose |
|-----|---------|
| `LUMID_DATA_URL` | Base URL for lumid.data's `/agent/v1` endpoint. Required when a workflow contains agent retrievals. |
| `LUMID_DATA_TOKEN` | Bearer token sent to lumid.data. |
| `LUMID_DATA_TIMEOUT_SECONDS` | HTTP timeout for lumid.data calls. Defaults to `30`. |

## Server and Logging

| Key | Purpose |
|-----|---------|
| `LUMILAKE_LOG_LEVEL` | Server log level. Defaults to `INFO`. |
| `LUMILAKE_SKIP_DOTENV_CHECK` | Set to `1` when the server's env is injected directly (Docker) so startup does not require a `.env` file. |
| `LUMILAKE_RUNTIME_TOKEN` | Bearer token sent with FlowMesh runtime requests. Empty by default. |
| `LUMILAKE_RUNTIME_MANAGER_TYPE` | Runtime dispatch backend. `default` or `flowmesh`. |
| `LUMILAKE_JOB_MANAGER_TYPE` | Job manager implementation. Currently only `priority`. |
| `LUMILAKE_HTTP_TIMEOUT_SECONDS` | Outbound HTTP timeout for server-side calls. Defaults to `300`. |

## Scheduler and Runtime Tuning

| Key | Purpose |
|-----|---------|
| `LUMILAKE_OPTIMIZER_TYPE` | Optimizer implementation. Defaults to `halo`. |
| `LUMILAKE_OPTIMIZER_BATCH_SIZE` | Maximum batch size used by the optimizer loop. |
| `LUMILAKE_BATCH_ACCUMULATION_SECONDS` | Wait window before forming the first optimization batch. |
| `LUMILAKE_STARVATION_LIMIT` | Starvation threshold; `0` means immediate override. |
| `LUMILAKE_OPTIMIZER_SUBPROCESS_TIMEOUT_SECONDS` | Timeout for optimizer subprocess execution. |
| `LUMILAKE_QUEUE_QUANTUM_HIGH` | High-priority queue quantum. Defaults to `20`. |
| `LUMILAKE_QUEUE_QUANTUM_MEDIUM` | Medium-priority queue quantum. Defaults to `10`. |
| `LUMILAKE_QUEUE_QUANTUM_LOW` | Low-priority queue quantum. Defaults to `5`. |
| `LUMILAKE_POLL_TIMEOUT_SECONDS` | Overall timeout for runtime polling. Defaults to `inf`. |
| `LUMILAKE_POLL_INTERVAL_SECONDS` | Interval between runtime status polls. Defaults to `5`. |
| `LUMILAKE_FLOWMESH_OUTPUT_DESTINATION` | FlowMesh result delivery mode. `local` (default) or `http`. |

## Worker Groups

| Key | Purpose |
|-----|---------|
| `LUMILAKE_CPU_WORKER_GROUP_SIZE` | Number of CPU workers FlowMesh provisions. |
| `LUMILAKE_GPU_WORKER_GROUP_SIZE` | Number of GPU workers FlowMesh provisions. At least one of CPU / GPU size must be `> 0`. |
| `LUMILAKE_GPU_DEVICES` | GPU device ids Lumilake assigns to FlowMesh workers (one per device). Specific index (`0`), comma-separated subset (`0,2`), or `all` to cover every `nvidia-smi` device. Leave blank to skip GPU worker creation. |

## Plugins

| Key | Purpose |
|-----|---------|
| `LUMILAKE_PLUGINS` | Comma-separated list of plugin module names the server imports at startup. See `docs/PLUGINS.md`. |

## Data Profiling

| Key | Purpose |
|-----|---------|
| `LUMILAKE_DATA_PROFILE_NUM_TEST_QUERIES` | Number of test queries per data profile. Defaults to `1`. |
| `LUMILAKE_S3_PROFILE_COST_PER_FILE` | Cost-model coefficient (per file) for S3 profile estimates. |
| `LUMILAKE_S3_PROFILE_COST_PER_MIB` | Cost-model coefficient (per MiB) for S3 profile estimates. |
| `LUMILAKE_LOCAL_DATA_PROFILE_PLAN_VARIANTS` | Comma-separated planner variants used for local data profiles. Defaults to `default,prefer_index,prefer_seq,prefer_nestloop`. |

## vLLM Backend

| Key | Purpose |
|-----|---------|
| `LUMILAKE_VLLM_MAX_NUM_BATCHED_TOKENS` | vLLM `max_num_batched_tokens`. Defaults to `2048`. |
| `LUMILAKE_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE` | vLLM CUDA-graph capture size. Defaults to `64`. |
| `LUMILAKE_VLLM_GPU_MEMORY_UTILIZATION` | vLLM GPU memory utilization fraction. Defaults to `0.9`. |

## Hardware Requirements

Used by FlowMesh worker placement.

| Key | Purpose |
|-----|---------|
| `HARDWARE_CPU_REQUIREMENT` | CPU cores per worker. Defaults to `8`. |
| `HARDWARE_MEMORY_REQUIREMENT` | Memory per worker. Defaults to `16Gi`. |
| `HARDWARE_GPU_REQUIREMENT` | GPUs per worker. Defaults to `1`. |
| `HARDWARE_GPU_MEMORY_REQUIREMENT` | GPU memory per worker. Defaults to `8Gi`. |

## FlowMesh TLS and Worker Config

Picked up by the bundled FlowMesh deploy helpers.

| Key | Purpose |
|-----|---------|
| `REDIS_TLS_DIR` | Directory holding Redis TLS certs. Empty disables TLS bind-mounts. |
| `SERVER_TLS_DIR` | Directory holding FlowMesh server gRPC TLS certs. Empty disables TLS. |
| `SERVER_WORKER_CONFIG` | Path to a FlowMesh worker config override. |

## SDK / CLI Client

Consumed by the SDK and CLI when talking to a running server.

| Key | Purpose |
|-----|---------|
| `LUMILAKE_BASE_URL` | Override for the saved server URL (see `docs/SDK.md` for the resolution order). |
| `LUMILAKE_TIMEOUT` | Override for the SDK/CLI HTTP timeout in seconds. Values `<= 0` are ignored. |
| `LUMILAKE_DEPLOY_DIR` | Default `--project-dir` for `lumilake deploy`. |

See `.env.example` for the deploy-time template and defaults.
