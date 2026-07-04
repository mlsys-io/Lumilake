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
| `LUMILAKE_IMAGE_TAG` | Docker image tag used by local deployment. Defaults to `dev` (the rolling main-built image). Pin to a `vX.Y.Z` semver tag for production. |
| `LUMILAKE_REGISTRY` | Container registry the deploy CLI pulls the server image from. Defaults to `ghcr.io/mlsys-io`. **Trust-bearing**: setting this points `lumilake deploy pull` at a different host, so only override it to a registry you control. |

## Lumid.data routing

All data access routes through lumid-data-app. All `DataRetrievalOp`s — `sql`, `s3`, and `agent` modes — route through lumid-data-app at runtime. Data profiling (EXPLAIN cost estimation via `POST /profile`, S3 object listing via `GET /blobs`, and live sampling via `POST /retrieve`) also routes through lumid-data-app. `LUMID_DATA_URL` is required for any deployment that runs workflows with `DataRetrievalOp`s or has data profiling enabled, and an effective lumid-data bearer token is required: `LUMID_DATA_TOKEN` overrides the fallback to `LUMILAKE_RUNTIME_TOKEN`.

| Key | Purpose |
|-----|---------|
| `LUMID_DATA_URL` | Base URL for the lumid-data-app instance. Required for all `DataRetrievalOp` modes and for data profiling. |
| `LUMID_DATA_TOKEN` | Bearer token sent to lumid-data-app. Optional — falls back to `LUMILAKE_RUNTIME_TOKEN` when unset; the effective bearer (this key OR the fallback) is required for all `DataRetrievalOp` modes and for data profiling. |
| `LUMID_DATA_TIMEOUT_SECONDS` | HTTP timeout for lumid-data-app calls. Defaults to `30`. |
| `S3_DATA_PREFIX` | Logical blob-key prefix in lumid-data-app's store for compute data (workflow inputs/outputs). Used as the base key prefix for S3-input resolution and output writes. |
| `S3_ARCHIVE_PREFIX` | Logical blob-key prefix in lumid-data-app's store for job records and runtime artifacts (the archive layer). Required. |

## Server and Logging

| Key | Purpose |
|-----|---------|
| `LUMILAKE_LOG_LEVEL` | Server log level. Defaults to `INFO`. |
| `LUMILAKE_SKIP_DOTENV_CHECK` | Set to `1` when the server's env is injected directly (Docker) so startup does not require a `.env` file. |
| `LUMILAKE_REQUIRE_IDENTITY_PROVIDER` | When truthy, rejects requests with 503 unless at least one `IdentityProvider` plugin is registered. Recommended for cloud deploys; leave unset locally. |
| `LUMILAKE_RUNTIME_TOKEN` | Scheduler-internal credential for FlowMesh control-plane reads (worker enumeration, profile fetches). Never used to authorize HTTP request handlers. Leave empty for local; in cloud, set to a token whose FlowMesh principal can read worker metadata. Required when `LUMILAKE_REQUIRE_IDENTITY_PROVIDER` is set. |
| `LUMILAKE_RECOVER_IN_FLIGHT_JOBS` | Whether server startup marks pending/running jobs as failed. Default `1` (recommended for single-instance deploys). Set to `0` in HA / multi-instance deploys so a starting standby doesn't clobber jobs the active instance is still running. |
| `LUMILAKE_RUNTIME_MANAGER_TYPE` | Runtime dispatch backend. `default` or `flowmesh`. |
| `LUMILAKE_JOB_MANAGER_TYPE` | Job manager implementation. Currently only `priority`. |
| `LUMILAKE_HTTP_TIMEOUT_SECONDS` | Outbound HTTP timeout for server-side calls. Defaults to `300`. |
| `LUMILAKE_LOG_DOWNLOAD_SPOOL_MAX_MB` | Memory threshold (MiB) for the log-download spool. The tar is built in a `SpooledTemporaryFile`; if the archive exceeds this size the spool spills to disk before the response body is read. Defaults to `16`. |

## Scheduler and Runtime Tuning

| Key | Purpose |
|-----|---------|
| `LUMILAKE_DEFAULT_OPTIMIZER` | Optimizer used when a job omits its own `optimizer` field (`POST /api/v1/jobs`, `POST /api/v1/jobs/preview`, CLI `--optimizer`). Must resolve to a `BaseOptimizer` registered in `OPTIMIZER_TYPES` — the built-ins `halo` and `topological-sort`, plus any types a plugin adds to `OPTIMIZER_TYPES` at install time. Types advertised only via an `OptimizerProvider` (e.g. `lumilake_hook.RemoteOptimizer`) are not eligible at boot and must be selected per job. Defaults to `halo`. |
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
| `LUMILAKE_PLUGIN_DIR` | Host directory bind-mounted to `/app/plugins` (read-only) and added to `PYTHONPATH`. Each `*.py` placed here is importable by the names listed in `LUMILAKE_PLUGINS`. Defaults to `./plugins` relative to the deploy directory. |
| `LUMILAKE_REMOTE_OPTIMIZER_URL` | Base URL of the external schedule-protocol service. Required only when a plugin's `OptimizerProvider` constructs a `RemoteOptimizer`. Leave empty in standard deployments. Must use `https://` in production; `http://` is permitted only for loopback addresses (`localhost`, `127.0.0.1`, `::1`). The remote optimizer receives the caller's lum.id bearer token verbatim via `Authorization: Bearer <token>`. Configure this to point only at a trusted upstream that you operate or that lum.id federates with. Use HTTPS in any production deployment. |

## Data Profiling

| Key | Purpose |
|-----|---------|
| `LUMILAKE_DATA_PROFILE_NUM_TEST_QUERIES` | Number of test queries per data profile. Defaults to `1`. |
| `LUMILAKE_S3_PROFILE_COST_PER_FILE` | Cost-model coefficient (per file) for S3 profile estimates. |
| `LUMILAKE_S3_PROFILE_COST_PER_MIB` | Cost-model coefficient (per MiB) for S3 profile estimates. |
| `LUMILAKE_LOCAL_DATA_PROFILE_PLAN_VARIANTS` | Comma-separated planner variants used for local data profiles. Defaults to `default,prefer_index,prefer_seq,prefer_nestloop`. |
| `LUMILAKE_DISABLE_DATA_PROFILE` | When truthy (`1`/`true`/`yes`/`on`), the server skips inline data-profile task build/run, skips `collect_data_profile` at batch dispatch, and the HALO optimizer falls back to its static cost model (any supplied profile results are dropped). Defaults to off. |
| `LUMILAKE_DATA_PROFILE_ENABLE_LIVE_SAMPLING` | When truthy (`1`/`true`/`yes`/`on`), preflight may issue a bounded `LIMIT N` query via lumid-data-app `POST /retrieve` against an upstream when no `sample_value` is set. Off by default — set `sample_value` on the upstream `data_spec` for zero live-execution surface, or set this var to `1` to opt in to bounded live queries. Requires `LUMID_DATA_URL` and an effective lumid-data bearer (set `LUMID_DATA_TOKEN`, or fall back to `LUMILAKE_RUNTIME_TOKEN`). |

## vLLM Backend

| Key | Purpose |
|-----|---------|
| `LUMILAKE_VLLM_MAX_NUM_BATCHED_TOKENS` | vLLM `max_num_batched_tokens`. Defaults to `2048`. |
| `LUMILAKE_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE` | vLLM CUDA-graph capture size. Defaults to `64`. |
| `LUMILAKE_VLLM_GPU_MEMORY_UTILIZATION` | vLLM GPU memory utilization fraction. Defaults to `0.9`. |
| `LUMILAKE_VLLM_MAX_MODEL_LEN` | vLLM `max_model_len` cap. Defaults to `0` (use the model's native context). Set `> 0` when a model's native context demands more KV cache than fits after weights on smaller-VRAM GPUs (else vLLM engine init fails with insufficient KV cache memory). |

## Hardware Requirements

Used by FlowMesh worker placement as the fallback when a job does not pass an
explicit per-job override (see `lumilake job submit --cpu / --memory / --gpu
/ --gpu-memory / --hardware-json`). Overrides merge field-by-field against
these defaults — unset fields fall back here.

| Key | Purpose |
|-----|---------|
| `HARDWARE_CPU_REQUIREMENT` | CPU cores per worker. Defaults to `8`. |
| `HARDWARE_MEMORY_REQUIREMENT` | Memory per worker. Defaults to `16Gi`. |
| `HARDWARE_GPU_REQUIREMENT` | GPUs per worker. Defaults to `1`. |
| `HARDWARE_GPU_MEMORY_REQUIREMENT` | GPU memory per worker. Defaults to `8Gi`. |

Jobs with different per-job override tuples (the raw ``hardware`` payload —
``None`` is its own partition, ``{"cpu": 8}`` is another, ``{"cpu": 8,
"memory": "16Gi"}`` is another) land in distinct FlowMesh dispatches because
a single FlowMesh task spec carries one hardware tuple. The partition is
keyed on the raw override, not the env-resolved value, so two jobs that
happen to resolve to the same effective spec via env defaults still split.

The worker filter is role-aware: ``cpu`` and ``memory`` filter every worker
(CPU and GPU), but ``gpu`` and ``gpu_memory`` only filter GPU-capable
workers. A mixed CPU+GPU graph that sets ``gpu=1`` therefore still picks up
CPU workers for its CPU ops; ``gpu`` is not a job-wide "GPU workers only"
switch.
Submitting `--gpu 0` against a workflow that contains a GPU op (vLLM /
transformers / diffusers / text-to-image) is rejected with a clear error
before the optimizer runs.

## FlowMesh TLS and Worker Config

Picked up by the bundled FlowMesh deploy helpers.

| Key | Purpose |
|-----|---------|
| `SERVER_HTTP_PORT` | FlowMesh HTTP API port in `.env.flowmesh`. Keep `LUMILAKE_RUNTIME_ORCHESTRATOR_URL` in `.env` and `FLOWMESH_BASE_URL` in `.env.flowmesh` aligned with this port. |
| `SERVER_GRPC_PORT` | FlowMesh gRPC port in `.env.flowmesh`. Change it when a co-tenant FlowMesh stack already owns `50051`. |
| `REDIS_CONTROL_PORT` | FlowMesh control Redis port in `.env.flowmesh`. Change it when another stack already owns `6379`. |
| `REDIS_TELEMETRY_PORT` | FlowMesh telemetry Redis port in `.env.flowmesh`. Change it when another stack already owns `6380`. |
| `FLOWMESH_BASE_URL` | FlowMesh HTTP base URL consumed by FlowMesh workers; should point at `SERVER_HTTP_PORT`. |
| `REDIS_TLS_DIR` | Directory holding Redis TLS certs. Empty disables TLS bind-mounts. |
| `SERVER_TLS_DIR` | Directory holding FlowMesh server gRPC TLS certs. Empty disables TLS. |
| `SERVER_WORKER_CONFIG` | Path to a FlowMesh worker config override. |

`lumilake deploy init --flowmesh` writes `.env.flowmesh`. On hosts that
also run another FlowMesh stack, check the common defaults before
`deploy up`: HTTP `8000`, gRPC `50051`, Redis control `6379`, and Redis
telemetry `6380`.

## SDK / CLI Client

Consumed by the SDK and deploy CLI helpers.

| Key | Purpose |
|-----|---------|
| `LUMILAKE_BASE_URL` | SDK / CLI override for the saved server URL (see `docs/SDK.md` and `docs/CLI.md` for resolution order). |
| `LUMILAKE_API_KEY` | Bearer token sent as `Authorization: Bearer <key>` by the SDK / CLI. Required when the server runs an `IdentityProvider` plugin with `LUMILAKE_REQUIRE_IDENTITY_PROVIDER=1`. |
| `LUMILAKE_TIMEOUT` | SDK / CLI HTTP timeout override in seconds. Values `<= 0` are ignored. |
| `LUMILAKE_DEPLOY_DIR` | Default `--project-dir` for `lumilake deploy`. |

See `.env.example` for the deploy-time template and defaults.
