# End-to-End Demo

This page is the reproduction recipe for the three demo workflow pairs
shipped under `examples/templates/`. It is built around **three
independent steps**; skip any step you've already done.

```text
┌───────────────────────────┐   ┌────────────────────────────┐   ┌──────────────────────────┐
│ Step 1 (optional)         │   │ Step 2 (optional)          │   │ Step 3                   │
│ Bring up lumid-data-app    │──▶│ Load the demo dataset      │──▶│ Deploy lumilake + run    │
│ (catalog + blob store)     │   │ (schema + blob objects)    │   │ workflows                │
└───────────────────────────┘   └────────────────────────────┘   └──────────────────────────┘
   skip if BYO lumid-data-app    skip if BYO data already         everyone does this
```

> **One HTTP boundary.** Step 2's `load_demo_data.py` writes Postgres
> via `pg_restore` directly against the bundled database, but uploads
> blobs through lumid-data-app's `PUT /blobs/<key>` API — the same
> boundary the lumilake server uses. `DATABASE_URL` is a per-script
> seed input; lumilake's own `.env` carries `LUMID_DATA_URL` /
> `LUMID_DATA_TOKEN` instead. See [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)
> for the control-plane overview.

## What you get

| Workflow pair | YAML | n8n | Demonstrates |
|---|---|---|---|
| trading-agent | `examples/templates/yaml/trading-agent.yaml` | `examples/templates/n8n/trading-agent.json` | 5 SQL pulls across `lumilake_demo.*` → bull / bear / risk synthesis → final verdict |
| agent-retrieval | `examples/templates/yaml/agent-retrieval.yaml` | `examples/templates/n8n/agent-retrieval.json` | Five lumid.data agent retrievals → analyst LLMs → research summary (requires lumid.data) |
| image-generation | `examples/templates/yaml/image-generation.yaml` | `examples/templates/n8n/image-generation.json` | SQL + S3 HTML fetch → digest → diffusion prompt → render → critic VLM → refined render |

All three read from the same `lumilake_demo.*` Postgres schema; the
image-generation pair also reads HTML files from
`s3://<bucket>[/<prefix>]/example-data/news/html/`.

## Prerequisites

- Python 3.12 + `uv` (or `pip install "lumilake[cli]"`).
- Docker + Docker Compose v2.
- `pg_dump` / `pg_restore` (PostgreSQL 16+) on `PATH` — used by the
  demo-data scripts.
- Disk: ~1 GB free in `~/.cache/lumilake-demo` for the bundle.
- For agent-retrieval only: a running lumid.data instance and
  `LUMID_DATA_URL` set.

The bundled demo workflows use locally-served open-weight models
(`Qwen/Qwen3-8B` for text, `llava-hf/llava-1.5-7b-hf` for vision,
`Tongyi-MAI/Z-Image-Turbo` for image gen) — no `OPENAI_API_KEY` is
required. Set one only if you author workflows that call hosted
providers.

The `mc` MinIO client is **not** required.

> **Working directory.** Step 1 and Step 2 commands use repo-relative
> paths (`scripts/dev/...`). Run them from a Lumilake source checkout
> root, or pass absolute paths. Step 3 commands use `lumilake deploy
> -C <dir>` and `lumilake job submit <abs-path>` so they work from any
> CWD once `lumilake` is installed.

---

## Step 1 — Bring up the data plane (optional)

If you don't already have a Postgres and S3-compatible store, the repo
ships a Docker Compose file that starts both on `127.0.0.1`. It is
**independent of `lumilake deploy`** — its lifecycle is managed
directly with `docker compose`.

```bash
docker compose -f scripts/dev/compose.data-plane.yml up -d
```

You now have:

| Service  | Endpoint            | Credentials                          |
|----------|---------------------|--------------------------------------|
| Postgres | `127.0.0.1:15432`   | user `lumilake` / pw `lumilake_password` / db `lumilake` |
| MinIO    | `127.0.0.1:19100`   | access `lumilake` / secret `lumilake_password`           |
| MinIO console | `127.0.0.1:19101` | same creds                          |

Bucket `lumilake-demo` is created automatically by a one-shot
`minio-init` container that exits after `mc mb` succeeds.

**Optional lumid.data.** The compose file ships a `lumid-data` service
under the `lumid` profile. Bring it up by exporting the API key first:

```bash
LUMID_DATA_API_KEY=<your-key> \
  docker compose -f scripts/dev/compose.data-plane.yml --profile lumid up -d
```

`LUMID_DATA_IMAGE` overrides the image tag (default
`ghcr.io/mlsys-io/lumid-data:latest`).

**Tear down.** Stop containers but keep data:

```bash
docker compose -f scripts/dev/compose.data-plane.yml down
```

Wipe everything (containers + volumes):

```bash
docker compose -f scripts/dev/compose.data-plane.yml down -v
```

**Skip this step entirely** if you have your own lumid-data-app
instance — just set `LUMID_DATA_URL` (+ optional `LUMID_DATA_TOKEN`),
`S3_DATA_PREFIX`, and `S3_ARCHIVE_PREFIX` in your `.env` to point at
that instance. Nothing in step 2 or 3 reads the bundled defaults if
you've overridden them.

---

## Step 2 — Load the demo dataset (optional)

`scripts/dev/load_demo_data.py` downloads the published bundle from the
`demo-data-v1` release of `mlsys-io/lumilake_OSS`, restores the
`lumilake_demo` schema, and uploads `news/{html,images}` into
`s3://<bucket>[/<prefix>]/example-data/news/` (where `<prefix>` is the
optional sub-path portion of `S3_DATA_PREFIX`, absent for the default
`lumilake-demo` value).

`load_demo_data.py` writes Postgres directly via `pg_restore` and
uploads blobs via lumid-data-app's HTTP API. Pass the seed-time URLs
as CLI flags (or set them in an env file via `--env-file`):

```bash
uv run python scripts/dev/load_demo_data.py \
  --database-url postgresql://lumilake:lumilake_password@127.0.0.1:5102/lumilake \
  --lumid-data-url http://127.0.0.1:5101 \
  --lumid-data-token devkey \
  --blob-prefix lumilake-demo
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--env-file PATH`         | Explicit path to an env file for the seed script (default: search upward from CWD). Independent of `lumilake deploy`'s `.env`. |
| `--database-url URL`      | Postgres connection string the seed script writes to. |
| `--lumid-data-url URL`    | lumid-data-app base URL (e.g. `http://127.0.0.1:5101`). Sets `LUMID_DATA_URL`. |
| `--lumid-data-token TOK`  | Bearer token for lumid-data-app. Sets `LUMID_DATA_TOKEN`; falls back to `LUMILAKE_RUNTIME_TOKEN` if unset. |
| `--blob-prefix PREFIX`    | Logical prefix inside lumid-data-app's blob store (matches the server's `S3_DATA_PREFIX`, e.g. `lumilake-demo`). |
| `--tag demo-data-v2`      | Pull a different bundle release. |
| `--cache-dir PATH`        | Where downloads land (default `~/.cache/lumilake-demo`). |
| `--news-key-prefix PATH`  | Key prefix to write `news/` under (default `example-data`). |
| `--drop-schema`           | `DROP SCHEMA lumilake_demo CASCADE` before restore (destructive — explicit only). |
| `--pg-restore-jobs N`     | `pg_restore -j` value (default 4). |

The dump embeds the schema name `lumilake_demo`; `pg_restore` recreates
it under that name. If you need a different name, rename after restore
with `ALTER SCHEMA lumilake_demo RENAME TO <target>` and update the
workflow templates accordingly.

After it finishes:

- Postgres has `lumilake_demo.{ohlc_10m, financial_income_statement,
  market_metrics, news_metadata, insider_sentiment, instrument_profile}`
  populated.
- S3 has `<bucket>[/<prefix>]/example-data/news/{html,images}/...` (1,512 objects,
  ~682 MiB).

**Skip this step** if your data plane is already seeded with these
tables and objects. Workflows only need the six `lumilake_demo.*`
tables plus the news objects at `example-data/news/html/{id}.html`
keyed off `news_metadata.id`.

### Maintainer side — producing a release bundle

```bash
uv run python scripts/dev/dump_demo_data.py
```

Writes `lumilake-demo-pg.dump` + `lumilake-demo-s3.tar.gz` into
`../demo-data-bundle/` (outside the repo by default). Upload them to a
release tag:

```bash
gh release create demo-data-v1 \
  ../demo-data-bundle/lumilake-demo-pg.dump \
  ../demo-data-bundle/lumilake-demo-s3.tar.gz \
  --notes "Demo dataset for Lumilake e2e workflows."
```

---

## Step 3 — Deploy lumilake and run a workflow

`lumilake deploy` accepts `-C <project-dir>` to locate the `.env` (or
set `LUMILAKE_DEPLOY_DIR=<path>` in your shell). The commands below
work the same whether `lumilake` is installed from PyPI or invoked
from a source checkout via `uv run lumilake ...`.

```bash
mkdir -p ~/lumilake-deploy
lumilake deploy -C ~/lumilake-deploy init --flowmesh     # writes .env + .env.flowmesh
```

`--flowmesh` is required for the bundled FlowMesh stack: without it
`lumilake deploy up` only starts the lumilake-server container and
every job submission fails with `FlowMeshConnectionError` because no
workers are reachable.

The shipped `.env.example` is **pre-pointed at step 1's lumid-data-app**
(`S3_DATA_PREFIX=lumilake-demo` resolved against the bundled instance's
blob store). Open `~/lumilake-deploy/.env` only if you need to:

- Set a model provider key (`OPENAI_API_KEY`, etc.) — only if you
  author workflows that call hosted providers; the bundled demos run
  on local open-weight models.
- Point at your own lumid-data-app (override `LUMID_DATA_URL`,
  `LUMID_DATA_TOKEN`, `S3_DATA_PREFIX`, `S3_ARCHIVE_PREFIX`).
- `LUMID_DATA_URL` is required for **all** retrieval modes (`sql`,
  `s3`, and `agent`) — every `DataRetrievalOp` routes through
  lumid-data-app.
- Set `LUMILAKE_GPU_DEVICES` to one or more free GPU indices on your
  host (default is empty — no GPU workers). On a shared host, pick an
  index that other stacks are not using rather than `"all"`. This is
  distinct from `CUDA_VISIBLE_DEVICES` in `.env.flowmesh`, which only
  scopes what the FlowMesh server container sees.

Then bring up the stack:

```bash
lumilake deploy -C ~/lumilake-deploy pull     # fetch the published server image
lumilake deploy -C ~/lumilake-deploy up
lumilake deploy -C ~/lumilake-deploy doctor   # validate env + connectivity
```

Submit a workflow. `lumilake job submit` reads the workflow file path
from your **current working directory**, so either pass an absolute
path or `cd` to a checkout that has `examples/templates/...`:

```bash
WORKFLOW_DIR=/path/to/lumilake_OSS/examples/templates

lumilake job submit "$WORKFLOW_DIR/yaml/trading-agent.yaml" \
  --format yaml \
  --input 'Stock=NVDA,AAPL,MSFT' \
  --output-prefix demo/trading-agent

# Watch progress.
lumilake job watch <job_id>

# Final verdict.
lumilake job result <job_id>
```

The other workflow pairs follow the same shape — swap the YAML path
(or JSON path + `--format n8n`).

**Image-generation note.** Requires a GPU-equipped FlowMesh worker.
`lumilake deploy up` reads `LUMILAKE_GPU_DEVICES` from `.env`: the
shipped default is blank (no GPU workers). Set it to a free GPU index
on your host (e.g. `"0"`) or a comma-separated subset for partial use.

**Agent-retrieval note.** All retrieval modes (`sql`, `s3`, and
`agent`) route through lumid-data-app — `LUMID_DATA_URL` is required
for every `DataRetrievalOp`, not just agent retrievals.

---

## What's in the dataset

`lumilake_demo` schema (756 news rows across 100 symbols,
Oct 2024–May 2025):

| Table | Used by | Notes |
|---|---|---|
| `lumilake_demo.ohlc_10m` | trading-agent, agent-retrieval | 10-minute candles |
| `lumilake_demo.news_metadata` | all three | news article ids + title + synopsis |
| `lumilake_demo.insider_sentiment` | trading-agent, agent-retrieval | monthly insider sentiment |
| `lumilake_demo.instrument_profile` | trading-agent, agent-retrieval | company / sector / market cap |
| `lumilake_demo.financial_income_statement` | trading-agent, agent-retrieval | quarterly revenue / net income / eps |
| `lumilake_demo.market_metrics` | trading-agent, agent-retrieval | 52w high/low, peTTM, etc. |

S3 bundle layout under `s3://<bucket>[/<prefix>]/example-data/news/`:

- `html/{id}.html` — full article bodies for image-generation.
- `images/{id}.png` — featured images.

`{id}` matches `lumilake_demo.news_metadata.id`; the Postgres and S3
halves are loaded as a unit and versioned together.

---

## Combos at a glance

| You have… | Step 1 | Step 2 | Step 3 |
|---|---|---|---|
| Nothing | run | run | run |
| Own pg + S3, no demo data | skip | run (pass `--database-url` + `--lumid-data-url` / `--lumid-data-token`) | run |
| Own pg + S3 + demo data already loaded | skip | skip | run |
| Bundled data plane + demo data | – | – | run |

---

## Troubleshooting

- **`pg_restore: error: relation already exists`** — re-running the
  loader against a half-populated schema. Pass `--drop-schema` to wipe
  and restore from scratch.
- **News-related ops fail with `Object not found`** — the blob upload
  didn't land under `<blob-prefix>/example-data/news/`. Re-run with
  `--news-key-prefix example-data` (the default) and verify with
  `curl -H "Authorization: Bearer $TOKEN" "$LUMID_DATA_URL/blobs?prefix=<blob-prefix>/example-data/news/"`.
- **`Waiting for worker group` hang** — no FlowMesh worker matches the
  workflow's hardware requirements. Confirm with `lumilake worker list`
  that at least one CPU worker is registered; image-generation also
  needs a GPU worker.
- **`unresolved placeholder` on `{symbol}`** — empty input list. Pass
  `--input 'Stock=NVDA,AAPL,MSFT'` (the YAML's `inputs: Stock: []` is a
  template slot, not a default value).
- **Port collision at `lumilake deploy up`** — the deploy CLI runs a
  pre-flight check and prints the conflicting port and its role. Free
  the port on the host (e.g. tear down a competing stack), or pick a
  free value for `LUMILAKE_SERVER_PORT` in `.env` (server) and
  `SERVER_HTTP_PORT` / `SERVER_GRPC_PORT` / `REDIS_CONTROL_PORT` /
  `REDIS_TELEMETRY_PORT` in `.env.flowmesh` (orchestrator). If you
  change `SERVER_HTTP_PORT`, update `LUMILAKE_RUNTIME_ORCHESTRATOR_URL`
  in `.env` and `FLOWMESH_BASE_URL` in `.env.flowmesh` to match.