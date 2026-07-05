# Workflow Formats

Lumilake accepts three workflow input formats:

- **YAML** - a concise Lumilake-authored format for examples and hand-written workflows.
- **Native graph JSON** - serialized Lumilake graph payloads.
- **n8n JSON** - exported n8n workflows parsed by the server.

Pass the format with the CLI `--format` option or the `Workflow-Format` HTTP header when submitting directly to the API.

## YAML Structure

A YAML workflow is a top-level mapping with optional `name`, optional `inputs`, required `ops`, and optional `outputs`.

```yaml
name: example
inputs:
  Stock: []

ops:
  - id: "Fundamentals"
    op: DataRetrievalOp
    inputs: [Stock]
    data_spec:
      type: lumid
      mode: agent
      description: "Show me {ref_stock}'s last 5 quarters of fundamentals."
      params:
        - label: ref_stock
          node: Stock

outputs:
  - name: result
    ref: "Fundamentals"
```

`inputs` declares workflow-level input names. `ops` declares user-facing operation IDs, operation types, dependencies, and op-specific fields. `outputs` exposes selected op values as named workflow outputs.

Supported YAML op types are `DataOp`, `DataRetrievalOp`, `MessageOp`, `LLMChatOp`, `LLMVisionOp`, `ImageGenerationOp`, `EmbeddingOp`, `FormatOp`, and `LambdaOp`. `InputOp` and `OutputOp` are generated from the `inputs` and `outputs` blocks.

## Output projection (`path:`)

`outputs[]` entries accept an optional `path:` selector that picks which field
of the upstream op's result items to emit. The shape is `items.<field>` for a
single-level field, or `items.<field>.<sub>...` for nested traversal:

```yaml
outputs:
  - name: ohlc_sql
    ref: "OHLC via SQL"
    path: items.table        # SQL retrieval — emit the rows table
  - name: news_html
    ref: "News HTML via S3"
    path: items.content      # S3 retrieval — emit the raw bytes/text
  - name: ohlc_agent
    ref: "OHLC via Agent"
    path: items.table        # agent retrieval — replays a generated plan and emits rows
  - name: prompt
    ref: "Summary"
    path: items.metadata.prompt   # nested traversal works for dict-shaped fields
```

When omitted, the path defaults are mode-derived:
`items.table` for `mode: sql`, `items.content` for `mode: s3`,
`items.table` for `mode: agent` (the agent replays a generated SQL/retrieval
plan and emits rows), and `items.output` for LLM-op outputs.

For DataFrame-shaped fields (e.g. `items.table` from SQL retrievals which
arrive as JSON-encoded DataFrames), the walker decodes the string once and
continues traversal — so `items.table.symbol` projects the `symbol` column.

> **n8n note.** n8n-imported workflows do **not** support the `path:`
> selector — n8n's UI has no equivalent annotation. They fall back to the
> mode-derived default automatically, which covers the common
> case. To override the projection on an n8n-imported workflow, edit the
> compiled native graph or re-export as YAML.

## References

Operation `inputs` reference workflow input names or other op IDs. Message bodies can reference another op by using its bare ID as `content`; the parser inserts the formatting node needed to wire that value into the message. For multi-value templating, use an explicit `FormatOp`.

Data specs can reference upstream values through `params` entries:

```yaml
params:
  - label: id
    node: "News Retrieval - SQL"
    path: items.table.id
```

For `DataRetrievalOp` placeholders whose upstream is another `DataRetrievalOp`, set `sample_value` in the upstream `data_spec` to supply a representative value for data-profile preflight without issuing a live sample query. Use a scalar for single-column sources, or a `{column: value, ...}` mapping when downstream `path` entries project specific columns.

Live sampling is **off by default**. The recommended path is to set `sample_value` on the upstream `data_spec` — that supplies a representative value at build time with zero live execution. To explicitly opt in to bounded live queries, set `LUMILAKE_DATA_PROFILE_ENABLE_LIVE_SAMPLING=1`. To skip data profiling entirely, set `LUMILAKE_DISABLE_DATA_PROFILE=1`.

When live sampling is enabled, preflight routes a bounded `LIMIT N` query through lumid-data-app (`POST /retrieve`), subject to the shared `LUMID_DATA_TIMEOUT_SECONDS` HTTP timeout. lumid-data-app is responsible for enforcing SELECT-only / read-only semantics and any per-tenant rate or row caps; the Lumilake server does not open database connections directly.

## FlowMesh Version Pairing

When a workflow uses `type: list` data with `s3://` items (typical for VLM / image-generation paths that funnel S3 URLs through a list), Lumilake stamps a `lumid_cfg` field onto the data spec so the FlowMesh worker can retrieve the blobs through lumid-data-app. The FlowMesh worker's `data` mixin must support `lumid_cfg` for this path to function; older workers that only read `s3_cfg` will fail at retrieval. Lumilake logs a warning at stamp time naming this gap. Pair Lumilake with a FlowMesh release that supports `lumid_cfg` on `type: list` items, or restructure the workflow to use a `DataRetrievalOp` (`mode: s3`) which is always routed through the dedicated `type: lumid` connector.

## Examples

Runnable examples live in `examples/templates/`. Each is shipped as a
YAML + n8n pair; see `docs/E2E_DEMO.md` for the full reproduction
recipe (data plane, demo dataset, deploy).

- `examples/templates/yaml/trading-agent.yaml` + `examples/templates/n8n/trading-agent.json`
- `examples/templates/yaml/agent-retrieval.yaml` + `examples/templates/n8n/agent-retrieval.json`
- `examples/templates/yaml/image-generation.yaml` + `examples/templates/n8n/image-generation.json`

Use `lumilake job preview examples/templates/yaml/trading-agent.yaml --format yaml --input Stock=NVDA` to validate and inspect a YAML schedule without dispatching runtime work. Add `--optimizer <name>` to preview with a specific optimizer instead of the server default.
