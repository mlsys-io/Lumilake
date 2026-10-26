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

## n8n code-node notes

An n8n **Code** node (`n8n-nodes-base.code`) becomes a `LambdaOp`. Its
`parameters.pythonCode` is the function source, and its `notes` field carries
Lumilake annotations as a JSON object with kebab-case keys:

| Notes key | Type | Meaning |
|---|---|---|
| `inputs` | list of node names | Upstreams; defaults to the node's main incoming edges. |
| `is-output` | bool | Emit this node's value as a workflow output. |
| `output-field` | string | Output name when `is-output` is set; defaults to the node name. |
| `mode` | string | `row` (default), `list` or `aligned`, as YAML `LambdaOp` `mode`. |
| `image` | string | Container image for the standalone `python` task. |
| `timeout-s` | number | Task timeout, as YAML `timeout_s` (at most 3600 with an `image`, else 600). |
| `memory-mb` | integer | Task memory limit, as YAML `memory_mb`. |
| `expected-runtime-s` | number | HALO planning hint: seconds per input row; never enforced. |
| `cache` | bool | Opt-in task caching, passed to FlowMesh as `cache: true` only when set. |
| `sample-value` | any non-null JSON | Representative value for data-profile preflight placeholders; never executed. |

A Code node with an `image` always runs as its own standalone `python` task.
`expected-runtime-s` is planning-only and never reaches the FlowMesh task spec.

In an n8n `chainLlm` prompt, a reference like `{{ $('Node').item.<field> }}`
renders that node's item field (nested traversal works, e.g.
`item.output.<field>` or `item.json.<field>`). A field reference to a row-mode
Code node runs that node as a standalone `python` task and binds the field as a
column; a field reference to a workflow input projects the field server-side;
a field reference to an upstream LLM's structured output binds the field
column. A whole-item reference (`item`, `item.json` or `item.output` with no
field) keeps the item as a whole.

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

The path `items` (exactly, with no field) selects the whole result item rather
than one field of it. Dynamic workflows use it to archive each leaf and each
`export`ed op so later rounds can read any field of the stored item; see
[`docs/SDK.md`](SDK.md#dynamic-workflows).

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

For `DataRetrievalOp` placeholders whose upstream is another `DataRetrievalOp`, set `sample_value` in the upstream `data_spec` to supply a representative value for data-profile preflight without issuing a live sample query. Use a scalar for single-column sources, or a `{column: value, ...}` mapping when downstream `path` entries project specific columns. For a `LambdaOp` upstream, set `sample_value` on the op (or the Code node's `sample-value` note) the same way — preflight never executes user code, so it needs a representative value to resolve the placeholder.

An `aggregate_table` column may reference a workflow input by `node` with an empty `path` (`path: ""`), in which case the input's literal values are inlined as the column's list rather than read from a runtime envelope.

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
