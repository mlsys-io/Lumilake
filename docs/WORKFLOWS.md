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
      type: agent
      description: "Show me {ref_stock}'s last 5 quarters of fundamentals."
      params:
        - label: ref_stock
          node: Stock

outputs:
  - name: result
    ref: "Fundamentals"
```

`inputs` declares workflow-level input names. `ops` declares user-facing operation IDs, operation types, dependencies, and op-specific fields. `outputs` exposes selected op values as named workflow outputs.

Supported YAML op types are `DataOp`, `DataRetrievalOp`, `MessageOp`, `LLMChatOp`, `LLMVisionOp`, `ImageGenerationOp`, `FormatOp`, and `LambdaOp`. `InputOp` and `OutputOp` are generated from the `inputs` and `outputs` blocks.

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

When live sampling is enabled, preflight executes a bounded `LIMIT N` query against the upstream's connection, subject to a TCP connect timeout (`LUMILAKE_DATA_PROFILE_CONNECT_TIMEOUT_S`, default `5 s`) and a Postgres `statement_timeout` (`LUMILAKE_DATA_PROFILE_STATEMENT_TIMEOUT_S`, default `10 s`). The lockdown enforces:

- **Static SELECT-only check**: multi-statement scripts and non-SELECT top-level statements (including inside CTEs) are rejected before any query reaches the database.
- **`READ ONLY` transaction**: writable CTEs and direct DML/DDL are rejected at the Postgres level.

Note: these controls do not prevent side-effecting functions inside `SELECT` — `dblink_exec(...)`, `pg_advisory_lock(N)`, or user-defined volatile functions can still execute with database-level effects. That is why opt-in is required when the connected database has such functions defined.

## Examples

Runnable examples live in `examples/templates/`. Each is shipped as a
YAML + n8n pair; see `docs/E2E_DEMO.md` for the full reproduction
recipe (data plane, demo dataset, deploy).

- `examples/templates/yaml/trading-agent.yaml` + `examples/templates/n8n/trading-agent.json`
- `examples/templates/yaml/agent-retrieval.yaml` + `examples/templates/n8n/agent-retrieval.json`
- `examples/templates/yaml/image-generation.yaml` + `examples/templates/n8n/image-generation.json`

Use `lumilake job preview examples/templates/yaml/trading-agent.yaml --format yaml --input Stock=NVDA` to validate and inspect a YAML schedule without dispatching runtime work. Add `--optimizer <name>` to preview with a specific optimizer instead of the server default.
