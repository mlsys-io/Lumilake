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

## Examples

Runnable examples live in `examples/templates/`:

- `examples/templates/yaml/agent-retrieval-mini.yaml`
- `examples/templates/yaml/image-generation.yaml`
- `examples/templates/n8n/image-generation.json`

Use `uv run lumilake job preview examples/templates/yaml/agent-retrieval-mini.yaml --format yaml --input Stock=AAPL` to validate and inspect a YAML schedule without dispatching runtime work.
