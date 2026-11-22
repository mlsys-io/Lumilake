# Workflow Formats

Lumilake accepts three workflow input formats:

- **YAML** - a concise Lumilake-authored format for examples and hand-written workflows.
- **Native graph JSON** - serialized Lumilake graph payloads.
- **n8n JSON** - exported n8n workflows parsed by the server.

Pass the format with the CLI `--format` option or the `Workflow-Format` HTTP header when submitting directly to the API.

## YAML Structure

A YAML workflow is a top-level mapping with optional `name`, optional `seed`, optional `inputs`, optional `functions`, required `ops`, and optional `outputs`.

```yaml
name: example
seed: 7
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

YAML reads unquoted `on`, `off`, `yes`, `no` and bare numbers as non-string keys, so the server refuses a workflow carrying one with a 400 naming the key and its path; quote such keys.

`seed` is the job seed, an integer that defaults to 0 and is recorded with the job. A rowwise chat op's keyed `rowwise_seed` derives each row's seed from it, so two jobs that share the seed, the op id, the round and a row's key make the same draw for that row; see [`docs/OPS.md`](OPS.md). A native graph payload takes the same top-level `seed` field. A job carries one seed: workflows submitted together must share it, and submitting them with different seeds is refused.

Supported YAML op types are `DataOp`, `DataRetrievalOp`, `MessageOp`, `LLMChatOp`, `LLMCompletionOp`, `LLMVisionOp`, `ImageGenerationOp`, `EmbeddingOp`, `ClassificationOp`, `HttpFetchOp`, `FormatOp`, `LambdaOp`, `ParseOp`, `RouterOp`, and `JoinOp`. `InputOp` and `OutputOp` are generated from the `inputs` and `outputs` blocks. A `LambdaOp` in expand mode takes `on_empty: drop | fail`, as `docs/OPS.md` describes.

`LLMChatOp` supports output validation and local beam search; see
[`docs/OPS.md`](OPS.md#llmchatop-output-enforcement-and-beam-search).

`ParseOp` takes `format`, `pattern`, `require`, `flags` and `on_fail`; see
[`docs/OPS.md`](OPS.md#parseop).

### Shared functions (`functions:` and `uses:`)

A helper several `LambdaOp`s call is written once under the top-level
`functions:` mapping, keyed by the name an op names in `uses:`. Each entry is
Python source: one or more top-level `def`s, classes, imports, assignments or
compound statements.

```yaml
name: shared_helpers
functions:
  seed:
    |
    def seed_value(row):
        return PREFIX + row["name"]
    PREFIX = "seed:"

inputs:
  rows: ['{"name": "a"}']

ops:
  - id: first
    op: LambdaOp
    inputs: [rows]
    fn_name: first
    uses: [seed]
    code: |
      def first(inputs):
          return seed_value({"name": inputs[0]})
```

The listed sources are appended after the op's own `code` in the listed order,
separated by a blank line, and everything after that sees the combined
program: dispatch and every digest treat it exactly as if the source had been
written in the op. A block is included once per op that uses it.

A `LambdaOp` that an `LLMChatOp` reads is evaluated inline in the FlowMesh
worker, whose namespace gives a module-level name no scope. Its `def` must be
the last statement of its source, with only a module docstring before or after
it, so a shared block cannot reach it: the block's statement follows the
`def`, submission is refused naming the binding, and the code must move what
it needs inside the function. The same rule refuses a trailing helper,
constant or other statement written into the op's own `code`, whatever the
workflow format. A `LambdaOp` that is a workflow output, or that only other
`LambdaOp`s read, runs as its own `python` task and can use blocks freely.

`code` is optional beside `uses`. Without it the op's code is the joined
blocks, which is how a block can hold the whole function the op runs. `uses`
needs a `def`: it cannot be combined with `code` that is a lambda expression,
because the block would then be part of that expression rather than a
module-level helper.

Submission is refused, naming the op and the function, when `uses` names a
function `functions:` does not define, names one twice, sits on a non-`LambdaOp`
op, or when a `functions:` entry does not parse. It is also refused when a
name is defined by both a used block and the op's own code, or by two used
blocks: the op must end up with a single definition per name. An import that
binds the same module under the same name in several places is not such a
collision.

`functions:` and `uses:` are YAML-only. A native payload carries resolved code
already and an n8n workflow has no equivalent annotation. They apply to a
static YAML workflow: a dynamic workflow's library and template entries are
built without a `functions:` block, so a `uses` there names an unknown
function.

## n8n code-node notes

An n8n **Code** node (`n8n-nodes-base.code`) becomes a `LambdaOp`. Its
`parameters.pythonCode` is the function source, and its `notes` field carries
Lumilake annotations as a JSON object with kebab-case keys:

| Notes key | Type | Meaning |
|---|---|---|
| `inputs` | list of node names | Upstreams; defaults to the node's main incoming edges. |
| `is-output` | bool | Emit this node's value as a workflow output. |
| `output-field` | string | Output name when `is-output` is set; defaults to the node name. |
| `mode` | string | `row` (default), `list`, `aligned`, `expand` or `reduce`, as YAML `LambdaOp` `mode`. |
| `by` | string | Grouping path for `mode: reduce`, as YAML `by`. |
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

A `sql` retrieval may name each emitted row's record key as
`key: {column: <name>}` inside its `data_spec`; a consumer then reads the key
at `items.table.<name>`, the same column a `path` reads. See
[`docs/OPS.md`](OPS.md#dataretrievalop) for the declaration and the checks
that run when each task's result arrives.

For `DataRetrievalOp` placeholders whose upstream is another `DataRetrievalOp`, set `sample_value` in the upstream `data_spec` to supply a representative value for data-profile preflight without issuing a live sample query. Use a scalar for single-column sources, or a `{column: value, ...}` mapping when downstream `path` entries project specific columns. For a `LambdaOp` upstream, set `sample_value` on the op (or the Code node's `sample-value` note) the same way — preflight never executes user code, so it needs a representative value to resolve the placeholder.

An `aggregate_table` or `rowwise_columns` column may reference a workflow input or a `DataOp` by `node` with an empty `path` (`path: ""`, or no `path` at all, or, for a `DataOp`, `items.output`), in which case the referenced literal values are inlined as the column's list rather than read from a runtime envelope.

A `sql`-mode retrieval must define its row order: end the query with a top-level `ORDER BY` that breaks ties on a unique column, or declare `rows: one` for a lookup that returns at most one row per parameter row. Submit rejects a query with neither, naming the op. A retrieval whose whole result is one value declares `rows: table` instead: each parameter row emits one record holding the whole table, in whatever order the query returns it, and an empty result is one record holding an empty table. See [`docs/OPS.md`](OPS.md) for the full rule.

Live sampling is **off by default**. The recommended path is to set `sample_value` on the upstream `data_spec` — that supplies a representative value at build time with zero live execution. To explicitly opt in to bounded live queries, set `LUMILAKE_DATA_PROFILE_ENABLE_LIVE_SAMPLING=1`. To skip data profiling entirely, set `LUMILAKE_DISABLE_DATA_PROFILE=1`.

When live sampling is enabled, preflight routes a bounded `LIMIT N` query through lumid-data-app (`POST /retrieve`), subject to the shared `LUMID_DATA_TIMEOUT_SECONDS` HTTP timeout. lumid-data-app is responsible for enforcing SELECT-only / read-only semantics and any per-tenant rate or row caps; the Lumilake server does not open database connections directly.

## Examples

Runnable examples live in `examples/templates/`. Each is shipped as a
YAML + n8n pair; see `docs/E2E_DEMO.md` for the full reproduction
recipe (data plane, demo dataset, deploy).

- `examples/templates/yaml/trading-agent.yaml` + `examples/templates/n8n/trading-agent.json`
- `examples/templates/yaml/agent-retrieval.yaml` + `examples/templates/n8n/agent-retrieval.json`
- `examples/templates/yaml/image-generation.yaml` + `examples/templates/n8n/image-generation.json`

Use `lumilake job preview examples/templates/yaml/trading-agent.yaml --format yaml --input Stock=NVDA` to validate and inspect a YAML schedule without dispatching runtime work. Add `--optimizer <name>` to preview with a specific optimizer instead of the server default.
