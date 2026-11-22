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

Supported YAML op types are `DataOp`, `DataRetrievalOp`, `MessageOp`, `LLMChatOp`, `LLMCompletionOp`, `LLMVisionOp`, `ImageGenerationOp`, `EmbeddingOp`, `ClassificationOp`, `HttpFetchOp`, `FormatOp`, `LambdaOp`, `ParseOp`, `RouterOp`, `JoinOp`, and `Repeat`. `InputOp` and `OutputOp` are generated from the `inputs` and `outputs` blocks. A `LambdaOp` in expand mode takes `on_empty: drop | fail`, as `docs/OPS.md` describes.

`Repeat` is a fixed-count repeat of an op body. A fixed loop is static, so the
parser unrolls it at submit into ordinary ops — one declared op per iteration —
and no driver round is involved:

```yaml
ops:
  - id: Retry
    op: Repeat
    times: 3
    input: Draft          # id the first iteration reads
    output: Rectify       # body id the next iteration reads
    body:
      - {id: Rectify, op: LLMChatOp, inputs: [$in], messages: [...], config: {...}}
      - {id: Score, op: LambdaOp, inputs: [Rectify], fn_name: score, code: "..."}
```

Iteration `i` (one-based) copies every body entry as an op id `Retry.i.<bodyId>`. Inside the body, `$in` reads `input` on the first iteration and `Retry.<i-1>.<output>` afterwards, and a reference to another body id resolves within the same iteration; any other reference reads an op outside the repeat, another repeat included, by its last iteration. A body may not name its own repeat id. Elsewhere in the workflow, `Retry` refers to `Retry.<times>.<output>`. A repeat's `input` may name another repeat, so two fixed loops can chain. Each unrolled iteration is an ordinary declared op, so seeds, cache keys, the optimizer, and the record store see distinct ids. A `RouterOp` cannot appear in a body.

A `Repeat` is refused at submit, with a 4xx naming it, when `times` is not a positive integer, the entry has an unknown field, the repeat id is already taken outside it, `input` names no known id, `output` is not a body id, a body entry is not a mapping, a body id contains `.` or collides with an id outside the repeat, a body contains another `Repeat` or a `RouterOp` or reads its own repeat id, an op outside the repeat reads a body id directly, `$in` appears outside a body or as `input`/`output`, two repeats name each other in a cycle, or the repeats would unroll more than 1,000 entries in total. The 1,000-entry cap counts every repeat in the workflow together, so splitting one large repeat into two does not raise the allowance.

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

## Per-record store (`store:`)

A workflow can ask the server to store every computed record, so a later read can
name what produced a value. Set `store: true` at the top level of a YAML
document, or beside `graph` in a native payload. It defaults to off, so existing
workflows are unchanged. A dynamic workflow sets it at its top level beside
`type: dynamic`, and then every round writes its records.

A record key is declared once, on the op or input that carries it:
`key: {node, path}` names the op or input that holds the keys, and `path` drills
into it, the same shape a rowwise column uses. Every op downstream derives its
key: a row-mode `LambdaOp` and a one-row-in-one-row-out model op inherit their
single keyed input's key by position. An op with no keyed input, a list-mode op
and a reduce are one whole-op record, and any op that exists only on an unmerged
branch (expand, reduce, join) stays whole-op here.

```yaml
name: keyed
store: true
inputs:
  Cases: ['{"case_id": "c1", "q": "first"}', '{"case_id": "c2", "q": "second"}']
ops:
  - id: Answer
    op: LambdaOp
    inputs: [Cases]
    fn_name: answer
    mode: row
    code: "def answer(c):\n    return c[0]\n"
    key: {node: Cases, path: case_id}
outputs:
  - name: answer
    ref: Answer
```

Each entry is written once, as one object under the archive prefix at
`records/<digest>.json`, and each write also updates the job's index object at
`records/jobs/<job id>.json`, which lists the job's entries by op, key and round.
The digest is a SHA-256 over the op's computation (code, prompt, config, model id
and decoding), the record's address (op, key, semantic round), the records it
read (a keyed record reads the one at its own position; a whole-op record reads
every record of every input, in order) and their values, the per-row seed it drew
with when it declares one, and the effect versions. Per-row data enters as the
record's inputs and a per-row seed as its realization, never through the op's own
spec: that covers a python step's plan entries, every literal rows list a model,
retrieval, join or format op carries in its data spec, and a rowwise seed column.
So two slices of one op at any `input_batch_size` store the same records under
the same keys. The address names the op the workflow declares. Physical node
ids, batch positions, job ids and workers never enter the digest.

The round is the semantic one: one-based for a round of a dynamic job, `0` for a
standalone job. A reader trusts an object only when the digest recorded inside it
equals its key. A failed record carries the same inputs, realization and versions
a success would, with `status: failed` added, so it never answers a request for a
success and two failures on different inputs never share a key.

An op that reads a retrieval or fetch that records no snapshot is marked
`versions.unversioned`, because such a read is never reusable; a hosted model
call counts as one, since no recorded revision fixes its model. A record whose
value no seed or snapshot fixes, an unversioned one or a sampling op with no
seed, carries a realization unique to its production (the job and task that made
it), so it never shares a key with another run's draw. An unversioned record
always names its production, even when its op declares a seed: the seed stays
beside it as information, since one seed does not promise the same value twice.
A sampling op decodes deterministically only under beam search wider than one, a
zero temperature or `do_sample: false`; an absent temperature samples at the
engine's own default.

Read a record's chain back with `GET /api/v1/jobs/{job_id}/lineage`; see
[`docs/API.md`](API.md).

## Reusing stored records (`reuse:`)

A workflow can keep what an earlier storing run already computed instead of
recomputing it. Set `reuse: true` beside `store: true` at the top level of a
YAML document, or beside `graph` in a native payload; `store: true` is required,
and the flag defaults to off, so an existing workflow is unchanged.

Before dispatch the server keys each op's records exactly as the store wrote
them and asks whether every record of the op is already there. An op is kept
only when all of its records answer; one missing record reruns the whole op.
Every op downstream of a kept op reads that op's stored values instead of
running, and every op downstream of a rerun misses with it. Granularity is the
whole op: an op that carries no record key is one record.

A kept op is not dispatched; its consumers read its stored values in key order.
A request the store answers in full submits nothing and returns the stored
values. A failed record is never served, and neither is one whose recorded
digest does not equal its key.

```yaml
name: revise
store: true
reuse: true
inputs:
  Cases: ['{"case_id": "c1", "q": "first"}']
ops:
  - id: Answer
    op: LambdaOp
    inputs: [Cases]
    fn_name: answer
    mode: row
    code: "def answer(c):\n    return c[0]\n"
    key: {node: Cases, path: case_id}
outputs:
  - name: answer
    ref: Answer
```

A job records one decision per op, `hit` or `miss` with the first reason it
missed; see [`docs/API.md`](API.md).

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
