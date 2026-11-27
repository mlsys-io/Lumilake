# Ops Reference

Lumilake workflows are DAGs of operation classes registered under `lumilake_server.ops`. YAML `op:` values and native graph `_op` values use these class names. The op registry lives inside the Lumilake server image; YAML and native workflow submitters reach it through the HTTP / SDK surface, not by importing the module directly.

| Op | Purpose |
|----|---------|
| `InputOp` | Generated from workflow-level inputs. |
| `OutputOp` | Generated from workflow-level outputs. |
| `DataOp` | Inline static data. |
| `DataRetrievalOp` | Retrieve data through lumid-data-app in `sql`, `s3` or `agent` mode (see below). |
| `MessageOp` | Build role/content message lists for language model calls. |
| `LLMChatOp` | Run text chat generation, including aggregate and row-wise table prompts. An optional `config.api` block routes the op to an external LLM API endpoint instead of a locally-loaded model (see below). |
| `LLMCompletionOp` | Run rendered string prompts through local generation without a chat template. |
| `LLMVisionOp` | Run vision-language generation over image inputs. |
| `ImageGenerationOp` | Generate images from text prompts. |
| `EmbeddingOp` | Embed text with a FlowMesh-served embedding model and return one vector per input text. |
| `ClassificationOp` | Classify query/candidate pairs with a FlowMesh classification task and return the classifier's output per pair. |
| `HttpFetchOp` | Fetch JSON over HTTP once per query row, with bounded timeout, retries, concurrency, and headers. |
| `FormatOp` | Template upstream outputs into strings for downstream ops. |
| `LambdaOp` | Execute a serialized Python function over upstream values. |
| `RouterOp` | Partition one source op's rows into named branches, one ordinary node per branch. |
| `JoinOp` | Merge keyed row columns and 1:1 outputs back onto the rows they came from. |
| `RepeatOp` | Run an op body a fixed number of times; the parser unrolls it into ordinary nodes. |

## YAML Use

YAML workflows declare user-authored ops under `ops:`. `InputOp` and `OutputOp` are generated from the top-level `inputs:` and `outputs:` blocks; do not declare them manually in YAML.

### DataRetrievalOp

`DataRetrievalOp` retrieves data through lumid-data-app in `sql`, `s3` or `agent` mode. Downstream ops pair rows by position: a row step matches each output to the row it read, a join attaches 1:1 outputs by position, and `input_batch_size` splits rows into slices by position. A result in no defined order could pair different rows on each run, so a retrieval defines the order its rows come back in:

- A `sql` retrieval ends its query with a top-level `ORDER BY`, or declares `rows: one` when a lookup returns at most one row per parameter row. Submit refuses a query with neither, naming the op; only the outer query of the last statement counts.
- The `ORDER BY` breaks ties on a column no two returned rows share, such as the primary key. Submit reads the template only, so the author keeps ties unique.
- `rows: one` is checked at run time: a parameter row that returns more than one row fails the job. An n8n Postgres node declares it as `"rows": "one"` in its `notes`.
- A retrieval whose whole result is one value declares `rows: table` instead, for example to put every fetched row into one prompt. Each parameter row emits one record holding the whole table; an empty result is one record with an empty table. The rows inside the table come in the database's order unless the query orders them, so a prompt built from them can change between runs. Submit refuses a path that indexes a position under the result field, naming the consumer, op and path; rename an all-digit column in the query (`AS y2024`). An n8n Postgres node declares it as `"rows": "table"` in its `notes`.
- `rows` is accepted only on a `sql` retrieval; submit rejects it elsewhere, naming the op.
- An `agent` retrieval replays SQL it writes at run time, so its order is the run-time SQL's. An `s3` retrieval reads one key per parameter row, in parameter-row order; a workflow `S3Location` input naming a prefix is expanded to its keys in lexicographic order.

```yaml
ops:
  - id: "OHLC via SQL"
    op: DataRetrievalOp
    inputs: [Stock]
    data_spec:
      type: lumid
      mode: sql
      template: |-
        SELECT symbol, close, timestamp
        FROM demo.fact_ohlc_10m
        WHERE symbol = '{symbol}'
        ORDER BY timestamp DESC
        LIMIT 5
      params:
        - {label: symbol, node: "Stock"}

  - id: "OHLC table"
    op: DataRetrievalOp
    inputs: [Stock]
    data_spec:
      type: lumid
      mode: sql
      rows: table
      template: |-
        SELECT symbol, close, timestamp
        FROM demo.fact_ohlc_10m
        WHERE symbol = '{symbol}'
      params:
        - label: symbol
          node: "Stock"
```

### FormatOp

`FormatOp` interpolates upstream op outputs into a Python `str.format` template. `format_kwargs` maps each `{name}` placeholder in `template` to the user-facing id of an upstream op (or workflow input). Listing the referenced ids in `inputs:` keeps the DAG wiring explicit.

```yaml
inputs:
  Stock: ["NVDA"]

ops:
  - id: "Prompt"
    op: FormatOp
    inputs: [Stock]
    template: "Summarize the latest news for {symbol}."
    format_kwargs:
      symbol: Stock

outputs:
  - name: prompt
    ref: "Prompt"
```

`format_args` is the positional variant: each id resolves to a `{0}`, `{1}`, ... placeholder. Use `format_kwargs` for named placeholders and `format_args` for positional ones; the two may be combined.

### EmbeddingOp

`EmbeddingOp` embeds text with a FlowMesh-served embedding model and returns one vector per input text. `config.model` is the embedding model id; the optional `gpu_memory_utilization` / `tensor_parallel_size` config fields are passed through to the vLLM engine. `input` references the upstream op (or workflow input) whose text is embedded — declare a `DataOp` for literal text, or point at an `InputOp` / upstream op output. Wire the reference into `inputs:` to keep the DAG explicit.

The op dispatches a FlowMesh `embedding` task whose `model.vllm.convert` is `embed`; that routes the task to the vLLM embedding executor. Vectors are returned as an artifact, not inline: the response carries `model`, `embedding_file` (`embeddings.safetensors`, tensor key `embeddings`, shape `[count, dim]` float32, row-aligned to the input texts), and a `usage` block whose `num_requests` / `embedding_dim` give the row count and vector dimension. The op surfaces the archived embedding artifact ref the same way `ImageGenerationOp` surfaces produced images; a consumer that needs raw vectors loads `embeddings.safetensors`.

```yaml
inputs:
  Docs: ["The quick brown fox.", "Lorem ipsum dolor sit amet."]

ops:
  - id: "Embed"
    op: EmbeddingOp
    inputs: [Docs]
    input: Docs
    config:
      model: BAAI/bge-small-en-v1.5

outputs:
  - name: vectors
    ref: "Embed"
```

The `vectors` output is a per-row list — one entry per input text, not an inline list of floats. Every entry carries its `row` index, the `model`, and a reference to the shared `embeddings.safetensors` artifact; row `i` is the embedding of input text `i`. A downstream op therefore receives `slice_length` per-row vector inputs (one embedding per input doc), and a consumer needing raw floats loads `embeddings.safetensors` and indexes by `row`.

### ClassificationOp

`ClassificationOp` classifies query/candidate pairs with a FlowMesh `classification` task and returns one `items[]` entry per pair, carrying the pair's `index` and the classifier's output. `config` is required and supplies `model` and `backend` (`vllm` or `transformers`); the optional `vllm` mapping is passed to the vLLM engine. A vLLM node uses `model.vllm.convert: classify` and reports the head's sigmoid probability; setting `config.vllm.apply_sigmoid: false` reports the raw logit instead. A Transformers node uses its single-label sequence-classification path.

Literal queries and candidates lower to a non-empty `data.items` list of `{text, text_pair}` mappings, the query as `text`. A single literal query broadcasts across the candidates; otherwise the two lists must have equal lengths. When either side comes from an upstream op, Lumilake first runs the internal `pairs` Python step, which broadcasts a single query or rejects unequal lengths, then passes its `value.items` list to the classification task. Empty query or candidate inputs, empty candidate lists, and mismatched literal pair counts of a graph built from YAML raise with the `ClassificationOp` id in the message; the Python constructor raises before an id exists.

```yaml
inputs:
  Query: ["How does retrieval augmentation work?"]
  Candidates: ["retrieval helps ground answers", "unrelated passage"]

ops:
  - id: "Rerank"
    op: ClassificationOp
    inputs: [Query, Candidates]
    query: Query
    candidates: Candidates
    config:
      model: cross-encoder/ms-marco-MiniLM-L6-v2
      backend: vllm

outputs:
  - name: labels
    ref: "Rerank"
```

### HttpFetchOp

`HttpFetchOp` dispatches one FlowMesh `api` request per row of its `query` input. The `{query}` placeholder must appear in `url`, `params`, or a POST `body`; the runtime converts it to FlowMesh's `{{prompt}}` placeholder. The API executor URL-encodes a `{{prompt}}` in a URL and passes `params` to `httpx`, so query values in `params` use the same form encoding as `requests`. `config.headers` carries the request headers, including credentials such as an API key: keep credentials there rather than in `url`, `params`, or `body`, which are archived as written. `config.timeout_sec`, `config.retries` (at most 10), and `config.max_concurrency` bound the request. Headers are sent as written, and every header value is redacted in the archived graph. Each input row produces one output row. `config.request_delay_sec` spaces actual request starts, including retries.

With `config.fail_open: true`, a failed fetch yields an error row whose content is null and whose error carries the failure text, instead of failing the task.

### LLMChatOp output enforcement and beam search

`config.output_enforcement` validates generated text for both API-backed and local `LLMChatOp` runs. Set exactly one of `json_schema` or `text_regex` and `max_attempts` (default `1`, maximum `10`). Retry mode leaves decoding untouched and re-asks only the rows that failed. A local deterministic decode is not retried: beam search and `temperature: 0` produce the same text again, so the vLLM worker runs one attempt and still records the outcome. Constrained mode asks local vLLM inference to constrain decoding to the requested format, so it is not available for API ops.

FlowMesh records each row's attempts and validation outcome in the task result; the Lumilake job output carries the final text only.

`config.num_beams` selects local beam search when it is greater than `1`. The highest-scoring sequence is returned, and API-backed ops reject the setting. `max_tokens` is the per-prompt ceiling. The beam path does not implement `stop`, `stop_token_ids`, `n`, `logprobs`, `min_tokens`, `presence_penalty`, `frequency_penalty`, `bad_words` or `seed`, so a non-default value for any of them fails at dispatch. A `repetition_penalty` other than `1.0` also fails unless `extra_sampling_params.renormalize_logits` is `true`, which ranks beams the way Transformers `generate` does.

```yaml
ops:
  - id: "Ask"
    op: LLMChatOp
    messages:
      - role: user
        content: "Pick the best answer."
    config:
      model: Qwen/Qwen3-8B
      max_tokens: 2048
      num_beams: 3
      repetition_penalty: 1.1
      extra_sampling_params:
        renormalize_logits: true
```

### LLMChatOp via external API

By default `LLMChatOp` runs against a locally-loaded model (vLLM / transformers) on a FlowMesh worker. Setting `config.api` routes the op to an external LLM API endpoint instead: the runtime builds a FlowMesh `api` task whose body is an OpenAI-style `{model, messages, ...samplers}` chat completion request, and the worker's `api_executor` performs the HTTP call.

```yaml
ops:
  - id: "Ask"
    op: LLMChatOp
    messages:
      - role: user
        content: "Summarize the latest news for {symbol}."
    config:
      model: Qwen/Qwen2.5-0.5B-Instruct
      api: {}
      max_tokens: 256
```

`config.api.url` is optional and defaults to `LUMILAKE_API_DEFAULT_URL`, whose own default is the serving endpoint `https://lum.id/llm/v1/chat/completions`. An operator can point every API-mode op that leaves the URL unset at a local OpenAI-compatible endpoint by setting `LUMILAKE_API_DEFAULT_URL`; an explicit `config.api.url` always wins over it. `config.model` is required in both local and API mode — `config.api` is a backend switch and does not relax the model requirement, so the workflow spec reads the same either way. The request model resolves from `config.api.model`, then the top-level `config.model`. Sampler fields on `config` (e.g. `max_tokens`, `temperature`) are merged into the request body.

`config.api.timeout_sec` sets the per-request timeout for the API call, in seconds. When set, it is emitted into the FlowMesh `api` task spec and overrides the executor's default; when unset, no timeout key is emitted and the executor's own default applies. Use it for long-running extractions over large documents, which can exceed the executor's default.

`config.api.concurrency` sets the number of in-flight requests for the API task. It must be a positive integer no greater than 8, the cap FlowMesh sets; when unset, FlowMesh uses 8.

The credential is resolved server-side, not supplied by the caller. If `config.api.url`'s origin is on the trusted-origin allowlist (the default `https://lum.id`, the origin of `LUMILAKE_API_DEFAULT_URL`, and everything listed in `LUMILAKE_API_TRUSTED_ORIGINS`), the server attaches `Authorization: Bearer <LUMILAKE_RUNTIME_TOKEN>` itself; for any other origin the caller must set `config.api.authorization` explicitly, or the request is rejected before dispatch. This header does become part of the FlowMesh task spec that is actually submitted for execution — it is not kept out of the spec — but it is redacted (replaced with `***REDACTED***`) before the job is archived or an error body is logged.

A message may also reference an upstream node's runtime output — the same way the local backend does, via `inputs:` plus that op's id in `messages:` (directly for a non-`LLMChatOp` upstream such as `DataRetrievalOp`, or through `FormatOp` when relaying another `LLMChatOp`'s output) — and API mode is not restricted to literal, build-time content. Such a reference renders as a FlowMesh `${node.path}` dispatch-time placeholder instead of a literal value: FlowMesh's dispatcher resolves it against the referenced node's real result immediately before the `api` executor runs, and the referenced node is added to this op's FlowMesh dependencies so dispatch waits for it. An upstream `LLMChatOp` (local or API-backed) renders as its text output; an upstream `DataRetrievalOp` renders per its mode (e.g. `sql` renders the retrieved table). A reference is always single-valued: an upstream `LLMChatOp` that produces multiple rows — whether by fanning out into several row-aligned nodes (see below) or by emitting several rows from a single node (a local rowwise op) — cannot be referenced this way; building the graph rejects it up front, since only this op's own message columns can carry that per-row alignment. The one exception is an API-mode consumer of an API-mode upstream: both are dispatched as `api` tasks, so the consumer fans out one node per upstream row, each referencing its own upstream row node (`${<row>.text}`) and declaring every upstream row node as a dependency. This includes an API rowwise upstream, which fans out into one node per row and is consumed per-row like any other fanned API upstream. Any other multi-row shape — a local consumer, a local upstream, a local rowwise upstream, or a `return_history` upstream — still fails closed at build time, because API mode cannot carry per-row alignment (or reconstruct per-row history) for it. An upstream `LLMChatOp` with `return_history` enabled is referenced the same way whether it ran locally or against an API: an API result carries no `metadata.prompt`, so the runtime inlines the literal prior prompt (the user message that was sent) at build time; a runtime-derived prior prompt cannot be reconstructed and fails closed.

When a message's literal content resolves to a multi-row input column (e.g. `Stock: ["NVDA", "AAPL"]`), the op fans out into one FlowMesh `api` task per row instead of one aggregate call: row 0 keeps the op's own id, and row `i` (`i >= 1`) runs as `<op id>__row<i>`, each with that row's value substituted into the message content and each mapped back to the same declared output. Rows dispatch and complete independently, but within a single FlowMesh workflow request the failure semantics are all-or-nothing: if any row's task fails, that workflow request fails and no partial per-row results are returned for it, even for rows that already completed successfully. Lumilake then cancels that FlowMesh workflow, so its tasks still pending or running stop holding workers; the same happens when the request times out or fails for any other reason after FlowMesh accepted it.

This guarantee holds per FlowMesh workflow request, not per job. A job's input rows can be split across multiple `input_batch_size` slices, each dispatched as its own independent workflow request; one slice's failure does not roll back another slice's already-merged results. The job's final response mixes the failed slice's rows (empty output plus an error entry) with the other slice's real per-row values.

A local `LLMChatOp` with a `rowwise_template` runs one generation per input row. When it is the workflow output, its per-row results are returned as one output value per input row (FlowMesh groups them back into one item per table internally; the runtime flattens that back to per-row values at aggregation).

An optional `rowwise_seed` binds a per-row sampling seed. It takes the same `{node, path}` shape as a rowwise column and must resolve to one integer (or null) per row; the runtime passes the column to FlowMesh as `spec.data.seeds`, so each row's generation carries its own seed on a vLLM worker. A null entry keeps the op's own `seed` for that row. `rowwise_seed` requires `rowwise_template`, is rejected with `num_beams > 1` and in API mode, and its producer must emit one integer per row: a `LambdaOp`, a local per-row `LLMChatOp` with `structural_outputs` that holds an integer field, or a literal input. A retrieval, an API LLM, or any op that groups rows emits one list per group instead, a completion op emits text, and a chat op without `structural_outputs` emits text, so all of them are refused at build time. The seed's producer must be listed in the op's `inputs:`; a literal input's values are inlined as the seed list and take no `path`.

```yaml
- id: Answer
  op: LLMChatOp
  inputs: [Prompts, Seeds]
  messages: [{role: user, content: placeholder}]
  rowwise_template: "{prompt}"
  rowwise_columns:
    - {label: prompt, node: Prompts, path: items.output.prompt}
  rowwise_seed: {node: Seeds, path: items.output.seed}
```

`rowwise_seed` also takes a key form, `{key: {node, path}}`, which derives each row's seed from its key: the first 8 bytes of SHA-256 over `[seed, op id, round, key]` as canonical JSON, reduced mod 2^31. The path names dict fields only and a key read under it keeps its JSON type, so `"123"` and `123` differ and a null key fails the job. A key that is the whole cell is read as its text, so an object or array cell is parsed back and a null one is the text `null`, which draws like any other string. The job's top-level `seed` (default 0), the op id as written and the dynamic round number are the other inputs, so two rows with the same key draw the same seed, and the parser generates a row-mode `LambdaOp` whose id a user op cannot hold. The key form needs `rowwise_template`, takes no other `rowwise_seed` field beside `key` and the optional `pairing`, and fails the job naming the op and row on a missing key.

Two rows of one op that carry the same key draw the same seed, so the key form refuses a repeat, naming the op and both row indices; add `pairing: shared` to the key form to allow it. At submit the check covers a key column the submission can see in full, over the whole input set before the job is split into slices, and a dynamic round checks its own key columns the same way when it is built, over the run input and the values it forwards. The generated seed step repeats the check over the rows one task sees, so a column an upstream op produces still fails when that task holds the repeating rows. The round is the round a subgraph runs in, counted from one; the proposer's opening round and the API planner's opening pass run no user op, so no keyed draw takes their number. Renumbering rounds changes every draw, so a rerun or any later coalescing must keep a job's round numbers. A retry of a failed task draws the same seed, and changing the job seed, the op id, the round or the key is a deliberate unpairing. An op that draws `n > 1` samples in its one seeded request draws all n from that request, so the set is reproducible while the individual samples are not separately addressable.

```yaml
- {id: Answer, op: LLMChatOp, inputs: [Cases], rowwise_template: "{question}",
   rowwise_seed: {key: {node: Cases, path: items.output.case_id}}}
```

When every upstream column a per-row chat reads is empty, FlowMesh completes the task with an empty result without dispatching a worker.

A `rowwise_messages` column replaces the rendered prompt with each row's whole conversation: `node` names the op whose output holds one message list per row, and `path` drills into that output (the whole output when omitted). Only `LLMChatOp` accepts it, it takes no `messages`, `system_messages`, `return_history`, `rowwise_template`, `rowwise_columns`, `aggregate_table` or `condition`, and its node must be listed in `inputs:`. The node is a list-mode `LambdaOp`, a data retrieval, or a literal input whose values are JSON message lists and which takes no `path`. Use it when the number of turns varies between rows.

### LLMCompletionOp

`LLMCompletionOp` generates from rendered string prompts without the model chat template. `prompt` references a workflow input, a `LambdaOp`, or a local LLM op that produces one prompt string per row; render a templated prompt with a `LambdaOp`. Use it when the prompt already carries the model's own chat control tokens. Sampling fields are the same as `LLMChatOp`'s. `config.api` and `config.chat_template_kwargs` are not accepted: the task never applies a chat template, so both would have no effect. The same holds for a prompt read from an upstream op: when that op's output column is empty the task completes with an empty result without dispatching a worker.

```yaml
ops:
  - id: Generate
    op: LLMCompletionOp
    inputs: [Prompt]
    prompt: Prompt
    config:
      model: Qwen/Qwen3-8B
      max_tokens: 3000
```

### LLMVisionOp

`LLMVisionOp` runs vision-language generation over images. When it declares a `rowwise_template` and its `image_source` is a `DataRetrievalOp`, the runtime compiles it as a flat per-row VLM: one `flatten` python step renders one prompt per image (row-major across all symbols), one `embedding` task embeds every image, one `inference` task runs the model over the flat prompts, and a `regroup` python step groups the outputs back to one item per symbol. Downstream readers see the VLM's outputs grouped one list per symbol, read at `value.items.output`. The image source defines the per-symbol groups (one prompt per image); any other retrieval column must have the same per-symbol counts.

### LambdaOp

`LambdaOp` runs a Python function you submit as source: `code` holds the source and `fn_name` names the top-level function to call. The source may define imports, constants and helpers before that function; for a `lambda` expression `fn_name` is only a label. The function takes one tuple with a value per entry in `inputs:`, in that order. A `LambdaOp` may name blocks from the workflow's top-level `functions:` map in `uses:`; see [`docs/WORKFLOWS.md`](WORKFLOWS.md#shared-functions-functions-and-uses).

```yaml
inputs:
  Stock: ["NVDA"]

ops:
  - id: "Lowercase"
    op: LambdaOp
    inputs: [Stock]
    fn_name: lowercase
    code: |
      def lowercase(inputs: tuple[str, ...]) -> str:
          (symbol,) = inputs
          return symbol.lower()

outputs:
  - name: lowercased
    ref: "Lowercase"
```

#### Modes

`mode` sets how often the function is called and how its return value becomes rows:

| `mode` | Calls | The function receives | It returns | Output rows |
|---|---|---|---|---|
| `row` (default) | once per row | the row's values | a value (a non-string is archived as its JSON text) | one per input row |
| `aligned` | once per run | each input as its whole list | a list with one value per input row | one per input row |
| `list` | once per run | each input as its whole list | a list of JSON values | one per element |
| `expand` | once per row | the row's values | a list | one per element, in input order; an empty list drops the row |
| `reduce` | once per group | the group's row tuples, in input order | one value | one per group, in order of first appearance |

- `reduce` groups rows by the value at its required `by` dotted path into the first input's value, compared as JSON text (`1` and `"1"` are different keys); `by` is accepted only in reduce mode. A row missing its `by` path fails the job naming the op and the row; an expand call that returns something other than a list fails naming the op.
- Each `expand` item carries its parent's input-row index under `parent` and its own position under `child`, so a downstream op can tell which parent an item came from.
- Every mode decodes each column value before the call: JSON object or array text becomes that object or list, and a serialized table (`{"df": "<json>"}`, or a retrieval item's `table`) becomes its row dicts.
- In `row` and `aligned` mode every multi-row input must carry the same number of rows; submit refuses unequal counts. A one-row input broadcasts to every row.
- An `aligned` function that returns a list of the wrong length fails the task.

`list`, `expand` and `reduce` produce a whole-list result per run, read at `value.items.output`:

- Another op reads it through a node column (`rowwise_columns` / `aggregate_table`); an LLM message chain cannot read it.
- It cannot be split across input slices. Submit refuses an `input_batch_size` that would split such an output, or a row-mode step below one (which keeps one value per element), and names the output and the fix: one input value per job, or an `input_batch_size` that keeps the whole input set in one slice. `aligned` output is one value per row, so it splits like `row` output.
- Only these modes can be a driver template's `stop_when_empty` export, since only their output reads as an empty list.
- An `LLMCompletionOp` prompt, `EmbeddingOp` content or `ImageGenerationOp` content whose list column binds one directly runs once per element and yields one whole-list value; an `aggregate_table` chat answers once per run as usual. A per-element node two or more hops below is not folded, and its several values fail the slice merge.
- A list-mode function that reads a local row-wise `LLMChatOp` receives one value per input row.

#### Failed records in expand and reduce

An expand parent whose row carries a [failed record](#failed-records) skips the function and emits one item with that marker at `child: 0`. `on_empty` (expand only) sets what a parent that returns an empty list emits: `drop` (the default) emits nothing, and `fail` emits one item with the marker `{"_fail": {"op": "<op id>", "reason": "empty expansion"}}` at `child: 0`. An expand with `on_empty: fail` is a marker source, which no model op may read.

`on_fail` (reduce only) sets what a group holding a failed record emits: `fail_group` (the default) emits the group's first marker, and `drop_failed` drops the failed rows before the call, so a group that loses every row emits `{"_fail": {"op": "<op id>", "reason": "empty group"}}`. A row's key is read before its other fields, so a marked row whose key reads joins its group; one whose key cannot be read forms its own group.

```yaml
ops:
  - id: "SplitKeywords"
    op: LambdaOp
    inputs: [Keywords]
    mode: expand
    on_empty: fail
    fn_name: split
    code: |
      def split(row):
          return [k for k in row[0].split(",") if k.strip()]
  - id: "TopThree"
    op: LambdaOp
    inputs: [Scores]
    mode: reduce
    by: patient_id
    on_fail: drop_failed
    fn_name: top_three
    code: |
      def top_three(rows):
          return sorted(rows, key=lambda r: -r[0]["score"])[:3]
```

#### Keyed seed

A `row` or `expand` `LambdaOp` takes an optional keyed `seed`, shaped and checked like a chat op's `rowwise_seed` key form. The seed reaches the function as a second argument, so it can draw from `random.Random(seed)`; a function without `seed` keeps its one-argument signature, and a seeded one must accept exactly two. The key's node must be listed in `inputs:`. `list`, `aligned` and `reduce` modes take no `seed`.

```yaml
ops:
  - id: "Sample"
    op: LambdaOp
    inputs: [Cases]
    fn_name: sample
    code: |
      def sample(row, seed):
          rng = random.Random(seed)
          return rng.choice(["positive", "negative"])
    seed: {key: {node: Cases, path: items.output.case_id}}
```

#### Where the function runs

- **Standalone `python` task**: every Lambda except the inlined case below, and always a Lambda with an `image` or `gpu`. The code runs in its own container, without network access, as an unprivileged user, on a CPU worker that advertises the `python` task type (a Docker-backed worker on a FlowMesh version that supports it), or, when the op sets `gpu`, on a GPU worker that advertises it. While every such worker is busy the batch waits; with none in the cluster the schedule fails before dispatch. The code may import any module the image provides (the default image has the standard library, numpy and pandas); relative imports are refused at submit. A raised exception fails the job naming the step, for example `Output task <id> failed (node <op>): python task failed: ValueError: ...`.
- **Inlined into an LLM task**: a row-mode Lambda that an LLM op reads as message content runs as a template step of that LLM's FlowMesh task, on the worker that runs it. This is not an isolation boundary. The worker's namespace has no `import`, so the code imports nothing; Lumilake wraps each row so it arrives decoded, and submit refuses code that binds a wrapper name, annotates a top-level assignment or uses a `Message`/`ops` type. If a non-LLM node also reads the Lambda, it also runs as a standalone task, so the function runs once per reader.
- **API-mode `LLMChatOp`**: cannot read a `LambdaOp`, since the request body carries no function step; the build fails naming the step. Make the Lambda a workflow output, or use a non-API chat.

The Lumilake server never runs submitted code: it only parses the source, and a Lambda whose inputs are all literal is not folded into a constant. A function passed as a Python callable through the SDK (`LambdaOp(fn=...)`) is the caller's own code and may run in process; its source is serialized with `dill.source.getsource`, with captured closure values written in.

A standalone task gets the shared Lambda runtime and every literal above 1 KiB as content-addressed blobs mounted as files, fetched through `LUMID_DATA_WORKER_URL` (or `LUMID_DATA_URL`); smaller literals stay inline.

#### Namespace

The function sees the builtins `int`, `float`, `str`, `bool`, `list`, `dict`, `tuple`, `set`, `len`, `sum`, `max`, `min`, `abs`, `round`, `sorted`, `reversed`, `enumerate`, `zip`, `map`, `filter`, `any`, `all`, `range` and `isinstance`, plus `json`, `re`, `math`, `np` and `pd` (the last two when the image provides them). A standalone task also has the common exception classes (`Exception`, `ArithmeticError`, `AssertionError`, `AttributeError`, `IndexError`, `KeyError`, `LookupError`, `NotImplementedError`, `OverflowError`, `RuntimeError`, `StopIteration`, `TypeError`, `ValueError`, `ZeroDivisionError`); inlined code has none. Submit refuses a name the code uses without binding, importing or defining it when the namespace lacks it, naming the name. The namespace decides which names resolve; the container is the security boundary.

#### Limits

These fields apply to the standalone task only:

| Field | Default | Bounds | Meaning |
|---|---|---|---|
| `timeout_s` | 600 | (0, 600]; (0, 3600] with `image` | Task timeout in seconds. |
| `memory_mb` | worker default | [128, 8192] | Task memory limit. |
| `image` | worker default | non-empty string | Container image; a Lambda with an image always runs as a standalone task. |
| `gpu` | none (CPU) | `count` in [1, 8]; optional `type`, `memory` | GPUs the task needs, sent as `resources.hardware.gpu`. The task is placed on a GPU worker that advertises the `python` task type. |
| `expected_runtime_s` | none | (0, effective `timeout_s`] | Planning hint for HALO: seconds per input row, never enforced. |
| `cache` | none | bool | Opt-in FlowMesh task caching, sent only when set. |
| `sample_value` | none | any non-null JSON | Value that data-profile preflight uses for a placeholder reading this Lambda; never executed. |

A `LambdaOp` with `gpu` is placed on a GPU worker that advertises the `python` task type by HALO and by `topological-sort`; every other `python` step runs on a CPU worker. The job waits while every GPU worker that can carry the step is busy and fails when the cluster has none; when the claimed GPU workers cannot carry it, HALO and `topological-sort` fail the job naming the node, and a remote optimizer refuses such a graph before submitting it. The step's own `gpu` request reaches FlowMesh unchanged and filters the workers it may land on, so `count`, `type` and `memory` choose the device; the job-level `hardware.gpu` hint does not override it. `memory` is a per-device floor in `Ki`, `Mi`, `Gi` or `Ti` units, so `24Gi` accepts a 24 GiB card or larger; a bare `24G` or `24GB` is rejected at submit. `type` is a case-insensitive substring of the device name, and `any`, `auto` or `*` means no type constraint. Use an `image` that provides the GPU libraries the code imports, for example `gpu: {count: 1, memory: 24Gi}` with a CUDA image.

When a job's input slices are coalesced into one workflow, the task runs their rows in sequence, so Lumilake coalesces only as many slices as keep `expected_runtime_s` times the rows within `timeout_s`; the rest run as separate copies of the workflow in the same batch. A slice whose row count is unknown before it runs is never coalesced.

### RouterOp

`RouterOp` partitions one source op's rows into named branches, one ordinary node per branch named `<id>.<branch>` that a consumer lists in `inputs:`. The first rule that matches wins; `match` is `contains`, `equals` or `empty`, optional `path` selects the field tested, and `default_branch` keeps unmatched rows. `error_branch: true` adds an `error` branch that receives the [failed records](#failed-records); every other branch skips them. Each branch is a list-mode `LambdaOp`, so a `path` walks its decoded row.

`match: empty` takes no `value` and matches `null`, `""`, `[]` or `{}`.

```yaml
ops:
  - {id: route, op: RouterOp, inputs: [norm], default_branch: patient,
     error_branch: true,
     rules: [{branch: diagnose, match: contains, value: "DIAGNOSIS READY"},
             {branch: test, match: contains, value: "REQUEST TEST"}]}
  - {id: split, op: RouterOp, inputs: [feedback], path: feedback_question,
     default_branch: ret, rules: [{branch: no_ret, match: empty}]}
```

### JoinOp

`JoinOp` merges row columns back by a key. A row column is any op with a runtime node, or a workflow input or `DataOp`, whose literal values are the column's rows. `how: left` (the default) keeps the first input's rows and their order, `how: inner` keeps only the keys every input carries, and `how: full` also appends a key only a later input carries, in first-seen order, with the fields it lacks left out. `require_match: true` fails the job unless every input carries the same keys. Each `attach` field joins one node's per-row output to the row it was computed from, under any `how`, and the node is resolved at submit, so no user code positions values. An attach target must be 1:1 over rows: a rowwise `LLMChatOp` (including one with a `rowwise_messages` column), a `LLMCompletionOp` or `EmbeddingOp` whose columns read one node, a row-mode or aligned `LambdaOp` over one non-broadcast input, an `HttpFetchOp`, a `ClassificationOp`, another `JoinOp`, or a chain of these that reaches one. A `FormatOp` counts only as a hop on such a chain: it has no runtime node of its own, so a format op named directly as an attach target is refused at submit, naming the op, the attach field and that reason. List every attach target in `inputs:` too, so its producer is built before the join. Router branches are disjoint, so a case lands in one branch and the other branch's fields are left out of the output row.

`by` is one dotted field path, or a list of them for a composite key; a live row whose key path is missing or null fails the job.

A first-input row carrying a failed record makes its output row that marker, and a marked further or attach row turns its key's output row into its marker; a marked further-input row whose key cannot be read fails the job.

`on_missing` governs an attach source only and defaults to `null_fill`: `null_fill` leaves the field out when the source lacks the row's key, and `fail_row` marks that row as failed instead. A duplicate key, an unpaired attach column and a field two inputs disagree on fail the job under either policy.

```yaml
ops:
  - {id: Merge, op: JoinOp, inputs: [Cases, Reply], by: case_id, how: inner,
     require_match: true, attach: {reply: Reply}, on_missing: null_fill}
```

### ParseOp

`ParseOp` parses one input column, one record per row, read at `value.items.output`.

```yaml
- {id: Verdict, op: ParseOp, inputs: [Answer], format: json, require: [eligibility], on_fail: marker}
```

`format: json` strips one fenced block and any prose before the first `{` or `[`, then decodes the JSON, which must end the reply; `require` names top-level keys it must carry. `format: regex` returns the named groups of `pattern`'s first match; `flags` is any of `i`, `s`, `m`. Invalid fields are refused at submit. A row the parse cannot handle becomes a failed record, or keeps its text under `on_fail: raw`, so a downstream op must accept both shapes.

### Failed records

A record that fails becomes a marker instead of an error, so one bad row costs neither its siblings nor the job. A failed record's value is the object `{"_fail": {"op": "<op id>", "reason": "<text>"}}` and nothing else; an ok record has no `_fail` key. A failed record keeps its position, so parallel columns stay aligned.

- `ParseOp` emits the marker for a row it cannot parse; the reason names what went wrong, such as `no JSON object`, `missing key 'eligibility'` or `pattern did not match`.
- A row `LambdaOp` does not call the function for a row with a marked cell; it emits the first such marker for that row. An aligned `LambdaOp` calls the function on the live columns and merges the marked rows back by position. A list `LambdaOp` receives markers as ordinary values, so a whole-list function sees every row and passes them on. A seeded row or expand `LambdaOp` draws no seed for a marked row, so a seed step that passes the marker on in that row's seed cell leaves the row marked instead of failing the task.
- An expand `LambdaOp` emits one tombstone item carrying the marker for a marked parent, and a parent that expands to no children emits its own marker under `on_empty: fail`.
- A reduce `LambdaOp` reads each row's key first: a marked row whose key reads joins its group, and `on_fail` decides that group. A row whose key cannot be read is its own group and emits its marker. See [`LambdaOp`](#lambdaop).
- A `JoinOp` keeps one output row per first-input row, in order: a marked first-input row is that row's marker, and a marked further input or attach row turns its key's output into the marker. No other row is affected. Its `on_missing: fail_row` marks the rows a further input lacks, and an attach length mismatch fails the job under either policy. See [`JoinOp`](#joinop).
- Downstream reads of an expand or join output see the marker under each item's `output`, so a row Lambda or a parse step treats such an item as a marked row.
- A user function that raises fails the task; only a parse or a pass-through produces a marker.
- `RouterOp` sends failed rows to an `error` branch when the op declares `error_branch: true`; otherwise a failed row goes to no branch. See [`RouterOp`](#routerop).
- Outputs archive the marker, so a job result shows which records failed and why.
- A source carries markers when a `ParseOp` or an input whose value is a marker reaches it through any op except a router's rule or default branch, which drop them, or when an expand `LambdaOp` declares `on_empty: fail`. A model op (`LLMChatOp` with `rowwise_template`, `LLMVisionOp`, `LLMCompletionOp`, `EmbeddingOp`, or a rowwise seed or aggregate table) may not read such a source: the job is refused at submit when a rowwise column, aggregate table, rowwise seed or embedding content reads one. Read a router's rule or default branch instead, so failed records never reach the model. An earlier round's output is an input like any other, so a reference that binds a marker follows the same rule.

`_fail` is reserved: a user function that returns an object carrying it fails the task, so a user value is never mistaken for a marker.

For workflow-format details, see `docs/WORKFLOWS.md`. For runnable examples, see `examples/templates/`.
