# Ops Reference

Lumilake workflows are DAGs of operation classes registered under `lumilake_server.ops`. YAML `op:` values and native graph `_op` values use these class names. The op registry lives inside the Lumilake server image; YAML and native workflow submitters reach it through the HTTP / SDK surface, not by importing the module directly.

| Op | Purpose |
|----|---------|
| `InputOp` | Generated from workflow-level inputs. |
| `OutputOp` | Generated from workflow-level outputs. |
| `DataOp` | Inline static data. |
| `DataRetrievalOp` | Retrieve data via lumid-data-app (`type: lumid`, `mode: sql\|s3\|agent`). All modes route through the lumid connector; `LUMID_DATA_URL` is required, plus an effective lumid-data bearer token (`LUMID_DATA_TOKEN` overrides the fallback to `LUMILAKE_RUNTIME_TOKEN`). Optional `data_spec.sample_value` short-circuits data-profile preflight when this op is used as a placeholder source for a downstream `DataRetrievalOp`. |
| `MessageOp` | Build role/content message lists for language model calls. |
| `LLMChatOp` | Run text chat generation, including aggregate and row-wise table prompts. An optional `config.api` block routes the op to an external LLM API endpoint instead of a locally-loaded model (see below). |
| `LLMVisionOp` | Run vision-language generation over image inputs. |
| `ImageGenerationOp` | Generate images from text prompts. |
| `EmbeddingOp` | Embed text with a FlowMesh-served embedding model and return one vector per input text. |
| `FormatOp` | Template upstream outputs into strings for downstream ops. |
| `LambdaOp` | Execute a serialized Python function over upstream values. |

## YAML Use

YAML workflows declare user-authored ops under `ops:`. `InputOp` and `OutputOp` are generated from the top-level `inputs:` and `outputs:` blocks; do not declare them manually in YAML.

### FormatOp

`FormatOp` interpolates upstream op outputs into a Python `str.format`
template. `format_kwargs` maps each `{name}` placeholder in `template`
to the user-facing id of an upstream op (or workflow input). Listing the
referenced ids in `inputs:` keeps the DAG wiring explicit.

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

`format_args` is the positional variant: each id resolves to a `{0}`,
`{1}`, ... placeholder. Use `format_kwargs` for named placeholders and
`format_args` for positional ones; the two may be combined.

### EmbeddingOp

`EmbeddingOp` embeds text with a FlowMesh-served embedding model and
returns one vector per input text. `config.model` is the embedding model
id; the optional `gpu_memory_utilization` / `tensor_parallel_size` config
fields are passed through to the vLLM engine. `input` references the
upstream op (or workflow input) whose text is embedded — declare a
`DataOp` for literal text, or point at an `InputOp` / upstream op output.
Wire the reference into `inputs:` to keep the DAG explicit.

The op dispatches a FlowMesh `embedding` task whose `model.vllm.convert`
is `embed`; that routes the task to the vLLM embedding executor. Vectors
are returned as an artifact, not inline: the response carries `model`,
`embedding_file` (`embeddings.safetensors`, tensor key `embeddings`,
shape `[count, dim]` float32, row-aligned to the input texts), and a
`usage` block whose `num_requests` / `embedding_dim` give the row count
and vector dimension. The op surfaces the archived embedding artifact ref
the same way `ImageGenerationOp` surfaces produced images; a consumer
that needs raw vectors loads `embeddings.safetensors`.

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

The `vectors` output is a per-row list — one entry per input text, not an
inline list of floats. Every entry carries its `row` index, the `model`,
and a reference to the shared `embeddings.safetensors` artifact; row `i`
is the embedding of input text `i`. A downstream op therefore receives
`slice_length` per-row vector inputs (one embedding per input doc), and a
consumer needing raw floats loads `embeddings.safetensors` and indexes by
`row`.

### LLMChatOp via external API

By default `LLMChatOp` runs against a locally-loaded model (vLLM /
transformers) on a FlowMesh worker. Setting `config.api` routes the op to
an external LLM API endpoint instead: the runtime builds a FlowMesh `api`
task whose body is an OpenAI-style `{model, messages, ...samplers}` chat
completion request, and the worker's `api_executor` performs the HTTP call.

```yaml
ops:
  - id: "Ask"
    op: LLMChatOp
    messages:
      - role: user
        content: "Summarize the latest news for {symbol}."
    config:
      api: {}
      max_tokens: 256
```

`config.api.url` is optional and defaults to the serving endpoint
`https://lum.id/llm/v1/chat/completions`. `config.model` is also optional
once `config.api` is set — the request model resolves from `config.api.model`,
then the top-level `config.model`, then the typed default `deepseek-v4-flash`
(`LLMVisionOp` always runs against a locally-loaded model and still requires
`config.model`). Sampler fields on `config` (e.g.
`max_tokens`, `temperature`) are merged into the request body.

`config.api.timeout_sec` sets the per-request timeout for the API call,
in seconds. When set, it is emitted into the FlowMesh `api` task spec and
overrides the executor's default; when unset, no timeout key is emitted and
the executor's own default applies. Use it for long-running extractions over
large documents, which can exceed the executor's default.

The credential is resolved server-side, not supplied by the caller. If
`config.api.url`'s origin is on the trusted-origin allowlist (the default
`https://lum.id`, extendable via `LUMILAKE_API_TRUSTED_ORIGINS`), the server
attaches `Authorization: Bearer <LUMILAKE_RUNTIME_TOKEN>` itself; for any
other origin the caller must set `config.api.authorization` explicitly, or
the request is rejected before dispatch. This header does become part of
the FlowMesh task spec that is actually submitted for execution — it is not
kept out of the spec — but it is redacted (replaced with `***REDACTED***`)
before the job is archived or an error body is logged.

A message may also reference an upstream node's runtime output — the same
way the local backend does, via `inputs:` plus that op's id in `messages:`
(directly for a non-`LLMChatOp` upstream such as `DataRetrievalOp`, or
through `FormatOp` when relaying another `LLMChatOp`'s output) — and API
mode is not restricted to literal, build-time content. Such a reference
renders as a FlowMesh `${node.path}` dispatch-time
placeholder instead of a literal value: FlowMesh's dispatcher resolves it
against the referenced node's real result immediately before the `api`
executor runs, and the referenced node is added to this op's FlowMesh
dependencies so dispatch waits for it. An upstream `LLMChatOp` (local or
API-backed) renders as its text output; an upstream `DataRetrievalOp`
renders per its mode (e.g. `sql` renders the retrieved table). A reference
is always single-valued: an upstream `LLMChatOp` that itself fanned out
into multiple row-aligned nodes (see below) cannot be referenced this way —
building the graph rejects it up front, since only this op's own message
columns can carry that per-row alignment. `return_history` also needs
per-item prompt metadata that only a locally-run `LLMChatOp` carries, so an
API-backed op with `return_history` enabled cannot be referenced by a
downstream op's message either.

When a message's literal content resolves to a multi-row input column
(e.g. `Stock: ["NVDA", "AAPL"]`), the op fans out into one FlowMesh `api`
task per row instead of one aggregate call: row 0 keeps the op's own id,
and row `i` (`i >= 1`) runs as `<op id>__row<i>`, each with that row's
value substituted into the message content and each mapped back to the
same declared output. Rows dispatch and complete independently, but
within a single FlowMesh workflow request the failure semantics are
all-or-nothing: if any row's task fails, that workflow request fails
and no partial per-row results are returned for it, even for rows that
already completed successfully.

This guarantee holds per FlowMesh workflow request, not per job. A
job's input rows can be split across multiple `input_batch_size`
slices, each dispatched as its own independent workflow request; one
slice's failure does not roll back another slice's already-merged
results. The job's final response mixes the failed slice's rows (empty
output plus an error entry) with the other slice's real per-row
values.

### LambdaOp

`LambdaOp` runs a serialized Python function against the listed
upstream values. YAML carries the function as source code (`code`) plus
a `fn_name`. The function must accept the input tuple in the same
order as `inputs:` and return a string.

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

For Python-side authoring, `lumilake_server.ops.LambdaOp(fn=...)`
serializes the function automatically via `dill.source.getsource` — see
`src/lumilake_server/ops/util_ops.py` for the closure-capture rules.

For workflow-format details, see `docs/WORKFLOWS.md`. For runnable examples, see `examples/templates/`.
