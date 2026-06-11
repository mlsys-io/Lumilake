# Ops Reference

Lumilake workflows are DAGs of operation classes registered under `lumilake_server.ops`. YAML `op:` values and native graph `_op` values use these class names. The op registry lives inside the Lumilake server image; YAML and native workflow submitters reach it through the HTTP / SDK surface, not by importing the module directly.

| Op | Purpose |
|----|---------|
| `InputOp` | Generated from workflow-level inputs. |
| `OutputOp` | Generated from workflow-level outputs. |
| `DataOp` | Inline static data. |
| `DataRetrievalOp` | Retrieve data via lumid-data-app (`type: lumid`, `mode: sql\|s3\|agent`). All modes route through the lumid connector; `LUMID_DATA_URL` is required, plus an effective lumid-data bearer token (`LUMID_DATA_TOKEN` overrides the fallback to `LUMILAKE_RUNTIME_TOKEN`). Optional `data_spec.sample_value` short-circuits data-profile preflight when this op is used as a placeholder source for a downstream `DataRetrievalOp`. |
| `MessageOp` | Build role/content message lists for language model calls. |
| `LLMChatOp` | Run text chat generation, including aggregate and row-wise table prompts. |
| `LLMVisionOp` | Run vision-language generation over image inputs. |
| `ImageGenerationOp` | Generate images from text prompts. |
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
