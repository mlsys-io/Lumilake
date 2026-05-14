# Ops Reference

Lumilake workflows are DAGs of operation classes registered under `lumilake_server.ops`. YAML `op:` values and native graph `_op` values use these class names. The op registry lives inside the Lumilake server image; YAML and native workflow submitters reach it through the HTTP / SDK surface, not by importing the module directly.

| Op | Purpose |
|----|---------|
| `InputOp` | Generated from workflow-level inputs. |
| `OutputOp` | Generated from workflow-level outputs. |
| `DataOp` | Inline static data. |
| `DataRetrievalOp` | Retrieve data through SQL, S3-compatible storage, or the lumid.data agent mode. |
| `MessageOp` | Build role/content message lists for language model calls. |
| `LLMChatOp` | Run text chat generation, including aggregate and row-wise table prompts. |
| `LLMVisionOp` | Run vision-language generation over image inputs. |
| `ImageGenerationOp` | Generate images from text prompts. |
| `FormatOp` | Template upstream outputs into strings for downstream ops. |
| `LambdaOp` | Execute a serialized Python function over upstream values. |

## YAML Use

YAML workflows declare user-authored ops under `ops:`. `InputOp` and `OutputOp` are generated from the top-level `inputs:` and `outputs:` blocks; do not declare them manually in YAML.

```yaml
ops:
  - id: "Prompt"
    op: FormatOp
    inputs: [Stock]
    template: "Summarize {ref0}"
    format_kwargs:
      ref0: Stock

  - id: "Summary"
    op: LLMChatOp
    inputs: ["Prompt"]
    messages:
      - role: user
        content: "Prompt"
```

For workflow-format details, see `docs/WORKFLOWS.md`. For runnable examples, see `examples/templates/`.
