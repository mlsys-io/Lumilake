"""Parse a Lumilake-native YAML workflow spec into ``graph_specs``.

The YAML format is a declarative alternative to the Python DSL. Each
list entry under ``ops:`` maps directly to a registered Lumilake op
(``DataOp``, ``LLMChatOp``, ``FormatOp``, ...). The resulting dict has
the same shape :func:`lumilake_server.parser.n8n.parse_n8n_payload`
produces, which :meth:`lumilake_server.graphs.Graph.from_json` then consumes.

```yaml
name: example_workflow            # optional top-level name
inputs:                           # workflow-level inputs -> InputOps
  query: ["NVDA", "TSLA"]         #   name -> list[str]
ops:
  - id: retrieve                  # unique id per op
    op: DataRetrievalOp           # registered Op class name
    inputs: [query]               # references to other op ids (or input names)
    data_spec:                    # op-specific fields are passed through
      type: lumid
      mode: s3
      template: docs/{entity}.json
      params:
        - label: entity
          node: query             # `node:` names an input/op id by user-facing id
          path: data
  - id: summarize
    op: LLMChatOp
    inputs: [retrieve]
    messages:
      - role: system
        content: "Summarize the documents the user provides."
      - role: user
        content: retrieve         # bare user-id -> implicit FormatOp wraps
                                  # the referenced op's output as this message
    config:
      model: meta-llama/Llama-3.1-8B-Instruct
      temperature: 0.2
outputs:
  - name: summary                 # output name (becomes InputOp("name"))
    ref: summarize                # id of the op whose value to expose
```

## Referencing upstream ops inside a message

Inline `messages:` content is either:

1. **Literal text** — passed through verbatim, no placeholder substitution.
2. **A bare user-facing id** (as in the example above: `content: retrieve`).
   The parser recognises that the string resolves to another op in the
   workflow and synthesizes an implicit `FormatOp` that wires that op's
   output as the message body.

For **templated composition** that interpolates multiple upstream outputs
(`"Summarize: {docs}"`, etc.), declare an explicit `FormatOp` and
reference it from the message:

```yaml
ops:
  - id: prompt
    op: FormatOp
    inputs: [retrieve]
    template: "Summarize these documents:\n{docs}"
    format_kwargs:
      docs: retrieve              # key -> user-facing id of the upstream op
  - id: summarize
    op: LLMChatOp
    inputs: [prompt]
    messages:
      - role: user
        content: prompt           # references the FormatOp above
    config: { model: meta-llama/Llama-3.1-8B-Instruct }
```

`LLMChatOp` has no `{placeholder}` substitution of its own — all
templating is done by upstream `FormatOp`s.

Supported op types: ``InputOp`` (auto-generated from ``inputs:`` block),
``DataOp``, ``DataRetrievalOp``, ``MessageOp``, ``LLMChatOp``,
``LLMCompletionOp``, ``LLMVisionOp``, ``ImageGenerationOp``, ``EmbeddingOp``,
``FormatOp``, ``LambdaOp``, ``RouterOp``, ``JoinOp``, ``OutputOp``
(auto-generated from ``outputs:`` block).

Users with an existing n8n workflow should submit it as
``Workflow-Format: n8n`` rather than transliterating it into YAML — no
YAML equivalent of the n8n wire format exists in this parser.
"""

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import yaml

from ..ops.router_ops import expand_routers
from ..ops.util_ops import CLEARS_KEY
from .common import make_id as _make_id
from .yaml_ops.common import EmitContext, OpEntry
from .yaml_ops.emitters import EMITTERS, op_id_prefix

SUPPORTED_OPS: frozenset[str] = frozenset(EMITTERS)


@dataclass
class _YamlGraphSpec:
    graph: dict[str, dict[str, Any]]
    inputs: dict[str, list[str]]


@dataclass
class _OutputEntry:
    name: str
    ref: str
    path: str | None = None


def parse_yaml_payload(
    payload: dict | str,
    *,
    round_number: int | None = None,
    template_ids: dict[str, str] | None = None,
    cleared_ids: Collection[str] = (),
) -> dict[str, dict[str, Any]]:
    """Parse a Lumilake YAML workflow spec into graph_specs.

    Args:
        payload: Either the raw YAML text, or an already-parsed dict.
        round_number: The dynamic round the ops run in, when they run inside
            one; a keyed row seed derives its draw from it.
        template_ids: Rendered op id -> the id a template body wrote, so a
            keyed row seed keeps a draw across rounds.
        cleared_ids: The ids ``expand_routers`` returned as cleared, so a round
            that re-parses its own expansion keeps the flag on those branches.

    Returns:
        ``{graph_name: {"graph": ops_dict, "inputs": inputs_dict,
        "seed": job_seed}}``, matching the shape produced by
        ``parse_n8n_payload``.

    Raises:
        ValueError: on any schema / reference / duplicate id problem.
    """
    spec_dict = _coerce_to_dict(payload)
    graph_name = _get_graph_name(spec_dict)

    if "ops" not in spec_dict:
        raise ValueError("YAML workflow must declare an 'ops:' list at the top level")

    job_seed = _get_job_seed(spec_dict)
    compiled_ops = _compile_workflow(
        spec_dict,
        graph_name,
        job_seed,
        round_number=round_number,
        template_ids=template_ids,
        cleared_ids=cleared_ids,
    )
    return {
        graph_name: {
            "graph": compiled_ops.graph,
            "inputs": compiled_ops.inputs,
            "seed": job_seed,
        }
    }


class YamlParseError(ValueError):
    """``ValueError`` subclass that preserves PyYAML's source location.

    ``MarkedYAMLError`` (PyYAML's error superclass for parse-time failures)
    carries a ``problem_mark`` with one-based ``line`` / ``column`` indices
    pointing at the offending token. We expose both as structured fields so
    callers (HTTP routes, the CLI) can surface them in error responses
    without re-parsing the YAML error message.
    """

    def __init__(self, message: str, *, line: int | None, column: int | None) -> None:
        super().__init__(message)
        self.line = line
        self.column = column


def _coerce_to_dict(payload: dict | str) -> dict[str, Any]:
    if isinstance(payload, str):
        try:
            loaded = yaml.safe_load(payload)
        except yaml.MarkedYAMLError as exc:
            mark = exc.problem_mark
            if mark is not None:
                line = mark.line + 1
                column = mark.column + 1
                message = f"Invalid YAML at line {line}, column {column}: {exc}"
                raise YamlParseError(message, line=line, column=column) from exc
            raise YamlParseError(
                f"Invalid YAML: {exc}", line=None, column=None
            ) from exc
        except yaml.YAMLError as exc:
            raise YamlParseError(
                f"Invalid YAML: {exc}", line=None, column=None
            ) from exc
        if not isinstance(loaded, dict):
            raise ValueError("YAML workflow must be a mapping at the top level")
        return loaded
    if isinstance(payload, dict):
        return payload
    raise ValueError("YAML workflow payload must be a string or dict")


def _get_graph_name(spec: dict[str, Any]) -> str:
    name = spec.get("name")
    if name is None:
        return "yaml_workflow"
    if not isinstance(name, str) or not name.strip():
        raise ValueError("workflow 'name' must be a non-empty string")
    return name.strip()


def _get_job_seed(spec: dict[str, Any]) -> int:
    """The workflow's job seed, which every keyed row seed derives from."""
    value = spec.get("seed", 0)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"workflow 'seed' must be an integer, got {value!r}")
    return value


def _compile_workflow(
    spec: dict[str, Any],
    scope: str,
    job_seed: int,
    *,
    round_number: int | None = None,
    template_ids: dict[str, str] | None = None,
    cleared_ids: Collection[str] = (),
) -> _YamlGraphSpec:
    inputs_block = spec.get("inputs", {}) or {}
    ops_block = spec.get("ops", []) or []
    outputs_block = spec.get("outputs", []) or []

    if not isinstance(inputs_block, dict):
        raise ValueError("'inputs' must be a mapping of name -> list[str]")
    if not isinstance(ops_block, list):
        raise ValueError("'ops' must be a list")
    if not isinstance(outputs_block, list):
        raise ValueError("'outputs' must be a list")

    inputs = _validate_inputs(inputs_block)
    expanded, cleared = expand_routers(ops_block)
    cleared = cleared | frozenset(cleared_ids)
    op_entries = [_parse_op_entry(entry, idx) for idx, entry in enumerate(expanded)]
    output_entries = [
        _parse_output_entry(entry, idx) for idx, entry in enumerate(outputs_block)
    ]

    # Build the registry of user-facing ids -> internal op ids.
    user_id_to_internal: dict[str, str] = {}
    graph_ops: dict[str, dict[str, Any]] = {}

    # 1. InputOps from the workflow-level inputs block.
    for input_name in inputs:
        if input_name in user_id_to_internal:
            raise ValueError(f"duplicate id '{input_name}' (input collides with op)")
        op_id = _make_id(scope, "input", input_name)
        graph_ops[op_id] = {
            "_id": op_id,
            "_op": "InputOp",
            "_max_iter": None,
            "_inputs": [],
            "name": input_name,
        }
        user_id_to_internal[input_name] = op_id

    # 2. Workflow ops. Register ids first so forward references in inputs
    #    can be resolved regardless of order.
    for entry in op_entries:
        if entry.id in user_id_to_internal:
            raise ValueError(f"duplicate id '{entry.id}'")
        if entry.op not in SUPPORTED_OPS:
            raise ValueError(
                f"unsupported op type '{entry.op}' for id '{entry.id}'. "
                f"Supported: {sorted(SUPPORTED_OPS)}"
            )
        user_id_to_internal[entry.id] = _make_id(
            scope, op_id_prefix(entry.op), entry.id
        )

    # Topological order: emit ops whose dependencies are already resolved.
    pending = {entry.id: entry for entry in op_entries}
    while pending:
        progress = False
        for user_id, entry in list(pending.items()):
            missing = [ref for ref in entry.inputs if ref not in user_id_to_internal]
            if missing:
                raise ValueError(f"op '{user_id}' references unknown id(s): {missing}")
            unresolved = [ref for ref in entry.inputs if ref in pending]
            if unresolved:
                continue
            op_dict = _build_op_dict(
                entry,
                user_id_to_internal,
                scope,
                graph_ops,
                job_seed,
                round_number=round_number,
                template_ids=template_ids,
            )
            if user_id in cleared:
                op_dict[CLEARS_KEY] = True
            for implicit in op_dict.pop("__implicit_ops__", []):
                graph_ops[implicit["_id"]] = implicit
            graph_ops[op_dict["_id"]] = op_dict
            pending.pop(user_id)
            progress = True
        if not progress:
            cycle = sorted(pending)
            raise ValueError(f"cycle detected in YAML workflow ops: {cycle}")

    # 3. OutputOps.
    for out in output_entries:
        if out.ref not in user_id_to_internal:
            raise ValueError(f"output '{out.name}' references unknown id '{out.ref}'")
        op_id = _make_id(scope, "output", out.name)
        if op_id in graph_ops:
            raise ValueError(f"duplicate output id derived for '{out.name}'")
        out_dict: dict[str, Any] = {
            "_id": op_id,
            "_op": "OutputOp",
            "_max_iter": None,
            "_inputs": [user_id_to_internal[out.ref]],
            "name": out.name,
        }
        if out.path is not None:
            out_dict["path"] = out.path
        graph_ops[op_id] = out_dict

    return _YamlGraphSpec(graph=graph_ops, inputs=inputs)


def _validate_inputs(inputs_block: Any) -> dict[str, list[str]]:
    if not isinstance(inputs_block, dict):
        raise ValueError("'inputs' must be a mapping of name -> list[str]")
    inputs: dict[str, list[str]] = {}
    for name, value in inputs_block.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("input names must be non-empty strings")
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(
                f"input '{name}' must be a list of strings, got {type(value).__name__}"
            )
        inputs[name] = list(value)
    return inputs


def _parse_op_entry(entry: Any, idx: int) -> OpEntry:
    if not isinstance(entry, dict):
        raise ValueError(f"op entry at index {idx} must be a mapping")
    reserved = {"id", "op", "inputs", "max_iter"}
    user_id = entry.get("id")
    op_type = entry.get("op")
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError(f"op entry at index {idx} missing 'id'")
    if not isinstance(op_type, str) or not op_type.strip():
        raise ValueError(f"op entry '{user_id}' missing 'op' type")

    raw_inputs = entry.get("inputs", []) or []
    if not isinstance(raw_inputs, list) or not all(
        isinstance(r, str) for r in raw_inputs
    ):
        raise ValueError(f"op '{user_id}' inputs must be a list of ids")

    fields_dict = {k: v for k, v in entry.items() if k not in reserved}
    max_iter = entry.get("max_iter")
    if max_iter is not None and not isinstance(max_iter, int):
        raise ValueError(f"op '{user_id}' max_iter must be int or null")

    return OpEntry(
        id=user_id.strip(),
        op=op_type.strip(),
        inputs=list(raw_inputs),
        fields=fields_dict,
        max_iter=max_iter,
    )


def _parse_output_entry(entry: Any, idx: int) -> _OutputEntry:
    if not isinstance(entry, dict):
        raise ValueError(f"output entry at index {idx} must be a mapping")
    name = entry.get("name")
    ref = entry.get("ref")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"output entry at index {idx} missing 'name'")
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError(f"output entry '{name}' missing 'ref'")
    path = entry.get("path")
    if path is not None and (not isinstance(path, str) or not path.strip()):
        raise ValueError(f"output entry '{name}' path must be a non-empty string")
    return _OutputEntry(
        name=name.strip(),
        ref=ref.strip(),
        path=path.strip() if isinstance(path, str) else None,
    )


def _build_op_dict(
    entry: OpEntry,
    user_id_to_internal: dict[str, str],
    scope: str,
    graph_ops: dict[str, dict[str, Any]],
    job_seed: int,
    *,
    round_number: int | None = None,
    template_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    internal_id = user_id_to_internal[entry.id]
    input_ids = [user_id_to_internal[ref] for ref in entry.inputs]

    op_dict: dict[str, Any] = {
        "_id": internal_id,
        "_op": entry.op,
        "_max_iter": entry.max_iter,
        "_inputs": input_ids,
    }

    ctx = EmitContext(
        entry=entry,
        user_id_to_internal=user_id_to_internal,
        scope=scope,
        graph_ops=graph_ops,
        job_seed=job_seed,
        round_number=round_number,
        template_ids=template_ids or {},
    )
    EMITTERS[entry.op].emit(op_dict, ctx)
    return op_dict
