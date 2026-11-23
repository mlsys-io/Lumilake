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
``FormatOp``, ``LambdaOp``, ``RouterOp``, ``JoinOp``, ``RepeatOp``, ``OutputOp``
(auto-generated from ``outputs:`` block). ``RouterOp`` and ``RepeatOp`` are expanded
before any graph is built.

Users with an existing n8n workflow should submit it as
``Workflow-Format: n8n`` rather than transliterating it into YAML — no
YAML equivalent of the n8n wire format exists in this parser.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from ..ops.router_ops import expand_routers
from ..ops.seed_spec import Pairing, SeedKey
from ..ops.spec_validation import validate_fields
from ..ops.util_ops import CLEARS_KEY
from . import keyed_seed
from .common import make_id as _make_id
from .common import read_workflow_seed
from .repeat import unroll_repeats
from .yaml_functions import WorkflowFunctions, parse_functions
from .yaml_ops.common import EmitContext, OpEntry
from .yaml_ops.emitters import EMITTERS, op_id_prefix

SUPPORTED_OPS: frozenset[str] = frozenset(EMITTERS)


class _OpEntrySpec(BaseModel):
    """One YAML op entry: its id, op, inputs, iteration cap and used functions."""

    model_config = ConfigDict(extra="allow")

    id: StrictStr
    op: StrictStr
    inputs: list[StrictStr] = Field(default_factory=list)
    max_iter: StrictInt | None = None
    uses: list[StrictStr] = Field(default_factory=list)
    code: str | None = None

    @field_validator("id", "op")
    @classmethod
    def _named(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()

    @field_validator("inputs", "uses", mode="before")
    @classmethod
    def _none_as_empty(cls, value: Any) -> Any:
        return [] if value is None else value

    @model_validator(mode="after")
    def _uses_on_a_lambda(self) -> Self:
        if not self.uses:
            return self
        if self.op != "LambdaOp":
            raise ValueError("declares 'uses', which only a LambdaOp accepts")
        if self.code is not None and self.code.strip().startswith("lambda"):
            raise ValueError(
                "cannot combine 'uses' with lambda-expression code; write the function "
                "as a 'def'"
            )
        for name in self.uses:
            if self.uses.count(name) > 1:
                raise ValueError(f"names the function {name!r} more than once")
        return self

    def op_fields(self) -> dict[str, Any]:
        """Every key the user wrote beyond the entry's own, ``code`` included."""
        fields = (self.model_extra or {}).copy()
        if self.code is not None:
            fields["code"] = self.code
        return fields


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
    cleared_ids: Collection[str] = (),
    round_number: int | None = None,
    template_ids: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Parse a Lumilake YAML workflow spec into graph_specs.

    Args:
        payload: Either the raw YAML text, or an already-parsed dict.
        cleared_ids: The ids ``expand_routers`` returned as cleared, so a round that
            re-parses its own expansion keeps the flag on those branches.
        round_number: The dynamic round the ops run in, when they run inside one; a
            keyed row seed derives its draw from it.
        template_ids: Rendered op id -> the id a template body wrote, so a keyed row
            seed keeps a draw across rounds.

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
        cleared_ids=cleared_ids,
        round_number=round_number,
        template_ids=template_ids,
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
    """The workflow's job seed."""
    return read_workflow_seed(spec, "workflow")


def check_keyed_seed_duplicates(
    spec: dict[str, Any], inputs: dict[str, list[str]]
) -> None:
    """Refuse a repeated key over the whole input set, before slicing."""
    ops_block = spec.get("ops")
    if not isinstance(ops_block, list):
        return
    literal: dict[str, list[Any]] = {}
    declared: list[tuple[str, str, SeedKey, Pairing]] = []
    for op in ops_block:
        if not isinstance(op, dict):
            continue
        op_kind = op.get("op")
        data = op.get("data")
        if op_kind == "DataOp" and isinstance(op.get("id"), str):
            if isinstance(data, list):
                literal[op["id"]] = data
        if op_kind not in ("LLMChatOp", "LambdaOp"):
            continue
        field = "seed" if op_kind == "LambdaOp" else "rowwise_seed"
        op_id = str(op.get("id"))
        entry = op.get(field)
        if not isinstance(entry, dict) or "key" not in entry:
            continue
        seed = keyed_seed.read_key_seed(op_id, op_kind, field, entry)
        declared.append((op_id, op_kind, seed.key, seed.pairing))
    for op_id, op_kind, key, pairing in declared:
        values = inputs.get(key.node, literal.get(key.node))
        if values is None:
            continue
        keyed_seed.check_unique(
            op_id,
            op_kind,
            keyed_seed.key_cells(values, keyed_seed.key_parts(key.path)),
            pairing,
        )


def _compile_workflow(
    spec: dict[str, Any],
    scope: str,
    job_seed: int,
    *,
    cleared_ids: Collection[str] = (),
    round_number: int | None = None,
    template_ids: dict[str, str] | None = None,
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
    functions = parse_functions(spec.get("functions"))
    unrolled = unroll_repeats(ops_block, set(inputs))
    expanded, cleared = expand_routers(unrolled.ops, unrolled.copied_routers)
    cleared = cleared | frozenset(cleared_ids)
    seed_templates = {**unrolled.template_ids, **(template_ids or {})}
    seed_rounds = unrolled.rounds
    op_entries = [
        _parse_op_entry(entry, idx, functions) for idx, entry in enumerate(expanded)
    ]
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
            refs = unrolled.bind(user_id_to_internal, f"op '{user_id}'", user_id)
            missing = [ref for ref in entry.inputs if ref not in refs]
            if missing:
                raise ValueError(f"op '{user_id}' references unknown id(s): {missing}")
            unresolved = [ref for ref in entry.inputs if refs.target(ref) in pending]
            if unresolved:
                continue
            op_dict = _build_op_dict(
                entry,
                refs,
                scope,
                graph_ops,
                job_seed,
                round_number=round_number,
                template_ids=seed_templates,
                rounds=seed_rounds,
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
        where = f"output '{out.name}'"
        refs = unrolled.bind(user_id_to_internal, where, where)
        if out.ref not in refs:
            raise ValueError(f"output '{out.name}' references unknown id '{out.ref}'")
        op_id = _make_id(scope, "output", out.name)
        if op_id in graph_ops:
            raise ValueError(f"duplicate output id derived for '{out.name}'")
        out_dict: dict[str, Any] = {
            "_id": op_id,
            "_op": "OutputOp",
            "_max_iter": None,
            "_inputs": [refs[out.ref]],
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


def _parse_op_entry(entry: Any, idx: int, functions: WorkflowFunctions) -> OpEntry:
    if not isinstance(entry, dict):
        raise ValueError(f"op entry at index {idx} must be a mapping")
    spec = validate_fields(_OpEntrySpec, entry, _entry_where(entry, idx))
    fields = spec.op_fields()
    if spec.uses:
        fields["code"] = functions.include(spec.uses, spec.id, spec.code)
    return OpEntry(
        id=spec.id,
        op=spec.op,
        inputs=spec.inputs,
        fields=fields,
        max_iter=spec.max_iter,
    )


def _entry_where(entry: dict[str, Any], idx: int) -> str:
    """Name an op entry by its id once it gives one, and by its index before."""
    user_id = entry.get("id")
    if isinstance(user_id, str) and user_id.strip():
        return f"op '{user_id.strip()}'"
    return f"op entry at index {idx}"


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
    user_id_to_internal: Mapping[str, str],
    scope: str,
    graph_ops: dict[str, dict[str, Any]],
    job_seed: int,
    *,
    round_number: int | None = None,
    template_ids: dict[str, str] | None = None,
    rounds: dict[str, int] | None = None,
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
        rounds=rounds or {},
    )
    EMITTERS[entry.op].emit(op_dict, ctx)
    return op_dict
