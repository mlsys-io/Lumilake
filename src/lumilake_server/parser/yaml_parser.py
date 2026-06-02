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
      type: s3
      connection_string: s3://bucket
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
``LLMVisionOp``, ``ImageGenerationOp``, ``FormatOp``, ``LambdaOp``,
``OutputOp`` (auto-generated from ``outputs:`` block).

Users with an existing n8n workflow should submit it as
``Workflow-Format: n8n`` rather than transliterating it into YAML — no
YAML equivalent of the n8n wire format exists in this parser.
"""

from dataclasses import dataclass, field
from typing import Any

import yaml
from lumilake import envs

from .common import make_id as _make_id

SUPPORTED_OPS: frozenset[str] = frozenset(
    {
        "DataOp",
        "DataRetrievalOp",
        "MessageOp",
        "LLMChatOp",
        "LLMVisionOp",
        "ImageGenerationOp",
        "FormatOp",
        "LambdaOp",
    }
)


@dataclass
class _YamlGraphSpec:
    graph: dict[str, dict[str, Any]]
    inputs: dict[str, list[str]]


@dataclass
class _OpEntry:
    id: str
    op: str
    inputs: list[str] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)
    max_iter: int | None = None


@dataclass
class _OutputEntry:
    name: str
    ref: str


def parse_yaml_payload(payload: dict | str) -> dict[str, dict[str, Any]]:
    """Parse a Lumilake YAML workflow spec into graph_specs.

    Args:
        payload: Either the raw YAML text, or an already-parsed dict.

    Returns:
        ``{graph_name: {"graph": ops_dict, "inputs": inputs_dict}}``,
        matching the shape produced by ``parse_n8n_payload``.

    Raises:
        ValueError: on any schema / reference / duplicate id problem.
    """
    spec_dict = _coerce_to_dict(payload)
    graph_name = _get_graph_name(spec_dict)

    if "ops" not in spec_dict:
        raise ValueError("YAML workflow must declare an 'ops:' list at the top level")

    compiled_ops = _compile_workflow(spec_dict, graph_name)
    return {
        graph_name: {
            "graph": compiled_ops.graph,
            "inputs": compiled_ops.inputs,
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


def _compile_workflow(spec: dict[str, Any], scope: str) -> _YamlGraphSpec:
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
    op_entries = [_parse_op_entry(entry, idx) for idx, entry in enumerate(ops_block)]
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
            scope, _op_id_prefix(entry.op), entry.id
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
            op_dict = _build_op_dict(entry, user_id_to_internal, scope, graph_ops)
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
        graph_ops[op_id] = {
            "_id": op_id,
            "_op": "OutputOp",
            "_max_iter": None,
            "_inputs": [user_id_to_internal[out.ref]],
            "name": out.name,
        }

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


def _parse_op_entry(entry: Any, idx: int) -> _OpEntry:
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

    return _OpEntry(
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
    return _OutputEntry(name=name.strip(), ref=ref.strip())


def _build_op_dict(
    entry: _OpEntry,
    user_id_to_internal: dict[str, str],
    scope: str,
    graph_ops: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    internal_id = user_id_to_internal[entry.id]
    input_ids = [user_id_to_internal[ref] for ref in entry.inputs]

    op_dict: dict[str, Any] = {
        "_id": internal_id,
        "_op": entry.op,
        "_max_iter": entry.max_iter,
        "_inputs": input_ids,
    }

    if entry.op == "DataOp":
        _emit_data_op(op_dict, entry)
    elif entry.op == "DataRetrievalOp":
        _emit_data_retrieval_op(op_dict, entry, user_id_to_internal)
    elif entry.op == "MessageOp":
        _emit_message_op(op_dict, entry, user_id_to_internal)
    elif entry.op == "LLMChatOp":
        _emit_llm_chat_op(op_dict, entry, user_id_to_internal, scope, graph_ops)
    elif entry.op == "LLMVisionOp":
        _emit_llm_vision_op(op_dict, entry, user_id_to_internal, scope, graph_ops)
    elif entry.op == "ImageGenerationOp":
        _emit_image_generation_op(op_dict, entry, user_id_to_internal, scope, graph_ops)
    elif entry.op == "FormatOp":
        _emit_format_op(op_dict, entry, user_id_to_internal)
    elif entry.op == "LambdaOp":
        _emit_lambda_op(op_dict, entry)
    else:  # pragma: no cover - guarded upstream
        raise ValueError(f"unsupported op type '{entry.op}'")

    return op_dict


def _emit_data_op(op_dict: dict[str, Any], entry: _OpEntry) -> None:
    data = entry.fields.get("data")
    if not isinstance(data, list) or not all(isinstance(v, str) for v in data):
        raise ValueError(f"DataOp '{entry.id}' requires 'data: list[str]'")
    op_dict["data"] = list(data)
    op_dict["_inputs"] = []  # DataOps have no inputs


def _emit_data_retrieval_op(
    op_dict: dict[str, Any], entry: _OpEntry, user_id_to_internal: dict[str, str]
) -> None:
    data_spec = entry.fields.get("data_spec")
    if not isinstance(data_spec, dict):
        raise ValueError(f"DataRetrievalOp '{entry.id}' requires 'data_spec'")
    op_dict["data_spec"] = _resolve_data_spec(data_spec, entry.id, user_id_to_internal)


def _resolve_data_spec(
    data_spec: dict[str, Any],
    entry_id: str,
    user_id_to_internal: dict[str, str],
    *,
    op_label: str = "DataRetrievalOp",
    field_label: str = "data_spec",
) -> dict[str, Any]:
    """Resolve ``${VAR}`` placeholders + ``params[*].node`` refs in a spec.

    ``op_label`` and ``field_label`` are threaded through every error
    message so the caller sees the op and field they actually authored.
    """
    resolved = dict(data_spec)
    conn = resolved.get("connection_string")
    if isinstance(conn, str):
        resolved["connection_string"] = _expand_env_placeholders(conn)
    params = resolved.get("params")
    if params is None:
        return resolved
    if not isinstance(params, list):
        raise ValueError(f"{op_label} '{entry_id}' {field_label}.params must be a list")
    new_params: list[dict[str, Any]] = []
    for i, param in enumerate(params):
        if not isinstance(param, dict):
            raise ValueError(
                f"{op_label} '{entry_id}' {field_label}.params[{i}] must be a mapping"
            )
        resolved_param = dict(param)
        if "node" in resolved_param:
            node_ref = resolved_param["node"]
            if not isinstance(node_ref, str):
                raise ValueError(
                    f"{op_label} '{entry_id}' {field_label}.params[{i}].node "
                    "must be a string"
                )
            if node_ref not in user_id_to_internal:
                raise ValueError(
                    f"{op_label} '{entry_id}' {field_label}.params[{i}].node "
                    f"references unknown input/op id '{node_ref}'"
                )
            resolved_param["node"] = user_id_to_internal[node_ref]
        new_params.append(resolved_param)
    resolved["params"] = new_params
    return resolved


def _emit_message_op(
    op_dict: dict[str, Any], entry: _OpEntry, user_id_to_internal: dict[str, str]
) -> None:
    messages = entry.fields.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"MessageOp '{entry.id}' requires non-empty 'messages'")
    serialized: list[dict[str, str]] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"MessageOp '{entry.id}' messages[{i}] must be a mapping")
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(
                f"MessageOp '{entry.id}' messages[{i}] requires string role + content"
            )
        # Allow referencing an upstream op by id in `content` — mirror n8n parser
        # behaviour, which stores the op id directly in the serialized message.
        resolved_content = user_id_to_internal.get(content, content)
        serialized.append({"role": role, "content": resolved_content})
    op_dict["messages"] = serialized


def _emit_llm_chat_op(
    op_dict: dict[str, Any],
    entry: _OpEntry,
    user_id_to_internal: dict[str, str],
    scope: str,
    graph_ops: dict[str, dict[str, Any]],
) -> None:
    _emit_llm_like_op(
        op_dict,
        entry,
        user_id_to_internal,
        scope,
        graph_ops,
        op_kind="LLMChatOp",
    )


def _emit_llm_vision_op(
    op_dict: dict[str, Any],
    entry: _OpEntry,
    user_id_to_internal: dict[str, str],
    scope: str,
    graph_ops: dict[str, dict[str, Any]],
) -> None:
    image_source = entry.fields.get("image_source")
    if not isinstance(image_source, str) or not image_source.strip():
        raise ValueError(
            f"LLMVisionOp '{entry.id}' requires 'image_source' referencing an op id"
        )
    if image_source not in user_id_to_internal:
        raise ValueError(
            f"LLMVisionOp '{entry.id}' image_source references unknown id "
            f"'{image_source}'"
        )
    image_path = entry.fields.get("image_path", "images")
    if not isinstance(image_path, str) or not image_path:
        raise ValueError(f"LLMVisionOp '{entry.id}' image_path must be a string")

    _emit_llm_like_op(
        op_dict,
        entry,
        user_id_to_internal,
        scope,
        graph_ops,
        op_kind="LLMVisionOp",
    )

    op_dict["image_source"] = user_id_to_internal[image_source]
    op_dict["image_path"] = image_path


def _emit_image_generation_op(
    op_dict: dict[str, Any],
    entry: _OpEntry,
    user_id_to_internal: dict[str, str],
    scope: str,
    graph_ops: dict[str, dict[str, Any]],
) -> None:
    config = entry.fields.get("config")
    if not isinstance(config, dict) or "model" not in config:
        raise ValueError(
            f"ImageGenerationOp '{entry.id}' requires 'config' with a 'model' field"
        )

    # Either reference an existing FormatOp/content op directly, or provide an
    # inline `content` string that is a bare user-facing id (mirroring how n8n
    # wraps `{{ $('Some Node') }}` prompts through an implicit FormatOp).
    content_ref = entry.fields.get("content")
    content_id: str | None = None
    if isinstance(content_ref, str) and content_ref.strip():
        target = content_ref.strip()
        if target not in user_id_to_internal:
            raise ValueError(
                f"ImageGenerationOp '{entry.id}' content references unknown id "
                f"'{target}'"
            )
        target_internal = user_id_to_internal[target]
        referenced = graph_ops.get(target_internal)
        if referenced and referenced.get("_op") == "FormatOp":
            content_id = target_internal
        else:
            fmt_id = _make_id(scope, "format", entry.id)
            fmt_op = {
                "_id": fmt_id,
                "_op": "FormatOp",
                "_max_iter": None,
                "_inputs": [target_internal],
                "template": "{ref0}",
                "format_args": [],
                "format_kwargs": {"ref0": target_internal},
            }
            op_dict.setdefault("__implicit_ops__", []).append(fmt_op)
            content_id = fmt_id
    else:
        raise ValueError(
            f"ImageGenerationOp '{entry.id}' requires a 'content' user-id reference"
        )

    op_dict["content"] = content_id
    op_dict["config"] = _build_generation_config(
        config, entry.id, op_kind="ImageGenerationOp"
    )
    op_dict["cacheable"] = bool(entry.fields.get("cacheable", False))
    op_dict["_inputs"] = [content_id]


def _emit_llm_like_op(
    op_dict: dict[str, Any],
    entry: _OpEntry,
    user_id_to_internal: dict[str, str],
    scope: str,
    graph_ops: dict[str, dict[str, Any]],
    *,
    op_kind: str,
) -> None:
    """Shared core for :class:`LLMChatOp` and :class:`LLMVisionOp`.

    Both op types consume a :class:`MessageOp` and carry a generation ``config``
    plus optional ``aggregate_table`` / ``rowwise_*`` / ``system_messages``
    fields. They differ in:

      * ``_inputs`` composition: ``LLMChatOp`` adds aggregate/rowwise node
        references to ``_inputs`` alongside the :class:`MessageOp`; the n8n
        compiler does *not* do this for :class:`LLMVisionOp`, so we mirror that
        asymmetry here.
      * Extra vision-only fields (``image_source``, ``image_path``) are set by
        the caller after this helper returns.
    """
    config = entry.fields.get("config")
    if not isinstance(config, dict) or "model" not in config:
        raise ValueError(
            f"{op_kind} '{entry.id}' requires 'config' with a 'model' field"
        )

    prompt_spec = entry.fields.get("prompt")
    messages_ref = entry.fields.get("messages_ref")
    inline_messages = entry.fields.get("messages")

    if messages_ref is not None:
        if not isinstance(messages_ref, str):
            raise ValueError(f"{op_kind} '{entry.id}' messages_ref must be a string id")
        if messages_ref not in user_id_to_internal:
            raise ValueError(
                f"{op_kind} '{entry.id}' messages_ref references unknown id "
                f"'{messages_ref}'"
            )
        msg_internal_id = user_id_to_internal[messages_ref]
        referenced_op = graph_ops.get(msg_internal_id)
        if referenced_op is None or referenced_op.get("_op") != "MessageOp":
            referenced_type = (
                referenced_op.get("_op") if referenced_op is not None else "<pending>"
            )
            raise ValueError(
                f"{op_kind} '{entry.id}' messages_ref '{messages_ref}' must "
                f"reference a MessageOp, got '{referenced_type}'"
            )
    elif inline_messages is not None:
        prepared_messages, extra_format_ops = _build_inline_messages(
            inline_messages,
            prompt_spec=prompt_spec,
            entry=entry,
            user_id_to_internal=user_id_to_internal,
            scope=scope,
            graph_ops=graph_ops,
        )
        for fmt_op in extra_format_ops:
            op_dict.setdefault("__implicit_ops__", []).append(fmt_op)

        # Create an implicit MessageOp that the LLM depends on. Derive its
        # internal id from the LLM op's user id — matching n8n's
        # `_make_id(scope, "message", node_name)`.
        msg_entry = _OpEntry(
            id=entry.id,
            op="MessageOp",
            inputs=[],
            fields={"messages": prepared_messages},
        )
        msg_internal_id = _make_id(scope, "message", entry.id)
        # `_inputs` on a MessageOp mirrors n8n: only ops whose ids appear in
        # message contents count.
        seen: list[str] = []
        for ref in entry.inputs:
            dep = user_id_to_internal[ref]
            if dep not in seen:
                seen.append(dep)
        for fmt_op in extra_format_ops:
            if fmt_op["_id"] not in seen:
                seen.append(fmt_op["_id"])
        resolved_contents: set[str] = set()
        for msg in prepared_messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            resolved_contents.add(user_id_to_internal.get(content, content))
        msg_inputs = [dep for dep in seen if dep in resolved_contents]
        msg_dict = {
            "_id": msg_internal_id,
            "_op": "MessageOp",
            "_max_iter": None,
            "_inputs": msg_inputs,
        }
        _emit_message_op(msg_dict, msg_entry, user_id_to_internal)
        op_dict.setdefault("__implicit_ops__", []).append(msg_dict)
    else:
        raise ValueError(
            f"{op_kind} '{entry.id}' requires 'messages' (inline) or 'messages_ref'"
        )

    op_dict["messages"] = msg_internal_id
    op_dict["config"] = _build_generation_config(config, entry.id, op_kind=op_kind)
    op_dict["return_history"] = bool(entry.fields.get("return_history", False))
    op_dict["cacheable"] = bool(entry.fields.get("cacheable", False))

    aggregate_table = entry.fields.get("aggregate_table")
    rowwise_columns = entry.fields.get("rowwise_columns")
    resolved_aggregate = _resolve_table_nodes(aggregate_table, user_id_to_internal)
    resolved_rowwise = _resolve_table_nodes(rowwise_columns, user_id_to_internal)

    if "structural_outputs" in entry.fields:
        op_dict["structural_outputs"] = entry.fields["structural_outputs"]
    if resolved_aggregate is not None:
        op_dict["aggregate_table"] = resolved_aggregate
    if "rowwise_template" in entry.fields:
        op_dict["rowwise_template"] = entry.fields["rowwise_template"]
    if resolved_rowwise is not None:
        op_dict["rowwise_columns"] = resolved_rowwise
    if "system_messages" in entry.fields:
        op_dict["system_messages"] = entry.fields["system_messages"]
    elif "rowwise_template" in entry.fields and isinstance(inline_messages, list):
        sys_msgs = [
            m["content"]
            for m in inline_messages
            if isinstance(m, dict)
            and m.get("role") == "system"
            and isinstance(m.get("content"), str)
        ]
        if sys_msgs:
            op_dict["system_messages"] = sys_msgs

    # For LLMChatOp, n8n adds aggregate/rowwise node refs into `_inputs`.
    # LLMVisionOp keeps `_inputs` as [msg_op] only (those refs flow via
    # rowwise_columns / image_source separately).
    llm_inputs = [msg_internal_id]
    if op_kind == "LLMChatOp":
        for table in (resolved_aggregate, resolved_rowwise):
            if not table:
                continue
            for item in table:
                if not isinstance(item, dict):
                    continue
                node_id = item.get("node")
                if isinstance(node_id, str) and node_id not in llm_inputs:
                    llm_inputs.append(node_id)
    op_dict["_inputs"] = llm_inputs


def _resolve_table_nodes(
    table: Any, user_id_to_internal: dict[str, str]
) -> list[dict[str, Any]] | None:
    """Resolve ``node:`` user-ids inside aggregate/rowwise table specs.

    Each row is copied, and a ``node`` entry that matches a known user-facing
    id gets rewritten to the corresponding internal op id. Rows without a
    ``node`` field (e.g. ``data``-only entries) pass through untouched.
    """
    if table is None:
        return None
    if not isinstance(table, list):
        raise ValueError("aggregate_table/rowwise_columns must be a list")
    out: list[dict[str, Any]] = []
    for item in table:
        if not isinstance(item, dict):
            raise ValueError("aggregate_table/rowwise_columns entries must be mappings")
        resolved = dict(item)
        node_ref = resolved.get("node")
        if isinstance(node_ref, str) and node_ref in user_id_to_internal:
            resolved["node"] = user_id_to_internal[node_ref]
        out.append(resolved)
    return out


def _build_inline_messages(
    messages: list[Any],
    *,
    prompt_spec: Any,
    entry: _OpEntry,
    user_id_to_internal: dict[str, str],
    scope: str,
    graph_ops: dict[str, dict[str, Any]],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Return (prepared_messages, implicit_format_ops).

    Two routes:

    1. ``prompt_spec`` is a mapping with ``template`` / ``format_kwargs``. The
       helper builds one implicit :class:`FormatOp` from that spec and rewrites
       every user-role message's ``content`` to the FormatOp's internal id.
       This is how aggregate / multi-ref prompts are expressed: the YAML
       author provides the raw template (including ``{df}`` / ``{ref0}``) once
       and this helper reproduces the ``FormatOp → MessageOp → LLMChatOp``
       shape n8n emits for the equivalent chain node.

    2. Otherwise fall back to :func:`_wrap_user_content_refs`, which only
       wraps bare user-id content through a trivial ``"{ref0}"`` FormatOp.
    """
    if isinstance(prompt_spec, dict):
        template = prompt_spec.get("template")
        format_kwargs = prompt_spec.get("format_kwargs", {}) or {}
        if not isinstance(template, str):
            raise ValueError(
                f"LLMChat/Vision '{entry.id}' prompt.template must be a string"
            )
        if not isinstance(format_kwargs, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in format_kwargs.items()
        ):
            raise ValueError(
                f"LLMChat/Vision '{entry.id}' prompt.format_kwargs must be "
                "dict[str, str]"
            )
        resolved_kwargs: dict[str, str] = {}
        fmt_inputs: list[str] = []
        for key, ref in format_kwargs.items():
            if ref not in user_id_to_internal:
                raise ValueError(
                    f"LLMChat/Vision '{entry.id}' prompt.format_kwargs "
                    f"references unknown id '{ref}'"
                )
            internal = user_id_to_internal[ref]
            resolved_kwargs[key] = internal
            if internal not in fmt_inputs:
                fmt_inputs.append(internal)
        fmt_id = _make_id(scope, "format", entry.id)
        fmt_op = {
            "_id": fmt_id,
            "_op": "FormatOp",
            "_max_iter": None,
            "_inputs": fmt_inputs,
            "template": template,
            "format_args": [],
            "format_kwargs": resolved_kwargs,
        }
        rewritten: list[Any] = []
        for msg in messages:
            if not isinstance(msg, dict):
                rewritten.append(msg)
                continue
            if msg.get("role") == "user":
                new_msg = dict(msg)
                new_msg["content"] = fmt_id
                rewritten.append(new_msg)
            else:
                rewritten.append(msg)
        return rewritten, [fmt_op]

    return _wrap_user_content_refs(
        messages,
        entry=entry,
        user_id_to_internal=user_id_to_internal,
        scope=scope,
        graph_ops=graph_ops,
    )


def _wrap_user_content_refs(
    messages: list[Any],
    *,
    entry: _OpEntry,
    user_id_to_internal: dict[str, str],
    scope: str,
    graph_ops: dict[str, dict[str, Any]],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Mirror n8n's prompt-wrapping for a bare user-role reference.

    When a user-role message's ``content`` is a single user-facing id that
    resolves to a non-FormatOp (e.g. an InputOp, LLMChatOp, DataRetrievalOp),
    emit an implicit FormatOp whose ``_id`` uses ``entry.id`` as the name
    component — the same scope/name pair n8n's ``_build_prompt_content`` uses
    for the equivalent chain node — and rewrite the message content to the
    FormatOp's ``_id``.

    No-ops for:
      - Contents that aren't user ids (plain strings / templates).
      - Contents that resolve to a FormatOp (already formatted upstream).
      - Non-user roles.
    """

    if not isinstance(messages, list) or not messages:
        return messages, []

    format_op: dict[str, Any] | None = None
    prepared: list[Any] = []
    for msg in messages:
        if not isinstance(msg, dict):
            prepared.append(msg)
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role != "user" or not isinstance(content, str):
            prepared.append(msg)
            continue
        op_id = user_id_to_internal.get(content)
        if op_id is None:
            prepared.append(msg)
            continue
        referenced = graph_ops.get(op_id)
        if referenced and referenced.get("_op") == "FormatOp":
            prepared.append(msg)
            continue
        # Reuse a single implicit FormatOp if multiple user messages reference
        # the same or different upstream ids. n8n only ever produces one per
        # chain node, so we follow that convention.
        if format_op is None:
            fmt_id = _make_id(scope, "format", entry.id)
            format_op = {
                "_id": fmt_id,
                "_op": "FormatOp",
                "_max_iter": None,
                "_inputs": [op_id],
                "template": "{ref0}",
                "format_args": [],
                "format_kwargs": {"ref0": op_id},
            }
        else:
            fmt_kwargs = format_op["format_kwargs"]
            key = f"ref{len(fmt_kwargs)}"
            fmt_kwargs[key] = op_id
            if op_id not in format_op["_inputs"]:
                format_op["_inputs"].append(op_id)
            # For a second distinct ref, template becomes "{ref0}{ref1}" — a
            # simple concatenation. Callers needing richer templates should
            # declare an explicit FormatOp instead.
            format_op["template"] = "".join("{" + k + "}" for k in fmt_kwargs)
        rewritten = dict(msg)
        rewritten["content"] = format_op["_id"]
        prepared.append(rewritten)

    if format_op is None:
        return messages, []
    return prepared, [format_op]


def _emit_format_op(
    op_dict: dict[str, Any], entry: _OpEntry, user_id_to_internal: dict[str, str]
) -> None:
    template = entry.fields.get("template")
    if not isinstance(template, str):
        raise ValueError(f"FormatOp '{entry.id}' requires 'template: str'")

    fmt_args = entry.fields.get("format_args", []) or []
    fmt_kwargs = entry.fields.get("format_kwargs", {}) or {}

    if not isinstance(fmt_args, list) or not all(isinstance(a, str) for a in fmt_args):
        raise ValueError(f"FormatOp '{entry.id}' format_args must be list[str]")
    if not isinstance(fmt_kwargs, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in fmt_kwargs.items()
    ):
        raise ValueError(f"FormatOp '{entry.id}' format_kwargs must be dict[str, str]")

    resolved_args = [user_id_to_internal.get(a, a) for a in fmt_args]
    resolved_kwargs = {k: user_id_to_internal.get(v, v) for k, v in fmt_kwargs.items()}

    op_dict["template"] = template
    op_dict["format_args"] = resolved_args
    op_dict["format_kwargs"] = resolved_kwargs


def _emit_lambda_op(op_dict: dict[str, Any], entry: _OpEntry) -> None:
    fn_name = entry.fields.get("fn_name")
    code = entry.fields.get("code") or entry.fields.get("_code")
    if not isinstance(fn_name, str):
        raise ValueError(f"LambdaOp '{entry.id}' requires 'fn_name: str'")
    if not isinstance(code, str):
        raise ValueError(f"LambdaOp '{entry.id}' requires 'code: str'")
    op_dict["fn_name"] = fn_name
    op_dict["_code"] = code


def _build_generation_config(
    config: dict[str, Any], entry_id: str, op_kind: str = "LLMChatOp"
) -> dict[str, Any]:
    # Mirror the subset that n8n parser produces; rest pass through.
    allowed = {
        "model",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "max_tokens",
        "n",
        "presence_penalty",
        "seed",
        "stop",
        "stream",
        "stream_options",
        "temperature",
        "top_p",
        "ignore_eos",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(
            f"{op_kind} '{entry_id}' config has unknown fields: {sorted(unknown)}"
        )
    return dict(config)


_ENV_PLACEHOLDERS: dict[str, Any] = {
    "${DATABASE_URL}": lambda: envs.DATABASE_URL,
    "${S3_URL}": lambda: envs.S3_WORKER_URL,
}


def _expand_env_placeholders(value: str) -> str:
    """Resolve ``${DATABASE_URL}`` / ``${S3_URL}`` sentinel values.

    Parsers bake ``envs.DATABASE_URL`` / ``envs.S3_WORKER_URL`` into each
    ``DataRetrievalOp.data_spec.connection_string`` at parse time. To keep the
    YAML portable across machines (CI, laptops, containers) the same ``envs``
    values are substituted here when a YAML author writes the sentinel as the
    literal connection string.
    """
    for placeholder, getter in _ENV_PLACEHOLDERS.items():
        if value == placeholder:
            actual = getter()
            if actual is None:
                raise ValueError(
                    f"YAML connection_string uses {placeholder} but the "
                    "corresponding envs value is unset"
                )
            return str(actual)
    return value


def _op_id_prefix(op_type: str) -> str:
    mapping = {
        "DataOp": "data",
        "DataRetrievalOp": "retrieval",
        "MessageOp": "message",
        "LLMChatOp": "llm",
        "LLMVisionOp": "llmvision",
        "ImageGenerationOp": "imagegen",
        "FormatOp": "format",
        "LambdaOp": "lambda",
    }
    return mapping.get(op_type, op_type.lower())
