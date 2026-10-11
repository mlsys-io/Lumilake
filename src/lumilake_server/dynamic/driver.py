"""Validation and graph-assembly helpers for dynamic workflow rounds.

A round is a single job whose DAG is ``[emitted subgraph] -> [observation
LambdaOp] -> [proposer LLMChatOp]``. The proposer reads the goal, all prior
observations, and the overall topology, and emits a structured plan selecting
the next subgraph (a small acyclic DAG of ops) or ``STOP``. These helpers
validate the emitted subgraph, assemble the fused round graph, and validate
the returned plan.
"""

import ast
import copy
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from lumilake_server import ops as ops_pkg
from lumilake_server.common import retrieval_items_path
from lumilake_server.dynamic.blocks import (
    INPUT_NODE_ID,
    OBSERVATION_NODE_ID,
    PROPOSER_NODE_ID,
    fused_round_graph,
)
from lumilake_server.ops.data_ops import DataRetrievalOp
from lumilake_server.ops.llm_ops import LLMOp
from lumilake_server.parser.common import make_id
from lumilake_server.parser.yaml_parser import (
    SUPPORTED_OPS,
    _op_id_prefix,
    parse_yaml_payload,
)
from lumilake_server.runtime.runtime_manager.flowmesh import (
    _coerce_output_value,
    _walk_output_path,
)

_OBSERVATION_TEMPLATE_FILE = Path(__file__).with_name("observation_template.py")


def observation_lambda(*, preview_width: int = 900) -> str:
    """Load the observation LambdaOp source for ``preview_width`` chars.

    Reads the standalone ``observation_template.py`` and substitutes the
    preview character budget into its ``{width}`` placeholder.
    """
    source = _OBSERVATION_TEMPLATE_FILE.read_text()
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        raise DriverProtocolError(
            f"observation template is not valid Python: {exc}"
        ) from exc
    functions = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "observe"
    ]
    if len(functions) != 1:
        raise DriverProtocolError(
            "observation template must define exactly one top-level observe function"
        )
    target = functions[0]
    start_line = target.lineno
    if target.decorator_list:
        start_line = min(decorator.lineno for decorator in target.decorator_list)
    lines = source.splitlines(keepends=True)
    function_source = "".join(lines[start_line - 1 : target.end_lineno])
    return function_source.replace("{width}", str(preview_width))


STOP = "STOP"
SUBGRAPH = "subgraph"

FIXED_ROUND_NODE_IDS: frozenset[str] = frozenset(
    {
        INPUT_NODE_ID,
        OBSERVATION_NODE_ID,
        PROPOSER_NODE_ID,
    }
)

_OP_CLASSES: dict[str, type] = {
    name: obj for name, obj in vars(ops_pkg).items() if isinstance(obj, type)
}
_MISSING_OP_CLASSES = SUPPORTED_OPS - _OP_CLASSES.keys()
if _MISSING_OP_CLASSES:
    raise RuntimeError(
        f"op types not exported by lumilake_server.ops: {sorted(_MISSING_OP_CLASSES)}"
    )
_OUTPUT_SOURCE_OPS: frozenset[str] = frozenset(
    name
    for name in SUPPORTED_OPS
    if issubclass(_OP_CLASSES[name], (LLMOp, DataRetrievalOp))
)


class DriverProtocolError(Exception):
    """Raised when a job result or driver configuration violates the contract.

    Distinct from :class:`RuntimeError` so callers can distinguish a malformed
    plan/config from a job that legitimately failed.
    """


class StopPlan(BaseModel):
    """Validated plan indicating the loop should halt with ``STOP``."""

    model_config = ConfigDict(extra="forbid")


class SubgraphPlan(BaseModel):
    """Validated plan selecting the next round's subgraph."""

    model_config = ConfigDict(extra="forbid")

    ops: list[dict[str, Any]]
    export: list[str] = []


class _RawPlan(BaseModel):
    """Structural shape of an untrusted plan dict."""

    model_config = ConfigDict(extra="forbid")

    next: str
    ops: list[dict[str, Any]] | None = None
    export: list[str] = []


class RoundBuild(BaseModel):
    """Typed result of assembling one round.

    ``graph`` is the native round graph; ``leaf_output_names`` lists the
    archived ``leaf_<internal_id>`` output names in deterministic order, so the
    route can validate that every expected leaf produced a result.
    ``export_output_names`` lists the archived ``export_<internal_id>`` output
    names for exported non-leaf ops. The ``*_user_ids`` lists run parallel to
    the output-name lists and give each archived output's subgraph user id, so
    the route can key stored results by the id later rounds reference.
    ``forwarded_inputs`` maps each forwarded workflow input name to its literal
    values; the route passes these to the child job dispatch alongside the
    Symbols input.
    """

    model_config = ConfigDict(extra="forbid")

    graph: dict[str, Any]
    leaf_output_names: list[str]
    leaf_user_ids: list[str]
    export_output_names: list[str] = []
    export_user_ids: list[str] = []
    forwarded_inputs: dict[str, list[str]] = {}


def round_output_location(
    base: dict[str, Any], run_namespace: str, round_index: int
) -> dict[str, Any]:
    """Derive a distinct output location for one round within a run.

    Folder outputs use fixed item names, so reusing one prefix would overwrite
    the previous round's export.
    """
    location_type = base["type"]
    if location_type == "s3":
        prefix = base["prefix"]
        sep = "" if prefix.endswith("/") else "/"
        return {
            "type": "s3",
            "prefix": f"{prefix}{sep}{run_namespace}/round-{round_index}/",
        }
    raise DriverProtocolError(f"unsupported output location type {location_type!r}")


def validate_library(library: dict[str, dict[str, Any]] | None) -> None:
    """Validate the reference library of pre-configured op templates.

    Each entry must be a mapping whose ``op`` type is a known op type. Raises
    :class:`DriverProtocolError` on any violation.
    """
    if not library:
        return
    for ref, template in library.items():
        if not isinstance(template, dict):
            raise DriverProtocolError(f"library entry {ref!r} must be a mapping")
        op_type = template.get("op")
        if not isinstance(op_type, str) or op_type not in SUPPORTED_OPS:
            raise DriverProtocolError(
                f"library entry {ref!r} has unsupported op type {op_type!r}; "
                f"supported: {sorted(SUPPORTED_OPS)}"
            )


_LIBRARY_REF_OVERRIDES = frozenset({"id", "inputs", "op"})


def _apply_param_bindings(
    template: dict[str, Any], ref: str, inputs: list[Any]
) -> dict[str, Any]:
    """Substitute the planner's ``inputs`` into the template's param slots.

    A library retrieval template declares ``param_bindings``: a list of
    ``{"label": <param label>, "input": <index into inputs>}``. Each binding
    sets the named param's ``node`` to the corresponding input, so the planner
    wires a retrieval to a prior-round node purely through ``inputs`` without
    mutating the template's config. The input count must match the declared
    binding count exactly, and every binding label must exist in the template's
    params.
    """
    data_spec = template.get("data_spec")
    if not isinstance(data_spec, dict):
        raise DriverProtocolError(
            f"library entry {ref!r} is not a retrieval op; only retrieval ops "
            "declare param bindings"
        )
    bindings = data_spec.get("param_bindings")
    if not isinstance(bindings, list):
        return data_spec
    if len(inputs) != len(bindings):
        raise DriverProtocolError(
            f"library entry {ref!r} declares {len(bindings)} param binding(s) "
            f"but got {len(inputs)} input(s)"
        )
    params = data_spec.get("params")
    if not isinstance(params, list):
        raise DriverProtocolError(
            f"library entry {ref!r} has param_bindings but no params"
        )
    by_label = {param.get("label"): dict(param) for param in params}
    for binding in bindings:
        if not isinstance(binding, dict):
            raise DriverProtocolError(
                f"library entry {ref!r} param_bindings entries must be mappings"
            )
        label = binding.get("label")
        slot = binding.get("input")
        if label not in by_label:
            raise DriverProtocolError(
                f"library entry {ref!r} param_binding references unknown "
                f"label {label!r}; params: {sorted(by_label)}"
            )
        if not isinstance(slot, int) or not 0 <= slot < len(inputs):
            raise DriverProtocolError(
                f"library entry {ref!r} param_binding for {label!r} has "
                f"invalid input slot {slot!r}"
            )
        by_label[label]["node"] = inputs[slot]
    return {**data_spec, "params": list(by_label.values())}


def resolve_subgraph(
    subgraph: list[dict[str, Any]],
    library: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Expand library references in an emitted subgraph into full op configs.

    An op that carries a ``ref`` key names a library template; the resolved op
    is the template's config with the emitted op's ``id`` and ``inputs`` applied
    on top. ``inputs`` wires the template's declared param slots (see
    :func:`_apply_param_bindings`); a declared ``op`` is asserted to match the
    template and ignored during merge. The planner may not override the
    template's other config (``data_spec``, ``model``, etc.). Ops without a
    ``ref`` pass through unchanged (fully-inline ops are allowed), so this is
    idempotent.
    """
    resolved: list[dict[str, Any]] = []
    for op in subgraph:
        if "ref" not in op:
            resolved.append(op)
            continue
        ref = op["ref"]
        if not isinstance(ref, str):
            raise DriverProtocolError(
                f"subgraph op {op.get('id')!r} 'ref' must be a string"
            )
        if not library or ref not in library:
            raise DriverProtocolError(
                f"subgraph op {op.get('id')!r} references unknown library "
                f"entry {ref!r}"
            )
        template = library[ref]
        merged = dict(template)
        declared_op = op.get("op")
        if declared_op is not None and merged.get("op") != declared_op:
            raise DriverProtocolError(
                f"subgraph op {op.get('id')!r} declares op {declared_op!r} "
                f"but library entry {ref!r} is {merged.get('op')!r}"
            )
        for key, value in op.items():
            if key == "ref" or key == "op":
                continue
            if key not in _LIBRARY_REF_OVERRIDES:
                raise DriverProtocolError(
                    f"subgraph op {op.get('id')!r} may not override "
                    f"{key!r} when referencing library entry {ref!r}; only "
                    "id and inputs may vary"
                )
            if key == "inputs":
                if not isinstance(value, list) or not all(
                    isinstance(v, str) for v in value
                ):
                    raise DriverProtocolError(
                        f"subgraph op {op.get('id')!r} inputs must be a list "
                        "of node ids"
                    )
                merged["inputs"] = value
                if isinstance(template.get("data_spec"), dict):
                    merged["data_spec"] = _apply_param_bindings(template, ref, value)
                continue
            merged[key] = value
        resolved.append(merged)
    return resolved


def _default_forward_path(op_type: str, op: dict[str, Any]) -> str:
    """The result path the runtime reads for an implicit reference to an op."""
    if op_type == "DataRetrievalOp":
        mode = (
            op.get("data_spec", {}).get("mode")
            if isinstance(op.get("data_spec"), dict)
            else None
        )
        return retrieval_items_path(mode) if isinstance(mode, str) else "items.output"
    return "items.output"


def _forward_values(
    items: list[dict[str, Any]], parts: tuple[str, ...], ref: str
) -> list[str]:
    """Walk each stored whole item with the runtime walker and coerce to strings."""
    return [_coerce_output_value(_walk_output_path(item, parts, ref)) for item in items]


def _forward_values_flattened(
    items: list[dict[str, Any]], parts: tuple[str, ...], ref: str
) -> list[str]:
    """Walk each stored whole item and flatten the result to one string per
    element (strings as-is, ``json.dumps`` otherwise)."""
    values: list[str] = []
    for item in items:
        walked = _walk_output_path(item, parts, ref)
        if isinstance(walked, list):
            values.extend(_coerce_output_value(v) for v in walked)
        else:
            values.append(_coerce_output_value(walked))
    return values


def _forward_values_per_row(
    items: list[dict[str, Any]],
    parts: tuple[str, ...],
    ref: str,
    consumer: str,
    rows: int,
) -> list[str]:
    """Walk each stored whole item to one string and align it with ``rows``.

    Each run row is its own input slice, so a slice carries exactly one value
    per input. The stored result must hold exactly one item per row (item ``i``
    feeds row ``i``); a result with any other item count, or a walked value that
    is a list of other than one element, cannot be matched to the rows and is
    rejected rather than mis-aligned.
    """
    values: list[str] = []
    for index, item in enumerate(items):
        walked = _walk_output_path(item, parts, ref)
        if isinstance(walked, list):
            if len(walked) != 1:
                raise DriverProtocolError(
                    f"op {consumer!r} references {ref!r}: item {index} holds "
                    f"{len(walked)} values at {'.'.join(parts)!r}, but a "
                    f"{rows}-row run binds exactly one value per row"
                )
            walked = walked[0]
        values.append(_coerce_output_value(walked))
    if len(values) != rows:
        raise DriverProtocolError(
            f"op {consumer!r} references {ref!r}: its result has "
            f"{len(values)} items for a {rows}-row run, so its items cannot "
            "be matched to rows"
        )
    return values


def _walk_values_per_row(
    items: list[dict[str, Any]],
    parts: tuple[str, ...],
    ref: str,
    consumer: str,
    rows: int,
) -> list[Any]:
    """Walk each stored whole item to its raw per-row value.

    Like :func:`_forward_values_per_row` but returns the walked values
    uncoerced, so a caller can tell a per-row scalar from a per-row list (a
    grouped reference) before deciding how to forward it.
    """
    values: list[Any] = []
    for index, item in enumerate(items):
        walked = _walk_output_path(item, parts, ref)
        if isinstance(walked, list) and len(walked) == 1:
            walked = walked[0]
        values.append(walked)
    if len(values) != rows:
        raise DriverProtocolError(
            f"op {consumer!r} references {ref!r}: its result has "
            f"{len(values)} items for a {rows}-row run, so its items cannot "
            "be matched to rows"
        )
    return values


def walk_archived_items(
    archived: dict[str, list[str]],
    configs: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    """Walk each archived whole item with its producer's default path.

    Leaf and export outputs archive the whole result item (``path: items``).
    The observation and stored round results must see the same text value an
    in-graph reference to that producer would read, so each item is walked with
    the producer's default forward path and coerced to a string.
    """
    walked: dict[str, list[str]] = {}
    for node_id, values in archived.items():
        config = configs.get(node_id, {})
        op_type = config.get("op", "")
        default_path = _default_forward_path(op_type, config)
        parts = tuple(default_path[len("items.") :].split("."))
        items: list[dict[str, Any]] = []
        for value in values:
            try:
                items.append(json.loads(value))
            except (TypeError, ValueError) as exc:
                raise DriverProtocolError(
                    f"archived output for {node_id!r} is not a JSON item: " f"{value!r}"
                ) from exc
        walked[node_id] = _forward_values(items, parts, node_id)
    return walked


def forward_refs(
    subgraph: list[dict[str, Any]],
    results: dict[str, list[dict[str, Any]]],
    configs: dict[str, dict[str, Any]],
    rows: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Rewrite a round's subgraph so references to earlier-round ops bind as
    workflow inputs carrying exactly the value the in-graph reference would
    have read.

    ``results`` maps an earlier op id to its decoded whole items (archived via
    a ``path: items`` output); ``configs`` is the node registry of resolved op
    configs. Returns the deep-copied, rewritten ops plus the workflow ``inputs``
    block (name -> literal values).

    Implicit references (``inputs`` entries, ``prompt.format_kwargs`` values,
    ``image_source``) bind one input named after the producer op id, leaving the
    reference unchanged. Explicit references (``aggregate_table`` columns and
    ``data_spec.params`` with a ``path``) bind one input per ``(node, path)`` and
    rewrite the reference to point at that input with ``path: ""``.

    With ``rows > 1`` the run has one input slice per row, so every bound input
    holds exactly one value per row, taken from the stored items in order; a
    result that cannot be matched to the rows raises :class:`DriverProtocolError`
    naming the consuming op and the reference.
    """
    inputs: dict[str, list[str]] = {}
    ops: list[dict[str, Any]] = []
    materializers: list[dict[str, Any]] = []
    materialized: set[str] = set()

    def add_materializer(ref: str, path: str, walked: list[Any]) -> str:
        """Bind a JSON input per row and append a row-aligned LambdaOp that
        decodes each row's JSON back to its list; returns the materializer id."""
        input_id = f"{ref} | {path}"
        if input_id not in materialized:
            inputs[input_id] = [_coerce_output_value(v) for v in walked]
            materializers.append(
                {
                    "id": f"{input_id} | rows",
                    "op": "LambdaOp",
                    "inputs": [input_id],
                    "fn_name": "decode_rows",
                    "code": (
                        "def decode_rows(columns):\n"
                        "    return [json.loads(v) for v in columns[0]]\n"
                    ),
                    "mode": "aligned",
                }
            )
            materialized.add(input_id)
        return f"{input_id} | rows"

    def bind_implicit(
        ref: str, consumer: str, *, image_source: bool = False
    ) -> str | None:
        """Bind an implicit reference; returns the rewritten ref (a materializer
        id) when the per-row value is a grouped list, else None to leave the
        reference unchanged."""
        if ref not in results:
            return None
        op_type = configs[ref].get("op", "")
        default_path = _default_forward_path(op_type, configs[ref])
        parts = tuple(default_path[len("items.") :].split("."))
        if rows > 1:
            walked = _walk_values_per_row(results[ref], parts, ref, consumer, rows)
            if any(isinstance(v, list) for v in walked):
                if not all(isinstance(v, list) for v in walked):
                    raise DriverProtocolError(
                        f"op {consumer!r} references {ref!r}: some rows hold a "
                        "list and some a scalar at "
                        f"{'.'.join(parts)!r}, which cannot be forwarded"
                    )
                if image_source:
                    raise DriverProtocolError(
                        f"op {consumer!r} image_source {ref!r} holds a list per "
                        "row and cannot be forwarded as JSON"
                    )
                return add_materializer(ref, default_path, walked)
            inputs[ref] = [_coerce_output_value(v) for v in walked]
        else:
            inputs[ref] = _forward_values(results[ref], parts, ref)
        return None

    def bind_explicit(ref: str, path: str, consumer: str) -> tuple[str, str]:
        """Forward an explicit ``(node, path)`` reference, returning the
        ``(node, path)`` the consumer should read.

        A scalar per row binds one input per ``(ref, path)`` and keeps the
        reference's path (``""`` after rewrite). A grouped reference — a walked
        per-row value that is a list — binds a JSON input per row plus a
        list-mode LambdaOp materializer that decodes each row's JSON back to
        its list, and the consumer reads the materializer's ``items.output`` so
        it sees one group per row.
        """
        input_id = f"{ref} | {path}"
        if input_id in materialized:
            return f"{input_id} | rows", "items.output"
        if input_id not in inputs:
            parts = tuple(path[len("items.") :].split("."))
            if rows > 1:
                walked = _walk_values_per_row(results[ref], parts, ref, consumer, rows)
                if any(isinstance(v, list) for v in walked):
                    if not all(isinstance(v, list) for v in walked):
                        raise DriverProtocolError(
                            f"op {consumer!r} references {ref!r}: some rows hold "
                            "a list and some a scalar at "
                            f"{'.'.join(parts)!r}, which cannot be forwarded"
                        )
                    return add_materializer(ref, path, walked), "items.output"
                inputs[input_id] = [_coerce_output_value(v) for v in walked]
            else:
                inputs[input_id] = _forward_values_flattened(results[ref], parts, ref)
        return input_id, ""

    for op in subgraph:
        op = copy.deepcopy(op)
        consumer = str(op.get("id"))
        for index, ref in enumerate(op.get("inputs", [])):
            if isinstance(ref, str):
                rewritten = bind_implicit(ref, consumer)
                if rewritten is not None:
                    op["inputs"][index] = rewritten
        prompt = op.get("prompt")
        if isinstance(prompt, dict):
            format_kwargs = prompt.get("format_kwargs")
            if isinstance(format_kwargs, dict):
                for key, ref in format_kwargs.items():
                    if isinstance(ref, str):
                        rewritten = bind_implicit(ref, consumer)
                        if rewritten is not None:
                            format_kwargs[key] = rewritten
        format_kwargs = op.get("format_kwargs")
        if isinstance(format_kwargs, dict):
            for key, ref in format_kwargs.items():
                if isinstance(ref, str):
                    rewritten = bind_implicit(ref, consumer)
                    if rewritten is not None:
                        format_kwargs[key] = rewritten
        image_source = op.get("image_source")
        if isinstance(image_source, str):
            bind_implicit(image_source, consumer, image_source=True)
        aggregate_table = op.get("aggregate_table")
        if isinstance(aggregate_table, list):
            for row in aggregate_table:
                if not isinstance(row, dict):
                    continue
                node = row.get("node")
                path = row.get("path")
                if isinstance(node, str) and isinstance(path, str) and node in results:
                    row["node"], row["path"] = bind_explicit(node, path, consumer)
        rowwise_columns = op.get("rowwise_columns")
        if isinstance(rowwise_columns, list):
            for column in rowwise_columns:
                if not isinstance(column, dict):
                    continue
                node = column.get("node")
                path = column.get("path")
                if isinstance(node, str) and isinstance(path, str) and node in results:
                    column["node"], column["path"] = bind_explicit(node, path, consumer)
        data_spec = op.get("data_spec")
        if isinstance(data_spec, dict):
            params = data_spec.get("params")
            if isinstance(params, list):
                for param in params:
                    if not isinstance(param, dict):
                        continue
                    node = param.get("node")
                    path = param.get("path")
                    if (
                        isinstance(node, str)
                        and isinstance(path, str)
                        and node in results
                    ):
                        param["node"], param["path"] = bind_explicit(
                            node, path, consumer
                        )
        ops.append(op)
    ops.extend(materializers)
    return ops, inputs


def _validate_declared_inputs(
    op_id: str,
    op_type: str,
    op: dict[str, Any],
    inputs: list[Any],
    available_ops: dict[str, dict[str, Any]],
) -> None:
    """Reject a mismatch between declared inputs and the native dependency set.

    The parser is the source of truth for what an op consumes. We build the op
    through the same ``parse_yaml_payload`` path ``build_round`` uses, then
    compare the declared ``inputs`` set against the parser-derived native
    dependency set in BOTH directions: a declared input the native config does
    not consume is rejected, and a native reference missing from declared
    inputs is rejected. If the op config references an undeclared node, the
    parser raises ``unknown id``, which we surface as a native-but-undeclared
    rejection.
    """
    declared = {ref for ref in inputs if ref != INPUT_NODE_ID}
    probe_ops: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_probe(ref: str) -> None:
        if ref == INPUT_NODE_ID or ref == op_id or ref in seen:
            return
        seen.add(ref)
        real = available_ops.get(ref)
        if real is None:
            probe_ops.append({"id": ref, "op": "DataOp", "inputs": [], "data": ["x"]})
            return
        probe_op = dict(real)
        probe_op["inputs"] = [dep for dep in real.get("inputs", []) if dep != op_id]
        probe_ops.append(probe_op)
        for dep in probe_op["inputs"]:
            if isinstance(dep, str):
                add_probe(dep)

    for ref in declared:
        add_probe(ref)
    workflow = {
        "name": "depcheck",
        "inputs": {INPUT_NODE_ID: []},
        "ops": [op, *probe_ops],
        "outputs": [],
    }
    try:
        parsed = parse_yaml_payload(workflow)
    except ValueError as exc:
        raise DriverProtocolError(
            f"subgraph op {op_id!r} references a node not in its declared "
            f"inputs: {exc}"
        ) from exc
    graph_name = next(iter(parsed))
    spec = parsed[graph_name]
    native = spec["graph"]
    internal_to_user: dict[str, str] = {}
    for probe_op in [op, *probe_ops]:
        uid = probe_op.get("id")
        otype = probe_op.get("op")
        if isinstance(uid, str) and isinstance(otype, str):
            internal_to_user[_internal_id(graph_name, otype, uid)] = uid

    native_op = native.get(_internal_id(graph_name, op_type, op_id))
    if native_op is None:
        raise DriverProtocolError(
            f"subgraph op {op_id!r} did not produce a native node"
        )

    def collect_config_refs(node: dict[str, Any]) -> set[str]:
        refs: set[str] = set()
        for key, value in node.items():
            if key.startswith("_"):
                continue
            if isinstance(value, str):
                refs.add(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        refs.add(item)
                    elif isinstance(item, dict):
                        refs.update(collect_config_refs(item))
            elif isinstance(value, dict):
                refs.update(collect_config_refs(value))
        return refs

    def resolve_to_user(internal: str) -> set[str]:
        user = internal_to_user.get(internal)
        if user is not None:
            return {user}
        implicit = native.get(internal)
        if implicit is None:
            return set()
        resolved: set[str] = set()
        for dep in implicit.get("_inputs", []):
            if isinstance(dep, str):
                resolved.update(resolve_to_user(dep))
        return resolved

    config_refs = collect_config_refs(native_op)
    if not any(internal in native for internal in config_refs):
        return
    native_deps: set[str] = set()
    for internal in config_refs:
        native_deps.update(resolve_to_user(internal))
    native_deps.discard(INPUT_NODE_ID)

    for ref in declared:
        if ref not in native_deps:
            raise DriverProtocolError(
                f"subgraph op {op_id!r} declares input {ref!r} but its "
                f"{op_type} config does not reference it"
            )
    for ref in native_deps:
        if ref not in declared:
            raise DriverProtocolError(
                f"subgraph op {op_id!r} config references {ref!r} but it is "
                f"missing from the declared inputs"
            )


def validate_emitted_subgraph(
    subgraph: list[dict[str, Any]],
    node_registry: dict[str, dict[str, Any]],
    max_nodes: int,
    library: dict[str, dict[str, Any]] | None = None,
    *,
    export: list[str] | None = None,
    results: dict[str, list[dict[str, Any]]] | None = None,
) -> None:
    """Structurally validate a planner-emitted subgraph.

    Enforces: node count within ``max_nodes``, known op types (or a ``ref`` to
    a known library entry), unique ids that do not collide with the fixed round
    graph, explicit inputs that reference either a prior-round node (in
    ``node_registry``) or another node in this subgraph, and acyclicity. Raises
    :class:`DriverProtocolError` on any violation.

    ``node_registry`` maps prior-round node ids to their RESOLVED op configs,
    so a cross-round reference resolves to the real op (with its real type),
    not a stub. ``results`` maps the earlier-round op ids whose results are
    available (leaves and exported ops); a reference to an earlier op whose
    result is not available is rejected rather than recomputed. ``export``
    names ops of this subgraph that later rounds may reference; each must be
    an op of this subgraph.
    """
    if len(subgraph) > max_nodes:
        raise DriverProtocolError(
            f"subgraph exceeds max_nodes_per_round={max_nodes}: got {len(subgraph)}"
        )
    if not subgraph:
        raise DriverProtocolError("subgraph must not be empty")

    resolved = resolve_subgraph(subgraph, library)

    ids: set[str] = set()
    types: dict[str, str] = {}
    available_ops: dict[str, dict[str, Any]] = {
        op["id"]: op for op in resolved if isinstance(op.get("id"), str)
    }
    available_ops.update(node_registry)
    for index, op in enumerate(resolved):
        if not isinstance(op, dict):
            raise DriverProtocolError(f"subgraph op at index {index} must be a mapping")
        op_id = op.get("id")
        if not isinstance(op_id, str) or not op_id:
            raise DriverProtocolError(
                f"subgraph op at index {index} requires a non-empty 'id'"
            )
        if op_id in FIXED_ROUND_NODE_IDS:
            raise DriverProtocolError(
                f"subgraph op id {op_id!r} is reserved for the fused round graph"
            )
        if op_id in node_registry:
            raise DriverProtocolError(
                f"subgraph op id {op_id!r} collides with an existing node; "
                "nodes are immutable once created"
            )
        if op_id in ids:
            raise DriverProtocolError(f"duplicate subgraph op id {op_id!r}")
        ids.add(op_id)
        op_type = op.get("op")
        if not isinstance(op_type, str) or op_type not in SUPPORTED_OPS:
            raise DriverProtocolError(
                f"subgraph op {op_id!r} has unsupported op type {op_type!r}; "
                f"supported: {sorted(SUPPORTED_OPS)}"
            )
        types[op_id] = op_type
        raw_inputs = op.get("inputs", [])
        if not isinstance(raw_inputs, list) or not all(
            isinstance(ref, str) for ref in raw_inputs
        ):
            raise DriverProtocolError(
                f"subgraph op {op_id!r} inputs must be a list of node ids"
            )
        _validate_declared_inputs(op_id, op_type, op, raw_inputs, available_ops)

    if export is not None:
        if not isinstance(export, list) or not all(isinstance(e, str) for e in export):
            raise DriverProtocolError("'export' must be a list of op ids")
        for exported_id in export:
            if exported_id not in ids:
                raise DriverProtocolError(
                    f"exported op id {exported_id!r} is not an op of this subgraph"
                )

    available_results = set(results) if results is not None else None
    seen: set[str] = set()
    for op in resolved:
        op_id = op["id"]
        for ref in op.get("inputs", []):
            if ref == INPUT_NODE_ID:
                continue
            if ref in node_registry:
                if available_results is not None and ref not in available_results:
                    raise DriverProtocolError(
                        f"subgraph op {op_id!r} references earlier op {ref!r}, "
                        "whose result is not available (it was neither a leaf "
                        "nor exported); no recompute is allowed"
                    )
                continue
            if ref not in ids:
                raise DriverProtocolError(
                    f"subgraph op {op_id!r} references unknown node id {ref!r}"
                )
            if ref not in seen:
                raise DriverProtocolError(
                    f"subgraph op {op_id!r} references {ref!r} which is not "
                    "emitted before it; subgraph must be acyclic"
                )
        seen.add(op_id)

    consumed: set[str] = set()
    for op in resolved:
        for ref in op.get("inputs", []):
            if isinstance(ref, str):
                consumed.add(ref)
    for op in resolved:
        op_id = op["id"]
        if op_id in consumed:
            continue
        if types[op_id] not in _OUTPUT_SOURCE_OPS:
            raise DriverProtocolError(
                f"subgraph leaf {op_id!r} is a {types[op_id]}, which cannot be "
                "archived; every leaf must be an LLMOp or DataRetrievalOp"
            )


_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _strip_plan_wrappers(value: str) -> str:
    """Strip a reasoning model's wrappers from around the plan JSON.

    A reasoning model emits its chain of thought in a ``<think>`` block and
    often fences the answer; both wrap the plan without changing it.
    """
    stripped = _THINK_BLOCK_RE.sub("", value)
    fenced = _FENCE_RE.match(stripped)
    return fenced.group(1) if fenced else stripped.strip()


def system_message(
    goal: str,
    observations: list[str],
    topology: list[str],
    threshold: float | None = None,
    library: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Render the proposer system message: goal + prior observations + topology."""
    lines = [
        "You are the planner for a dynamic analysis workflow. Each round you "
        "emit a small acyclic subgraph of ops that advances the goal, or STOP "
        "when the accumulated evidence is sufficient.",
        "",
        f"GOAL: {goal}",
    ]
    if observations:
        lines.append("")
        lines.append("PRIOR OBSERVATIONS:")
        for index, observation in enumerate(observations):
            lines.append(f"--- observation {index} ---")
            lines.append(observation)
    lines.append("")
    lines.append("AVAILABLE NODES (existing, immutable, reference by id):")
    lines.append(", ".join([INPUT_NODE_ID, *topology]))
    lines.append(
        f"{INPUT_NODE_ID} is the run's input node, carrying the symbol under "
        "analysis; wire it to any op that needs the symbol."
    )
    lines.append(
        'Every "id" you emit must be NEW and must not be any id already '
        "listed in AVAILABLE NODES. To use an existing node, name it in "
        '"inputs" only — never declare it again as an op. For example, if '
        "top_sector_1 already exists, the correct plan is a single op "
        '{"id": "peers_in_sector_1", "ref": "peers_in_sector", '
        '"inputs": ["top_sector_1"]} — not one that also declares '
        "top_sector_1 again."
    )
    lines.append("")
    lines.append(
        'Emit a structured plan: either {"next": "STOP"} or '
        '{"next": "subgraph", "ops": [...]} where each op is '
        '{"id": <unique id>, "op": <op type>, "inputs": [<node ids>], '
        "...fields}. Each op's inputs must reference an existing node id or "
        "another op id in the same subgraph, and the subgraph must be acyclic. "
        "Available op types: " + ", ".join(sorted(SUPPORTED_OPS)) + "."
    )
    if library:
        lines.append("")
        lines.append("REFERENCE LIBRARY (pre-configured op templates):")
        for ref, template in library.items():
            op_type = template.get("op", "?")
            lines.append(f"- {ref}: {op_type}")
            for key, value in template.items():
                if key == "op":
                    continue
                rendered = str(value).replace("\n", " ")
                lines.append(f"    {key}: {rendered[:200]}")
            lines.append(
                f'    To use it, emit {{"id": <unique id>, "ref": "{ref}", '
                '"inputs": [<node ids>]} — key "ref", never "op"; the '
                "template fills in the rest."
            )
            bindings = template.get("data_spec", {}).get("param_bindings")
            if isinstance(bindings, list) and bindings:
                lines.append(
                    "    Its inputs wire its declared param slots in order; "
                    "the template config is otherwise immutable."
                )
    if threshold is not None:
        lines.append("")
        lines.append(
            f"Stop when the observed evidence meets this sufficiency threshold: "
            f"{threshold}."
        )
    return "\n".join(lines)


def planner_messages(
    *,
    goal: str,
    observations: list[str],
    topology: list[str],
    threshold: float | None,
    library: dict[str, dict[str, Any]] | None,
    has_subgraph: bool,
    round_observation: str | None = None,
) -> tuple[str, str]:
    """Render the planner system and user messages.

    The system message carries the goal, prior observations, and topology; the
    user message asks for the first subgraph or for the next one based on the
    last round's observation, followed by ``round_observation`` when given. The
    in-graph proposer renders the same text (its observation is appended inside
    the graph); the server-side API planner passes it here.
    """
    system = system_message(goal, observations, topology, threshold, library)
    user = (
        "Emit the first subgraph that starts to advance the goal."
        if not has_subgraph
        else "Here is the observation from the last round. Based on it, emit "
        "the next subgraph, or STOP."
    )
    if round_observation is not None:
        user = f"{user}\n\n{round_observation}"
    return system, user


def compute_observation(leaf_outputs: dict[str, list[str]], preview_width: int) -> str:
    """Compute the round observation string from the archived leaf outputs.

    The observation LambdaOp runs in-graph to feed the proposer, but its output
    is not an archived output (only LLMOp/DataRetrievalOp can be). The server
    recomputes the same compact summary from the leaf outputs (archived via the
    per-leaf OutputOps) so it can be accumulated across rounds.
    """
    source = observation_lambda(preview_width=preview_width)
    namespace: dict[str, Any] = {"json": json}
    exec(source, namespace)  # noqa: S102 - server-authored template
    observe = namespace["observe"]
    per_leaf: list[Any] = [leaf_outputs[leaf_id] for leaf_id in sorted(leaf_outputs)]
    return observe(per_leaf)


def result_outputs(result: Any) -> dict[str, Any]:
    """Extract and validate the nested outputs envelope from a job result."""
    if not isinstance(result, dict):
        raise DriverProtocolError(
            f"job result envelope must be a dict, got {type(result).__name__}"
        )
    nested = result.get("result")
    if not isinstance(nested, dict):
        raise DriverProtocolError(
            "job result envelope must contain a dict 'result' field"
        )
    outputs = nested.get("outputs")
    if not isinstance(outputs, dict):
        raise DriverProtocolError(
            "job result 'result' field must contain a dict 'outputs' field"
        )
    return outputs


def validate_plan(raw_outputs: dict[str, Any]) -> StopPlan | SubgraphPlan:
    """Parse and validate the proposer's plan from a job result.

    Requires exactly one ``plan`` output carrying one serialized value, with
    ``next`` drawn from ``{STOP, SUBGRAPH}``. Anything malformed raises
    :class:`DriverProtocolError` rather than degrading to a silent success.
    """
    plan_outputs: list[Any] = []
    for graph_outputs in raw_outputs.values():
        if not isinstance(graph_outputs, dict):
            raise DriverProtocolError(
                "malformed graph outputs: expected dict, got "
                f"{type(graph_outputs).__name__}"
            )
        if "plan" in graph_outputs:
            plan_outputs.append(graph_outputs["plan"])
    if not plan_outputs:
        raise DriverProtocolError("job result has no 'plan' output")
    if len(plan_outputs) != 1:
        raise DriverProtocolError(
            f"expected exactly one 'plan' output across the result, "
            f"got {len(plan_outputs)}"
        )
    plan_values = plan_outputs[0]
    if not isinstance(plan_values, list):
        raise DriverProtocolError(
            f"'plan' output must be a list, got {type(plan_values).__name__}"
        )
    if len(plan_values) != 1:
        raise DriverProtocolError(
            f"expected exactly one plan value, got {len(plan_values)}"
        )
    value = plan_values[0]
    if not isinstance(value, str):
        raise DriverProtocolError(
            f"plan value must be a serialized string, got {type(value).__name__}"
        )
    return validate_plan_text(value)


def validate_plan_text(value: str) -> StopPlan | SubgraphPlan:
    """Parse and validate one serialized plan string.

    ``next`` must be drawn from ``{STOP, SUBGRAPH}``; anything malformed raises
    :class:`DriverProtocolError`.
    """
    try:
        plan = json.loads(_strip_plan_wrappers(value))
    except (ValueError, TypeError) as exc:
        raise DriverProtocolError(f"plan is not valid JSON: {exc}") from exc
    if not isinstance(plan, dict):
        raise DriverProtocolError(
            f"plan must decode to an object, got {type(plan).__name__}"
        )
    try:
        raw = _RawPlan.model_validate(plan)
    except Exception as exc:
        raise DriverProtocolError(f"malformed plan: {exc}") from exc

    if raw.next == STOP:
        if raw.ops:
            raise DriverProtocolError("STOP plan must not include ops")
        return StopPlan()
    if raw.next == SUBGRAPH:
        if raw.ops is None:
            raise DriverProtocolError("subgraph plan requires an 'ops' list")
        return SubgraphPlan(ops=raw.ops, export=raw.export)
    raise DriverProtocolError(
        f"plan 'next' must be one of {sorted([STOP, SUBGRAPH])}, got {raw.next!r}"
    )


def plan_to_dict(plan: StopPlan | SubgraphPlan) -> dict[str, Any]:
    """Render a validated plan back to a plain dict."""
    if isinstance(plan, StopPlan):
        return {"next": STOP}
    return {"next": SUBGRAPH, "ops": plan.ops, "export": plan.export}


def _current_leaves(subgraph: list[dict[str, Any]]) -> list[str]:
    """Return the ids of current-subgraph nodes no other current node consumes."""
    consumed: set[str] = set()
    for op in subgraph:
        for ref in op.get("inputs", []):
            if isinstance(ref, str):
                consumed.add(ref)
    return [op["id"] for op in subgraph if op["id"] not in consumed]


def build_round(
    subgraph: list[dict[str, Any]],
    *,
    node_registry: dict[str, dict[str, Any]],
    round_index: int,
    goal: str,
    observations: list[str],
    topology: list[str],
    preview_width: int,
    model: str,
    max_tokens: int,
    temperature: float,
    threshold: float | None = None,
    library: dict[str, dict[str, Any]] | None = None,
    results: dict[str, list[dict[str, Any]]] | None = None,
    export: Sequence[str] = (),
    chat_template_kwargs: dict[str, Any] | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    dtype: str | None = None,
    extra_engine_kwargs: dict[str, Any] | None = None,
    include_proposer: bool = True,
    rows: int = 1,
) -> RoundBuild:
    """Assemble the native graph for one round.

    The round graph is the emitted subgraph (converted to native nodes via the
    YAML parser) plus one OutputOp per leaf and per exported op, the observation
    LambdaOp, and the proposer LLMChatOp. References to earlier-round ops are
    forwarded as workflow inputs from ``results`` (the decoded whole items of
    earlier ops), so no earlier op is pulled back into this round's graph. The
    proposer sees the goal, all prior observations, and the topology. With
    ``include_proposer=False`` the graph carries only the emitted ops and their
    archived outputs; the planner runs server-side. ``rows`` is the number of
    run rows (one input slice each); forwarded values are aligned to them.
    Returns the graph plus the ordered archived leaf output names.
    """
    subgraph = resolve_subgraph(subgraph, library)
    forwarded, inputs = forward_refs(
        subgraph, results if results is not None else {}, node_registry, rows
    )
    workflow_inputs: dict[str, list[str]] = {INPUT_NODE_ID: []}
    workflow_inputs.update(inputs)
    workflow = {
        "name": f"round_{round_index}",
        "inputs": workflow_inputs,
        "ops": forwarded,
        "outputs": [],
    }
    parsed = parse_yaml_payload(workflow)
    graph_name = next(iter(parsed))
    native = parsed[graph_name]["graph"]
    leaves = _current_leaves(subgraph)
    leaf_pairs = sorted(
        (_internal_id(graph_name, op["op"], op["id"]), op["id"])
        for op in subgraph
        if op["id"] in leaves
    )
    leaf_ids = [internal for internal, _ in leaf_pairs]
    leaf_user_ids = [user for _, user in leaf_pairs]
    export_pairs = sorted(
        (_internal_id(graph_name, op["op"], op["id"]), op["id"])
        for op in subgraph
        if op["id"] in export and op["id"] not in leaves
    )
    export_ids = [internal for internal, _ in export_pairs]
    export_user_ids = [user for _, user in export_pairs]
    system, user = planner_messages(
        goal=goal,
        observations=observations,
        topology=topology,
        threshold=threshold,
        library=library,
        has_subgraph=bool(subgraph),
    )
    graph = fused_round_graph(
        native,
        leaf_ids=leaf_ids,
        export_ids=export_ids,
        proposer_system=system,
        proposer_user=user,
        lambda_code=observation_lambda(preview_width=preview_width),
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        chat_template_kwargs=chat_template_kwargs,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        extra_engine_kwargs=extra_engine_kwargs,
        include_proposer=include_proposer,
    )
    return RoundBuild(
        graph=graph,
        leaf_output_names=[f"leaf_{leaf_id}" for leaf_id in leaf_ids],
        leaf_user_ids=leaf_user_ids,
        export_output_names=[f"export_{export_id}" for export_id in export_ids],
        export_user_ids=export_user_ids,
        forwarded_inputs=inputs,
    )


def _internal_id(scope: str, op_type: str, user_id: str) -> str:
    """Derive the native op id the YAML parser assigns for ``(scope, op, id)``."""
    return make_id(scope, _op_id_prefix(op_type), user_id)
