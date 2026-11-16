"""YAML emitter for the Router op."""

from typing import Any

from lumilake_server.ops.router_ops import RouterMatch, RouterRule

from .common import EmitContext, require_str


def emit_router_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    entry = ctx.entry
    if len(op_dict["_inputs"]) != 1:
        raise ValueError(
            f"RouterOp '{entry.id}' requires exactly one input, the source node"
        )
    rules = entry.fields.get("rules")
    if not isinstance(rules, list) or not all(isinstance(rule, dict) for rule in rules):
        raise ValueError(f"RouterOp '{entry.id}' rules must be a list of mappings")
    op_dict.update(
        rules=[
            RouterRule(
                branch=require_str(
                    rule, "branch", f"RouterOp '{entry.id}' rules[{index}]"
                ),
                match=_require_match(rule, f"RouterOp '{entry.id}' rules[{index}]"),
                value=require_str(
                    rule, "value", f"RouterOp '{entry.id}' rules[{index}]"
                ),
            ).to_dict()
            for index, rule in enumerate(rules)
        ],
        default_branch=entry.fields.get("default_branch"),
        input_path=entry.fields.get("input_path", "items.output"),
    )


def _require_match(rule: dict[str, Any], where: str) -> RouterMatch:
    """A rule's match mode, typed for the rule that validates it."""
    match = require_str(rule, "match", where)
    if match == "equals":
        return "equals"
    if match == "contains":
        return "contains"
    raise ValueError(
        f"{where}: requires the 'match' field to be 'equals' or 'contains'"
    )
