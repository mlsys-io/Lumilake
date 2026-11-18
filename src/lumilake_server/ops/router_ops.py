"""The ``RouterOp`` YAML construct and its expansion into branch lambdas.

``RouterOp`` partitions one source op's rows into named branches. It is a
paper construct, not a runtime op: :func:`expand_routers` rewrites it, before
any graph is built, into one ordinary list-mode ``LambdaOp`` per branch named
``<id>.<branch>``, so a consumer lists ``route.test`` in ``inputs`` like any
other node and reads exactly that branch's rows. The generated function
carries its own row decoder and ``path`` walk.
"""

import re
from dataclasses import dataclass
from typing import Any

ROUTER_OP = "RouterOp"

_MATCHES = ("contains", "equals")
_BRANCH_RE = re.compile(r"^[A-Za-z0-9_]+$")
_DOTTED_RE = re.compile(r"^\w+(\.\w+)*$")

_DECODER = """\
    def decode_column(column):
        rows = []
        for raw in column:
            candidate = raw
            while True:
                if isinstance(candidate, list) and len(candidate) == 1:
                    candidate = candidate[0]
                    continue
                if isinstance(candidate, str):
                    candidate = json.loads(candidate)
                    continue
                if isinstance(candidate, dict) and isinstance(
                    candidate.get("rows"), list
                ):
                    candidate = candidate["rows"]
                    continue
                break
            if isinstance(candidate, dict):
                rows.append(candidate)
            elif isinstance(candidate, list) and all(
                isinstance(row, dict) for row in candidate
            ):
                rows.extend(candidate)
            else:
                raise ValueError(
                    f"{LABEL} expected a row list or object"
                )
        return rows
"""


@dataclass(frozen=True)
class RouterRule:
    """One ordered branch rule: rows whose value matches it take that branch."""

    branch: str
    match: str
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.branch, str) or not _BRANCH_RE.match(self.branch):
            raise ValueError(
                "RouterRule branch must be letters, digits or underscores, got "
                f"{self.branch!r}"
            )
        if self.match not in _MATCHES:
            raise ValueError(
                f"RouterRule match must be one of {list(_MATCHES)}, got "
                f"{self.match!r}"
            )
        if not isinstance(self.value, str) or not self.value:
            raise ValueError("RouterRule requires a non-empty string value")


@dataclass(frozen=True)
class RouterOp:
    """A validated router entry: one source, ordered rules, a default branch."""

    id: str
    inputs: tuple[str, ...]
    path: tuple[str, ...]
    default_branch: str
    rules: tuple[RouterRule, ...]

    def __post_init__(self) -> None:
        if len(self.inputs) != 1:
            raise ValueError(
                f"RouterOp {self.id!r} takes exactly one input, got {len(self.inputs)}"
            )
        if not self.rules:
            raise ValueError(f"RouterOp {self.id!r} rules must not be empty")
        if not _BRANCH_RE.match(self.default_branch):
            raise ValueError(
                f"RouterOp {self.id!r} default_branch must be letters, digits or "
                f"underscores, got {self.default_branch!r}"
            )
        branches = [rule.branch for rule in self.rules]
        if len(set(branches)) != len(branches):
            raise ValueError(f"RouterOp {self.id!r} rule branches must be unique")
        if self.default_branch in branches:
            raise ValueError(
                f"RouterOp {self.id!r} rules already name the default branch "
                f"{self.default_branch!r}; a default catches what no rule matched"
            )

    @property
    def branches(self) -> tuple[str, ...]:
        """The default branch first, then every rule branch in rule order."""
        return (self.default_branch, *(rule.branch for rule in self.rules))

    def match_test(self, rule: RouterRule) -> str:
        if rule.match == "contains":
            return f"{rule.value!r} in text"
        return f"text == {rule.value!r}"

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> "RouterOp":
        """Validate a raw YAML router entry."""
        router_id = entry.get("id")
        raw_inputs = entry.get("inputs")
        if not isinstance(raw_inputs, list) or not all(
            isinstance(ref, str) for ref in raw_inputs
        ):
            raise ValueError(f"RouterOp {router_id!r} inputs must be a list of ids")
        raw_path = entry.get("path")
        if raw_path is None:
            path: tuple[str, ...] = ()
        elif isinstance(raw_path, str) and _DOTTED_RE.match(raw_path):
            path = tuple(raw_path.split("."))
        else:
            raise ValueError(
                f"RouterOp {router_id!r} path must be a dotted field name, got "
                f"{raw_path!r}"
            )
        raw_rules = entry.get("rules")
        if not isinstance(raw_rules, list):
            raise ValueError(f"RouterOp {router_id!r} rules must be a list")
        rules: list[RouterRule] = []
        for raw_rule in raw_rules:
            if not isinstance(raw_rule, dict):
                raise ValueError(f"RouterOp {router_id!r} rules must be mappings")
            branch = raw_rule.get("branch")
            match = raw_rule.get("match")
            value = raw_rule.get("value")
            if not (
                isinstance(branch, str)
                and isinstance(match, str)
                and isinstance(value, str)
            ):
                raise ValueError(
                    f"RouterOp {router_id!r} rules require string branch, match "
                    f"and value, got {raw_rule!r}"
                )
            rules.append(RouterRule(branch=branch, match=match, value=value))
        default_branch = entry.get("default_branch")
        if not isinstance(default_branch, str):
            raise ValueError(f"RouterOp {router_id!r} requires a default_branch")
        if not isinstance(router_id, str) or not router_id:
            raise ValueError("RouterOp requires a non-empty id")
        return cls(
            id=router_id,
            inputs=tuple(raw_inputs),
            path=path,
            default_branch=default_branch,
            rules=tuple(rules),
        )


def branch_lambda_code(router: RouterOp, branch: str) -> str:
    """The list-mode LambdaOp source selecting ``branch``'s rows from ``router``.

    A rule branch keeps the rows its own rule matches and no earlier rule does;
    the default branch keeps the rows no rule matches at all.
    """
    fn_name = f"route_{branch}"
    rules = router.rules
    guards: list[str] = []
    if branch == router.default_branch:
        guards = [router.match_test(rule) for rule in rules]
    else:
        index = [rule.branch for rule in rules].index(branch)
        guards = [router.match_test(rule) for rule in rules[:index]]

    lines = [
        f"def {fn_name}(items):",
        f'    LABEL = "{router.id}.{branch}"',
        _DECODER,
        "    rows = decode_column(items[0]) if items else []",
        "    out = []",
        "    for row in rows:",
    ]
    if router.path:
        lines.append("        value = row")
        lines.append(f"        for part in {router.path!r}:")
        lines.append("            if not isinstance(value, dict):")
        lines.append(
            '                raise ValueError(f"{LABEL} row is not an object at "\n'
            '                                 f"{part!r}")'
        )
        lines.append("            if part not in value:")
        lines.append('                raise ValueError(f"{LABEL} row has no {part!r}")')
        lines.append("            value = value[part]")
        lines.append(
            "        text = value if isinstance(value, str) else json.dumps(value)"
        )
    else:
        lines.append("        text = row if isinstance(row, str) else json.dumps(row)")
    if guards:
        lines.append(f"        if {' or '.join(guards)}:")
        lines.append("            continue")
    if branch == router.default_branch:
        lines.append("        out.append(row)")
    else:
        rule = rules[[item.branch for item in rules].index(branch)]
        lines.append(f"        if {router.match_test(rule)}:")
        lines.append("            out.append(row)")
    lines.append("    return out")
    return "\n".join(lines) + "\n"


def expand_routers(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace every ``RouterOp`` entry with one list-mode LambdaOp per branch.

    Each branch reads the router's single input, so a consumer that names
    ``<router id>.<branch>`` in its ``inputs`` gets exactly that branch's rows.
    """
    expanded: list[dict[str, Any]] = []
    for entry in ops:
        if not isinstance(entry, dict) or entry.get("op") != ROUTER_OP:
            expanded.append(entry)
            continue
        router = RouterOp.from_entry(entry)
        for branch in router.branches:
            expanded.append(
                {
                    "id": f"{router.id}.{branch}",
                    "op": "LambdaOp",
                    "inputs": list(router.inputs),
                    "mode": "list",
                    "fn_name": f"route_{branch}",
                    "code": branch_lambda_code(router, branch),
                }
            )
    return expanded
