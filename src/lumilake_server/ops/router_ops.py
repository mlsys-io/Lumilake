"""The ``RouterOp`` YAML construct and its expansion into branch lambdas.

``RouterOp`` partitions one source op's rows into named branches. It is a
YAML construct, not a runtime op: :func:`expand_routers` rewrites it, before
any graph is built, into one ordinary list-mode ``LambdaOp`` per branch named
``<id>.<branch>``, so a consumer lists ``route.test`` in ``inputs`` like any
other node and reads exactly that branch's rows; the generated function
decodes each input value with the shared Lambda decoder and walks its ``path``.
"""

import re
from dataclasses import dataclass
from typing import Any, Self

from lumilake_server.utils.lambda_runtime import DECODER_SOURCE

ROUTER_OP = "RouterOp"

_MATCHES = ("contains", "equals")
_NAME_RE = re.compile(r"^\w+$")
_PATH_RE = re.compile(r"^\w+(\.\w+)*$")

_ROWS = """\
    rows = []
    for value in items[0] if items else []:
        decoded = _decode(value)
        if isinstance(decoded, dict):
            rows.append(decoded)
        elif isinstance(decoded, list) and all(isinstance(r, dict) for r in decoded):
            rows.extend(decoded)
        else:
            raise ValueError(LABEL + " expected a row list or object")
"""


def _indent(source: str) -> str:
    """One generated block of decoder source, indented to function level."""
    return "\n".join(
        f"    {line}" if line.strip() else "" for line in source.splitlines()
    )


@dataclass(frozen=True)
class RouterRule:
    """One ordered branch rule: rows whose value matches it take that branch."""

    branch: str
    match: str
    value: str

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.branch):
            raise ValueError(
                f"RouterRule branch must be letters, digits or underscores, got "
                f"{self.branch!r}"
            )
        if self.match not in _MATCHES:
            raise ValueError(
                f"RouterRule match must be one of {_MATCHES}, got {self.match!r}"
            )
        if not self.value:
            raise ValueError("RouterRule requires a non-empty string value")

    def matches(self) -> str:
        """The Python expression this rule's match test renders to."""
        return (
            f"{self.value!r} in text"
            if self.match == "contains"
            else f"text == {self.value!r}"
        )


@dataclass(frozen=True)
class RouterOp:
    """A validated router entry: one source, ordered rules, a default branch."""

    id: str
    inputs: list[str]
    path: list[str]
    default_branch: str
    rules: list[RouterRule]

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.id):
            raise ValueError(
                f"RouterOp id must be letters, digits or underscores, got {self.id!r}"
            )
        if len(self.inputs) != 1:
            raise ValueError(
                f"RouterOp {self.id!r} takes exactly one input, got {len(self.inputs)}"
            )
        if not self.rules:
            raise ValueError(f"RouterOp {self.id!r} rules must not be empty")
        if not _NAME_RE.match(self.default_branch):
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

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> Self:
        """Validate a raw YAML router entry."""
        router_id = entry.get("id")
        if not isinstance(router_id, str) or not router_id:
            raise ValueError("RouterOp requires a non-empty id")
        raw_inputs = entry.get("inputs")
        if not isinstance(raw_inputs, list) or not all(
            isinstance(ref, str) for ref in raw_inputs
        ):
            raise ValueError(f"RouterOp {router_id!r} inputs must be a list of ids")
        raw_path = entry.get("path")
        if raw_path is None:
            path: list[str] = []
        elif isinstance(raw_path, str) and _PATH_RE.match(raw_path):
            path = raw_path.split(".")
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
            rules.append(RouterRule(branch, match, value))
        default_branch = entry.get("default_branch")
        if not isinstance(default_branch, str):
            raise ValueError(f"RouterOp {router_id!r} requires a default_branch")
        return cls(router_id, raw_inputs, path, default_branch, rules)


def branch_lambda_code(router: RouterOp, branch: str) -> str:
    """The list-mode LambdaOp source selecting ``branch``'s rows from ``router``:
    a rule branch keeps the rows its own rule matches and no earlier rule
    matches, and the default branch keeps the rows no rule matches at all."""
    rules = router.rules
    names = [rule.branch for rule in rules]
    own = names.index(branch) if branch in names else None
    tests = [rule.matches() for rule in rules]
    if own is None:
        # The default branch keeps the rows no rule matched.
        skip, keep = tests, None
    else:
        skip, keep = tests[:own], [tests[own]]
    lines = [
        f"def route_{branch}(items):",
        f'    LABEL = "{router.id}.{branch}"',
        _indent(DECODER_SOURCE),
        _ROWS,
        "    out = []",
        "    for row in rows:",
    ]
    if router.path:
        lines.append("        value = row")
        lines.append(f"        for part in {router.path!r}:")
        lines.append("            if not isinstance(value, dict) or part not in value:")
        lines.append('                raise ValueError(f"{LABEL} row has no {part!r}")')
        lines.append("            value = value[part]")
        lines.append(
            "        text = value if isinstance(value, str) else "
            "json.dumps(value, ensure_ascii=False)"
        )
    else:
        lines.append("        text = json.dumps(row, ensure_ascii=False)")
    if skip:
        lines.append(f"        if {' or '.join(skip)}:")
        lines.append("            continue")
    if keep is None:
        lines.append("        out.append(row)")
    else:
        lines.append(f"        if {' or '.join(keep)}:")
        lines.append("            out.append(row)")
    lines.append("    return out")
    return "\n".join(lines) + "\n"


def expand_routers(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand every ``RouterOp`` entry into one list-mode LambdaOp per branch,
    each reading the router's single input, so a consumer that names
    ``<router id>.<branch>`` in its ``inputs`` gets exactly that branch's rows."""
    expanded: list[dict[str, Any]] = []
    for entry in ops:
        if not isinstance(entry, dict) or entry.get("op") != ROUTER_OP:
            expanded.append(entry)
            continue
        router = RouterOp.from_entry(entry)
        expanded.extend(
            {
                "id": f"{router.id}.{branch}",
                "op": "LambdaOp",
                "inputs": router.inputs.copy(),
                "mode": "list",
                "fn_name": f"route_{branch}",
                "code": branch_lambda_code(router, branch),
            }
            for branch in router.branches
        )
    return expanded
