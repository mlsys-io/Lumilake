"""The ``RouterOp`` YAML construct and its expansion into branch lambdas.

``RouterOp`` partitions one source op's rows into named branches. It is a
YAML construct, not a runtime op: :func:`expand_routers` rewrites it, before
any graph is built, into one ordinary list-mode ``LambdaOp`` per branch named
``<id>.<branch>``, so a consumer lists ``route.test`` in ``inputs`` like any
other node and reads exactly that branch's rows; the generated function reads
its already-decoded values and walks its ``path``.
"""

import re
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)

from lumilake_server.ops.spec_validation import validate_fields
from lumilake_server.runtime.lambda_task_helpers import FAIL_KEY

ROUTER_OP = "RouterOp"
ERROR_BRANCH = "error"

_NAME_RE = re.compile(r"^\w+$")
_PATH_RE = re.compile(r"^\w+(\.\w+)*$")

_MARKER = f"""\
    def marked(row):
        return isinstance(row, dict) and list(row) == ["{FAIL_KEY}"]
"""

_ROWS = """\
    rows = []
    for value in items[0] if items else []:
        if isinstance(value, dict):
            rows.append(value)
        elif isinstance(value, list) and all(isinstance(r, dict) for r in value):
            rows.extend(value)
        else:
            raise ValueError(LABEL + " expected a row list or object")
"""


class RouterRuleSpec(BaseModel):
    """One ordered branch rule: rows whose value matches it take that branch."""

    model_config = ConfigDict(extra="ignore")

    branch: StrictStr
    match: Literal["contains", "equals", "empty"]
    value: StrictStr | None = None

    @field_validator("branch")
    @classmethod
    def _branch_name(cls, value: str) -> str:
        if not _NAME_RE.match(value):
            raise ValueError(f"must be letters, digits or underscores, got {value!r}")
        return value

    @model_validator(mode="after")
    def _match_value(self) -> Self:
        if self.match == "empty":
            if self.value is not None:
                raise ValueError(
                    f"branch {self.branch!r} matches 'empty' and takes no value, got"
                    f" {self.value!r}"
                )
            return self
        if not self.value:
            raise ValueError("requires a non-empty string value")
        return self

    def matches(self) -> str:
        """The Python expression this rule's match test renders to."""
        if self.match == "empty":
            return "value is None or value == '' or value == [] or value == {}"
        return (
            f"{self.value!r} in text"
            if self.match == "contains"
            else f"text == {self.value!r}"
        )


class RouterSpec(BaseModel):
    """A router entry: one source, ordered rules, a default branch and an error flag."""

    model_config = ConfigDict(extra="ignore")

    id: StrictStr
    inputs: list[StrictStr]
    path: list[StrictStr] = Field(default_factory=list)
    default_branch: StrictStr
    rules: list[RouterRuleSpec]
    error_branch: StrictBool = False

    @field_validator("default_branch")
    @classmethod
    def _name(cls, value: str) -> str:
        if not _NAME_RE.match(value):
            raise ValueError(f"must be letters, digits or underscores, got {value!r}")
        return value

    @field_validator("path", mode="before")
    @classmethod
    def _dotted_path(cls, value: Any) -> Any:
        """Take the tested field as parts, from one dotted name."""
        if value is None:
            return []
        if not isinstance(value, str) or not _PATH_RE.match(value):
            raise ValueError(f"must be a dotted field name, got {value!r}")
        return value.split(".")

    @model_validator(mode="after")
    def _branches(self) -> Self:
        if len(self.inputs) != 1:
            raise ValueError(f"takes exactly one input, got {len(self.inputs)}")
        if not self.rules:
            raise ValueError("rules must not be empty")
        names = [rule.branch for rule in self.rules]
        if len(set(names)) != len(names):
            raise ValueError("rule branches must be unique")
        if self.default_branch in names:
            raise ValueError(
                f"rules already name the default branch {self.default_branch!r}; a"
                " default catches what no rule matched"
            )
        if self.error_branch and (
            ERROR_BRANCH in names or self.default_branch == ERROR_BRANCH
        ):
            raise ValueError(
                f"must not name {ERROR_BRANCH!r} as a rule branch or the default"
                " branch; its declared error branch receives failed records"
            )
        return self

    @property
    def branches(self) -> tuple[str, ...]:
        """The branches in routing order, with ``error`` last when declared."""
        ordered = (self.default_branch, *(rule.branch for rule in self.rules))
        return (*ordered, ERROR_BRANCH) if self.error_branch else ordered


def branch_lambda_code(router: RouterSpec, branch: str) -> str:
    """The list-mode LambdaOp source that selects one ``branch`` of a router."""
    rules = router.rules
    names = [rule.branch for rule in rules]
    header = [f"def route_{branch}(items):", f'    LABEL = "{router.id}.{branch}"']
    if branch == ERROR_BRANCH and router.error_branch:
        return (
            "\n".join(
                [
                    *header,
                    _ROWS,
                    _MARKER,
                    "    return [row for row in rows if marked(row)]",
                ]
            )
            + "\n"
        )
    own = names.index(branch) if branch in names else None
    tests = [rule.matches() for rule in rules]
    if own is None:
        # The default branch keeps the rows no rule matched.
        skip, keep = tests, None
    else:
        skip, keep = tests[:own], [tests[own]]
    lines = [
        *header,
        _ROWS,
        _MARKER,
        "    out = []",
        "    for row in rows:",
        "        if marked(row):",
        "            continue",
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
        lines.append("        value = row")
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


def expand_routers(
    ops: list[dict[str, Any]], copied: frozenset[str] = frozenset()
) -> tuple[list[dict[str, Any]], frozenset[str]]:
    """Expand each router into per-branch lambdas; return the ops and cleared ids.

    ``copied`` holds the ids of routers an unrolled repeat copied, whose
    unrolled ids may be dotted.
    """
    expanded: list[dict[str, Any]] = []
    cleared: set[str] = set()
    for entry in ops:
        if not isinstance(entry, dict) or entry.get("op") != ROUTER_OP:
            expanded.append(entry)
            continue
        router = validate_fields(
            RouterSpec, entry, f"RouterOp '{entry.get('id', '?')}'"
        )
        if router.id not in copied and not _NAME_RE.match(router.id):
            raise ValueError(
                f"RouterOp '{router.id}' id: must be letters, digits or underscores,"
                f" got {router.id!r}"
            )
        for branch in router.branches:
            expanded.append(
                {
                    "id": f"{router.id}.{branch}",
                    "op": "LambdaOp",
                    "inputs": router.inputs.copy(),
                    "mode": "list",
                    "fn_name": f"route_{branch}",
                    "code": branch_lambda_code(router, branch),
                }
            )
            if branch != ERROR_BRANCH:
                cleared.add(f"{router.id}.{branch}")
    return expanded, frozenset(cleared)
