from dataclasses import dataclass
from typing import Any, Literal

from lumilake_server.ops.ops import FunctionalOp, Op

RouterMatch = Literal["equals", "contains"]


@dataclass(frozen=True)
class RouterRule:
    branch: str
    match: RouterMatch
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.branch, str) or not self.branch:
            raise ValueError("RouterRule requires a non-empty branch")
        if self.match not in {"equals", "contains"}:
            raise ValueError(
                f"RouterRule match must be 'equals' or 'contains', got {self.match!r}"
            )
        if not isinstance(self.value, str):
            raise ValueError("RouterRule requires a string value")

    def to_dict(self) -> dict[str, str]:
        return {"branch": self.branch, "match": self.match, "value": self.value}


@dataclass(frozen=True)
class RowGate:
    """Runs an ``LLMChatOp`` only on the rows a router marked with ``branch``."""

    node: str
    branch: str
    terminal: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.node, str) or not self.node:
            raise ValueError("RowGate requires a node")
        if not isinstance(self.branch, str) or not self.branch:
            raise ValueError("RowGate requires a non-empty branch")
        if not isinstance(self.terminal, bool):
            raise ValueError("RowGate terminal must be a bool")

    def to_dict(self) -> dict[str, Any]:
        return {"node": self.node, "branch": self.branch, "terminal": self.terminal}


@dataclass(frozen=True)
class MessagesColumn:
    """Locates one serialized message array per row, read from ``node``'s ``path``."""

    node: str
    path: str
    label: str

    def __post_init__(self) -> None:
        if not isinstance(self.node, str) or not self.node:
            raise ValueError("MessagesColumn requires a node")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("MessagesColumn requires a path")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("MessagesColumn requires a label")

    def to_dict(self) -> dict[str, str]:
        return {"node": self.node, "path": self.path, "label": self.label}


@Op.registry.register("RouterOp")
class RouterOp(FunctionalOp):
    """Declarative per-row routing decision.

    A router reads one field from each source row and marks it with a branch.
    A downstream row gate selects the rows whose branch matches its own.
    """

    rules: list[RouterRule]
    default_branch: str
    input_path: str

    def __init__(
        self,
        source: Op,
        *,
        rules: list[RouterRule],
        default_branch: str,
        input_path: str = "items.output",
    ) -> None:
        if not rules:
            raise ValueError("RouterOp rules must not be empty")
        branches = [rule.branch for rule in rules]
        if len(set(branches)) != len(branches):
            raise ValueError("RouterOp rule branches must be unique")
        if default_branch not in branches:
            raise ValueError("RouterOp default_branch must be one of rule branches")
        if not input_path:
            raise ValueError("RouterOp input_path must be a non-empty string")
        super().__init__([source])
        self.rules = list(rules)
        self.default_branch = default_branch
        self.input_path = input_path

    @property
    def source(self) -> Op:
        return self.inputs[0]

    @property
    def branches(self) -> list[str]:
        return [rule.branch for rule in self.rules]

    def _serialize(self) -> dict[str, Any]:
        return {
            "rules": [rule.to_dict() for rule in self.rules],
            "default_branch": self.default_branch,
            "input_path": self.input_path,
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "RouterOp":
        source_ids = data["_inputs"]
        if len(source_ids) != 1:
            raise ValueError("RouterOp must have exactly one input")
        return cls(
            other_ops[source_ids[0]],
            rules=[RouterRule(**rule) for rule in data["rules"]],
            default_branch=data["default_branch"],
            input_path=data["input_path"],
        )
