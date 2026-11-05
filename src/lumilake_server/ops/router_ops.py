from dataclasses import dataclass
from typing import Any, Literal

from lumilake_server.ops.ops import FunctionalOp, Op

RouterMatch = Literal["equals", "contains"]


@dataclass(frozen=True)
class RouterRule:
    branch: str
    match: RouterMatch
    value: str


@Op.registry.register("RouterOp")
class RouterOp(FunctionalOp):
    """Declarative per-row routing decision.

    A router reads one field from each source row and marks it with a branch.
    A downstream row gate selects the rows whose branch matches its own.
    """

    rules: list[RouterRule]
    default_branch: str
    source_node: str
    source_path: str
    input_path: str

    def __init__(
        self,
        source: Op,
        *,
        rules: list[RouterRule],
        default_branch: str,
        source_path: str = "items.output",
        input_path: str = "items.output",
    ) -> None:
        if not rules:
            raise ValueError("RouterOp rules must not be empty")
        branches = [rule.branch for rule in rules]
        if any(not branch for branch in branches):
            raise ValueError("RouterOp rule branches must be non-empty strings")
        if len(set(branches)) != len(branches):
            raise ValueError("RouterOp rule branches must be unique")
        for rule in rules:
            if rule.match not in {"equals", "contains"}:
                raise ValueError(
                    f"RouterOp match must be 'equals' or 'contains', got {rule.match!r}"
                )
        if default_branch not in branches:
            raise ValueError("RouterOp default_branch must be one of rule branches")
        if not source_path:
            raise ValueError("RouterOp source_path must be a non-empty string")
        if not input_path:
            raise ValueError("RouterOp input_path must be a non-empty string")
        super().__init__([source])
        self.rules = list(rules)
        self.default_branch = default_branch
        self.source_node = source.id
        self.source_path = source_path
        self.input_path = input_path

    @property
    def source(self) -> Op:
        return self.inputs[0]

    @property
    def branches(self) -> list[str]:
        return [rule.branch for rule in self.rules]

    def _serialize(self) -> dict[str, Any]:
        return {
            "rules": [
                {"branch": rule.branch, "match": rule.match, "value": rule.value}
                for rule in self.rules
            ],
            "default_branch": self.default_branch,
            "source_node": self.source_node,
            "source_path": self.source_path,
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
            source_path=data["source_path"],
            input_path=data["input_path"],
        )
