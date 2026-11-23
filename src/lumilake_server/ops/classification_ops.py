from dataclasses import dataclass, field
from typing import Any, Literal, Self

from lumilake_server.ops.data_ops import DataOp
from lumilake_server.ops.ops import Op


@dataclass(frozen=True)
class ClassificationConfig:
    """Model and backend settings for a FlowMesh classification task."""

    model: str
    backend: Literal["vllm", "transformers"] = "vllm"
    vllm: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("ClassificationOp config.model must be a non-empty string")
        if self.backend not in {"vllm", "transformers"}:
            raise ValueError(
                "ClassificationOp config.backend must be vllm or transformers"
            )
        if not isinstance(self.vllm, dict):
            raise ValueError("ClassificationOp config.vllm must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "backend": self.backend,
            "vllm": self.vllm.copy(),
        }


@dataclass(frozen=True)
class ClassificationPair:
    """One cross-encoder query/candidate pair."""

    query: str
    candidate: str

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("ClassificationOp query must be a non-empty string")
        if not self.candidate.strip():
            raise ValueError("ClassificationOp candidate must be a non-empty string")


def _as_values(value: str | list[str], name: str) -> list[str]:
    values = [value] if isinstance(value, str) else value
    if not values or not all(isinstance(item, str) and item.strip() for item in values):
        raise ValueError(f"ClassificationOp requires non-empty {name} input")
    return values


def literal_pairs(
    query: list[str], candidates: list[str], *, op_id: str | None = None
) -> list[ClassificationPair]:
    """Validate and broadcast literal query and candidate lists. ``op_id``
    names the op in the error once it has one; construction has no id yet."""
    if len(query) == 1:
        queries = query * len(candidates)
    elif len(query) == len(candidates):
        queries = query
    else:
        where = "ClassificationOp" if op_id is None else f"ClassificationOp '{op_id}'"
        raise ValueError(f"{where} query and candidates must have equal lengths")
    return [
        ClassificationPair(query=query_value, candidate=candidate)
        for query_value, candidate in zip(queries, candidates)
    ]


@Op.registry.register("ClassificationOp")
class ClassificationOp(Op):
    """Classify query/candidate pairs with a FlowMesh classification task.

    The classification task returns the classifier's output for each pair, the
    cross-encoder label probability unless ``config.vllm.apply_sigmoid`` is
    disabled."""

    def __init__(
        self,
        query: str | list[str] | Op,
        candidates: str | list[str] | Op,
        config: ClassificationConfig,
    ) -> None:
        self.config = config
        query_op = (
            query if isinstance(query, Op) else DataOp(data=_as_values(query, "query"))
        )
        candidates_op = (
            candidates
            if isinstance(candidates, Op)
            else DataOp(data=_as_values(candidates, "candidate"))
        )
        if isinstance(query_op, DataOp) and isinstance(candidates_op, DataOp):
            literal_pairs(query_op.data, candidates_op.data)
        super().__init__([query_op, candidates_op])

    @property
    def query(self) -> Op:
        return self.inputs[0]

    @property
    def candidates(self) -> Op:
        return self.inputs[1]

    result_path = "items"

    @property
    def row_source(self) -> Op:
        return self.candidates

    def _serialize(self) -> dict[str, Any]:
        return {
            "query": self.query.id,
            "candidates": self.candidates.id,
            "config": self.config.to_dict(),
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, Op]) -> Self:
        return cls(
            query=other_ops[data["query"]],
            candidates=other_ops[data["candidates"]],
            config=ClassificationConfig(**data["config"]),
        )


def classify(
    query: str | list[str] | Op,
    candidates: str | list[str] | Op,
    config: ClassificationConfig,
) -> ClassificationOp:
    return ClassificationOp(query, candidates, config)
