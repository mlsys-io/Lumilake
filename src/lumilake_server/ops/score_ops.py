from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from lumilake_server.ops.data_ops import DataOp
from lumilake_server.ops.ops import Op


@dataclass(frozen=True)
class ScoreConfig:
    """Model and backend settings for a FlowMesh classification task."""

    model: str
    backend: Literal["vllm", "transformers"] = "vllm"
    vllm: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("ScoreOp config.model must be a non-empty string")
        if self.backend not in {"vllm", "transformers"}:
            raise ValueError("ScoreOp config.backend must be vllm or transformers")
        if not isinstance(self.vllm, dict):
            raise ValueError("ScoreOp config.vllm must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "backend": self.backend,
            "vllm": dict(self.vllm),
        }


@dataclass(frozen=True)
class ScorePair:
    """One cross-encoder query/candidate pair."""

    query: str
    candidate: str

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("ScoreOp query must be a non-empty string")
        if not self.candidate.strip():
            raise ValueError("ScoreOp candidate must be a non-empty string")


def _as_values(value: str | list[str], name: str) -> list[str]:
    values = [value] if isinstance(value, str) else list(value)
    if not values or not all(isinstance(item, str) and item.strip() for item in values):
        raise ValueError(f"ScoreOp requires non-empty {name} input")
    return values


def _literal_pairs(
    query: list[str], candidates: list[str], *, op_id: str | None = None
) -> list[ScorePair]:
    """Validate and broadcast literal query and candidate lists. ``op_id``
    names the op in the error once it has one; construction has no id yet."""
    if len(query) == 1:
        queries = query * len(candidates)
    elif len(query) == len(candidates):
        queries = query
    else:
        where = "ScoreOp" if op_id is None else f"ScoreOp '{op_id}'"
        raise ValueError(f"{where} query and candidates must have equal lengths")
    return [
        ScorePair(query=query_value, candidate=candidate)
        for query_value, candidate in zip(queries, candidates)
    ]


@Op.registry.register("ScoreOp")
class ScoreOp(Op):
    """Cross-encoder scoring of query/candidate pairs on FlowMesh."""

    def __init__(
        self,
        query: str | list[str] | Op,
        candidates: str | list[str] | Op,
        config: ScoreConfig,
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
            _literal_pairs(query_op.data, candidates_op.data)
        super().__init__([query_op, candidates_op])

    @property
    def query(self) -> Op:
        return self.inputs[0]

    @property
    def candidates(self) -> Op:
        return self.inputs[1]

    def consumer_result_path(self) -> str:
        return "items"

    def consumer_row_count(self, input_counts: Sequence[int | None]) -> int | None:
        return input_counts[1] if len(input_counts) > 1 else None

    def _serialize(self) -> dict[str, Any]:
        return {
            "query": self.query.id,
            "candidates": self.candidates.id,
            "config": self.config.to_dict(),
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, Op]) -> "ScoreOp":
        return cls(
            query=other_ops[data["query"]],
            candidates=other_ops[data["candidates"]],
            config=ScoreConfig(**data["config"]),
        )


def score(
    query: str | list[str] | Op,
    candidates: str | list[str] | Op,
    config: ScoreConfig,
) -> ScoreOp:
    return ScoreOp(query, candidates, config)
