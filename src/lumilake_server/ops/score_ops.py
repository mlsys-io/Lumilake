import math
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
        if not self.model.strip():
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


@dataclass(frozen=True)
class ScoreItem:
    """One FlowMesh classification result item."""

    index: int
    score: float


@dataclass(frozen=True)
class ScoreResult:
    """Scores returned in their input pair order."""

    model: str | None
    items: tuple[ScoreItem, ...]
    usage: dict[str, Any]
    artifacts: dict[str, Any]


def _as_values(value: str | list[str], name: str) -> list[str]:
    values = [value] if isinstance(value, str) else list(value)
    if not values or not all(isinstance(item, str) and item.strip() for item in values):
        raise ValueError(f"ScoreOp requires non-empty {name} input")
    return values


def _literal_pairs(
    query: list[str], candidates: list[str], *, op_id: str = "ScoreOp"
) -> list[ScorePair]:
    if len(query) == 1:
        queries = query * len(candidates)
    elif len(query) == len(candidates):
        queries = query
    else:
        raise ValueError(
            f"ScoreOp '{op_id}' query and candidates must have equal lengths"
        )
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
        config: ScoreConfig | None = None,
    ) -> None:
        if config is None:
            raise ValueError("ScoreOp requires config with a non-empty model")
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

    @staticmethod
    def parse_response(result: dict[str, Any]) -> ScoreResult:
        """Parse FlowMesh classification items in contiguous index order."""
        if not isinstance(result, dict):
            raise ValueError(
                f"ScoreOp response must be a mapping (got {type(result).__name__})"
            )
        if result.get("ok") is False or result.get("error"):
            raise ValueError(f"ScoreOp response reports failure: {result.get('error')}")
        raw_items = result.get("items")
        if not isinstance(raw_items, list):
            raise ValueError("ScoreOp response is missing the 'items' list")
        usage = result.get("usage")
        if not isinstance(usage, dict):
            raise ValueError("ScoreOp response is missing the 'usage' block")

        items: list[ScoreItem] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                raise ValueError("ScoreOp response item must be a mapping")
            index = raw.get("index")
            score = raw.get("score")
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("ScoreOp response item index must be an integer")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("ScoreOp response item score must be numeric")
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise ValueError("ScoreOp response item score must be finite")
            items.append(ScoreItem(index=index, score=numeric_score))

        items.sort(key=lambda item: item.index)
        expected_indices = list(range(len(items)))
        if [item.index for item in items] != expected_indices:
            raise ValueError(
                "ScoreOp response item indices must be contiguous from zero"
            )

        artifacts = result.get("_artifacts")
        model = result.get("model")
        return ScoreResult(
            model=str(model) if model is not None else None,
            items=tuple(items),
            usage=usage,
            artifacts=artifacts if isinstance(artifacts, dict) else {},
        )


def score(
    query: str | list[str] | Op,
    candidates: str | list[str] | Op,
    config: ScoreConfig | None = None,
) -> ScoreOp:
    return ScoreOp(query, candidates, config)
