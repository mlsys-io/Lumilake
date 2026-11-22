"""Each op class declares whether its output is a random draw."""

import pytest

from lumilake_server.common import GenerationConfig, Message
from lumilake_server.ops import (
    EmbeddingOp,
    ImageGenerationOp,
    LambdaOp,
    LLMChatOp,
    LLMCompletionOp,
    Op,
    OpMessage,
)

_MODEL = "model-a"


def _chat(temperature: float | None = None, num_beams: int | None = None) -> LLMChatOp:
    return LLMChatOp(
        [OpMessage(role="user", content="hi")],
        config=GenerationConfig(
            model=_MODEL, temperature=temperature, num_beams=num_beams
        ),
    )


def _echo(rows: tuple[str | list[Message], ...]) -> str:
    return str(rows[0])


@pytest.mark.parametrize(
    ("op", "samples"),
    [
        pytest.param(_chat(), True, id="absent-temperature"),
        pytest.param(_chat(temperature=0.0), False, id="greedy"),
        pytest.param(_chat(temperature=0.7), True, id="sampled"),
        pytest.param(_chat(temperature=0.7, num_beams=4), False, id="beam"),
        pytest.param(
            LLMCompletionOp("hi", GenerationConfig(model=_MODEL)),
            True,
            id="completion",
        ),
        pytest.param(
            ImageGenerationOp("a cat", GenerationConfig(model=_MODEL)),
            True,
            id="image",
        ),
        pytest.param(
            EmbeddingOp("text", GenerationConfig(model=_MODEL)), False, id="embedding"
        ),
        pytest.param(LambdaOp([["a"]], _echo), False, id="lambda"),
    ],
)
def test_samples_is_a_property_of_the_op(op: Op, samples: bool) -> None:
    assert op.samples is samples
