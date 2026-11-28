"""The documented keyed-seed formula, for tests that check a generated draw."""

import hashlib
import json
from typing import Any


def documented_seed(
    keys: list[Any],
    seed: int = 7,
    op_id: str = "Answer",
    round_number: int | None = None,
    samples: int = 1,
) -> list[int]:
    """The docs/OPS.md seed formula, repeated per sample when ``samples`` > 1."""

    def one(key: Any) -> int:
        text = json.dumps([seed, op_id, round_number, key], separators=(",", ":"))
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % 2**31

    if samples == 1:
        return [one(key) for key in keys]
    return [one([key, sample]) for key in keys for sample in range(samples)]
