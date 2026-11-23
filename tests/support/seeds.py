"""The documented keyed-seed formula, for tests that check a generated draw."""

import hashlib
import json
from typing import Any


def documented_seed(
    keys: list[Any],
    seed: int = 7,
    op_id: str = "Answer",
    round_number: int | None = None,
) -> list[int]:
    """The docs/OPS.md seed formula."""

    def one(key: Any) -> int:
        text = json.dumps([seed, op_id, round_number, key], separators=(",", ":"))
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % 2**31

    return [one(key) for key in keys]
