"""The two shapes a user-supplied seed takes.

A seed is either a column holding one value per row, or a key whose every row's
value derives a seed; the parser expands the key form into a generated row
function.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictStr

Pairing = Literal["unique", "shared"]
PAIRINGS: tuple[Pairing, Pairing] = ("unique", "shared")


class SeedKey(BaseModel):
    """The column a key form reads each row's key from."""

    model_config = ConfigDict(extra="forbid")

    node: StrictStr
    path: StrictStr


class ColumnSeed(BaseModel):
    """A seed read from a column at ``path``, an empty path taking its value."""

    model_config = ConfigDict(extra="ignore")

    node: StrictStr
    path: StrictStr = ""


class KeySeed(BaseModel):
    """A seed derived from each row's key, shared or one per key."""

    model_config = ConfigDict(extra="forbid")

    key: SeedKey
    pairing: Pairing = PAIRINGS[0]
