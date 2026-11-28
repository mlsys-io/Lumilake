"""The two shapes a user-supplied seed takes: a column, or a per-row key."""

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
