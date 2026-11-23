"""The record-store flags a workflow sets at its top level."""

from typing import Self

from pydantic import BaseModel, ConfigDict, StrictBool, model_validator
from pydantic_core import PydanticCustomError


class RecordFlags(BaseModel):
    """Whether a workflow stores its records and whether it reuses stored ones."""

    model_config = ConfigDict(extra="ignore")

    store: StrictBool = False
    """Whether the workflow writes its records to the per-record store."""
    reuse: StrictBool = False
    """Whether the workflow keeps the records an earlier storing run computed."""

    @model_validator(mode="after")
    def _reuse_needs_store(self) -> Self:
        if self.reuse and not self.store:
            raise PydanticCustomError(
                "reuse_needs_store", "'reuse' requires 'store: true'"
            )
        return self
