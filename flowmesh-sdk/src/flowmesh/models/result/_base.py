# Necessary for the recursive ``children`` forward reference.
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    SerializerFunctionWrapHandler,
    model_serializer,
)

from ..artifacts import ArtifactContext

if TYPE_CHECKING:
    from .catalog import AnyExecutorResult


class DropNoneModel(BaseModel):
    @model_serializer(mode="wrap")
    def _drop_none_fields(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        dumped = handler(self)
        if not isinstance(dumped, dict):
            return dumped
        optional = {
            field.alias or name
            for name, field in type(self).model_fields.items()
            if not field.is_required()
        }
        return {
            key: value
            for key, value in dumped.items()
            if value is not None or key not in optional
        }


class StrictModel(DropNoneModel):
    model_config = ConfigDict(extra="forbid")


class BaseExecutorResult(DropNoneModel):
    model_config = ConfigDict(extra="allow", serialize_by_alias=True)

    ok: bool = True
    children: dict[str, SerializeAsAny[AnyExecutorResult]] = Field(
        default_factory=dict, exclude_if=lambda v: not v
    )
    artifacts_: ArtifactContext | None = Field(default=None, alias="_artifacts")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        if "artifacts_" in cls.__annotations__:
            raise TypeError(
                f"{cls.__name__} may not redefine the internal "
                "BaseExecutorResult.artifacts_ field"
            )


class StrictExecutorResult(BaseExecutorResult):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)
