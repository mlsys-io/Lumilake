"""Validating one user-supplied op entry with its Pydantic model."""

from typing import Any

from pydantic import BaseModel, ValidationError


def validate_fields[SpecT: BaseModel](
    model: type[SpecT],
    fields: dict[str, Any],
    where: str,
    names: dict[str, str] | None = None,
) -> SpecT:
    """Validate one op entry, naming the op and field in the parser's error."""
    try:
        return model.model_validate(fields)
    except ValidationError as exc:
        parts: list[str] = []
        for error in exc.errors():
            message = error["msg"].removeprefix("Value error, ")
            field = ".".join(str(loc) for loc in error["loc"])
            field = names[field] if names is not None and field in names else field
            parts.append(f"{field}: {message}" if field else message)
        raise ValueError(f"{where}: {'; '.join(parts)}") from exc
