from collections.abc import Iterable


def extend_params(
    params: list[tuple[str, str]], key: str, values: str | bool | Iterable[str] | None
) -> None:
    if values is None:
        return
    if isinstance(values, bool):
        params.append((key, str(values).lower()))
        return
    if isinstance(values, str):
        params.append((key, values))
        return
    for value in values:
        params.append((key, value))
