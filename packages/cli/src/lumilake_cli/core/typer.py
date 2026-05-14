from typing import Any

import typer


def get_typer(**kwargs: Any) -> typer.Typer:
    defaults: dict[str, Any] = {
        "context_settings": {"help_option_names": ["-h", "--help"]},
        "pretty_exceptions_show_locals": False,
    }
    defaults.update(kwargs)
    return typer.Typer(**defaults)
