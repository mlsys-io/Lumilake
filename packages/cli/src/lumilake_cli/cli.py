"""Lumilake CLI entrypoint.

Installs a typer-aware handler on the shared ``Lumilake`` logger root so
records from every workspace package render with the CLI colour palette.
"""

import logging

import typer
from lumilake.log import ColorFormatter, get_default_logger

from .commands import register
from .core.typer import get_typer

root_app = get_typer(help="Lumilake command line interface.")


class _TyperHandler(logging.Handler):
    """Routes records through ``typer.echo``: stdout for INFO/DEBUG,
    stderr for WARNING/ERROR/CRITICAL. Honours typer's color detection.
    Coloring is left to ``ColorFormatter``.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            self.handleError(record)
            return
        typer.echo(message, err=record.levelno >= logging.WARNING)


def _install_handler() -> None:
    root = get_default_logger()
    if any(isinstance(h, _TyperHandler) for h in root.handlers):
        return
    # Drop get_default_logger's StreamHandler so records aren't doubled.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    handler = _TyperHandler()
    handler.setFormatter(ColorFormatter())
    root.addHandler(handler)
    root.propagate = False


def main() -> None:
    """Attach command groups and dispatch to Typer."""
    _install_handler()
    register(root_app)
    root_app()


if __name__ == "__main__":
    main()
