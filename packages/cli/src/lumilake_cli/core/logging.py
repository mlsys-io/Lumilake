"""Logging helpers for the CLI."""

import logging

import typer

LOG_LEVEL: int = logging.INFO


def log(
    message: str, level: int = logging.INFO, color: str | None = None, err: bool = False
) -> None:
    if level < LOG_LEVEL:
        return
    if color is None:
        typer.echo(message, err=err)
    else:
        typer.secho(message, err=err, fg=color)


def info(message: str) -> None:
    log(message, logging.INFO, typer.colors.BLUE, err=False)


def success(message: str) -> None:
    log(message, logging.INFO, typer.colors.GREEN, err=False)


def warning(message: str) -> None:
    log(message, logging.WARNING, typer.colors.YELLOW, err=False)


def error(message: str) -> None:
    log(message, logging.ERROR, typer.colors.RED, err=True)
