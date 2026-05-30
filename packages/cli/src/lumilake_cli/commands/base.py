import json
from pathlib import Path

import typer
from lumilake._base_client import resolve_config
from lumilake.config import LumilakeConfig
from lumilake.errors import ConfigInvalidError, ConfigNotFoundError

from ..core import logging
from ..core.config import DEFAULT_CONFIG_PATH
from ..core.http import HttpError, client_from_config
from ..core.typer import get_typer

app = get_typer()


def _redact_api_key(
    api_key: str | None, prefix: int | None = None, suffix: int | None = None
) -> str | None:
    if api_key is None:
        return None
    length = len(api_key)
    if prefix is None and suffix is None:
        prefix = suffix = 4
        if length <= prefix + suffix:
            return "*" * length
    elif prefix is None or suffix is None:
        raise ValueError("Both prefix and suffix must be provided together")
    elif length <= prefix + suffix:
        raise ValueError(
            "API key is too short to redact with the given prefix/suffix lengths"
        )
    masked = "*" * (length - prefix - suffix)
    return f"{api_key[:prefix]}{masked}{api_key[-suffix:]}"


@app.command()
def init(
    url: str = typer.Argument(
        "http://127.0.0.1:9000",
        help="Lumilake server URL (e.g. http://127.0.0.1:9000).",
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", help="Bearer presented as Authorization: Bearer …"
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH, "--config", help="Path to save configuration file."
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing config file without confirmation.",
        show_default=False,
    ),
) -> None:
    """Initialize configuration for the Lumilake CLI."""
    if config_path.exists():
        if force:
            logging.warning(f"Overwriting existing config at {config_path}")
        elif not typer.confirm(f"Config file {config_path} already exists. Overwrite?"):
            raise typer.Exit()
    LumilakeConfig(base_url=url, api_key=api_key).save(config_path)
    logging.success(f"Config saved to {config_path}")


@app.command()
def deinit(
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH, "--config", help="Path to configuration file."
    ),
) -> None:
    """Delete the saved configuration file."""
    if config_path.exists():
        config_path.unlink()
        logging.success(f"Deleted config file {config_path}")
    else:
        logging.warning(f"No config file found at {config_path}")


@app.command()
def config(
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH, "--config", help="Path to configuration file."
    ),
    source: str = typer.Option(
        "auto",
        "--source",
        help="Configuration source: auto, file, or env.",
    ),
    show_api_key: bool = typer.Option(
        False, "--show-api-key", help="Show api_key in plain text."
    ),
) -> None:
    """Display the resolved Lumilake CLI configuration."""
    normalized = source.strip().lower()
    match normalized:
        case "file":
            try:
                cfg = LumilakeConfig.from_file(config_path)
            except (ConfigNotFoundError, ConfigInvalidError) as exc:
                logging.error(f"Error loading config from file: {exc}")
                raise typer.Exit(code=1)
        case "env":
            try:
                cfg = LumilakeConfig.from_env()
            except ConfigInvalidError as exc:
                logging.error(f"Error loading config from environment: {exc}")
                raise typer.Exit(code=1)
        case "auto":
            try:
                cfg = resolve_config(config_path=config_path)
            except Exception as exc:
                logging.error(f"Error resolving config: {exc}")
                raise typer.Exit(code=1)
        case _:
            logging.error("Invalid --source value. Expected one of: auto, file, env")
            raise typer.Exit(code=2)

    cfg.api_key = cfg.api_key if show_api_key else _redact_api_key(cfg.api_key)
    logging.log(json.dumps(cfg.to_mapping(), indent=2))


@app.command()
def info() -> None:
    """Query server health status."""
    client = client_from_config()
    try:
        response = client.get("/healthz")
    except HttpError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(json.dumps(response.json(), indent=2))


@app.command()
def health() -> None:
    """Check if the Lumilake server is reachable and healthy."""
    info()
