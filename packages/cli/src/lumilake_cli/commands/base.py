import json
from pathlib import Path

import typer

from ..core import logging
from ..core.config import DEFAULT_CONFIG_PATH
from ..core.http import HttpError, _resolve_base_url, client_from_config
from ..core.typer import get_typer

app = get_typer()


@app.command()
def config(
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH, "--config", help="Path to configuration file"
    ),
) -> None:
    """Show the resolved Lumilake CLI configuration."""
    base_url, source = _resolve_base_url(config_path)
    payload = {"base_url": base_url, "source": source, "config_path": str(config_path)}
    logging.log(json.dumps(payload, indent=2))


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
