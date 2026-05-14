import json
from pathlib import Path

import typer

from ..core import logging
from ..core.config import DEFAULT_CONFIG_PATH, LumilakeConfig, save_config
from ..core.http import HttpClient, HttpError, client_from_config
from ..core.typer import get_typer

app = get_typer()


@app.command()
def login(
    url: str = typer.Argument(
        ..., help="Lumilake server URL (e.g., http://localhost:9000)"
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH, "--config", help="Path to save configuration file"
    ),
) -> None:
    """Configure connection to a Lumilake server and save it locally."""
    client = HttpClient(base_url=url)
    try:
        client.get("/healthz")
    except HttpError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    save_config(LumilakeConfig(base_url=url), path=config_path)
    logging.success(f"Login successful. Saved config to {config_path}")


@app.command()
def logout(
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH, "--config", help="Path to configuration file to remove"
    ),
) -> None:
    """Delete saved configuration."""
    config_path.unlink(missing_ok=True)
    logging.success(f"Logout successful. Removed config {config_path}")


@app.command()
def config(
    refresh: bool = typer.Option(
        False, "--refresh", "-r", help="Reload and show the latest configuration"
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH, "--config", help="Path to configuration file"
    ),
) -> None:
    """Show the current Lumilake CLI configuration."""
    from ..core.http import _require_config

    cfg = _require_config(config_path)
    if refresh:
        client = HttpClient(base_url=cfg.base_url)
        try:
            client.get("/healthz")
        except HttpError as exc:
            logging.error(str(exc))
            raise typer.Exit(code=1)
        save_config(cfg, path=config_path)
        logging.success(f"Configuration refreshed. Saved to {config_path}")
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
