import json

import typer

from ..core import logging
from ..core.http import HttpError, client_from_config
from ..core.typer import get_typer

app = get_typer(help="Query runtime workers.")


@app.command("list")
def list_workers() -> None:
    """List all available workers."""
    client = client_from_config()
    try:
        response = client.get("/workers", version_prefix=True)
    except HttpError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(json.dumps(response.json(), indent=2))


@app.command("get")
def get_worker(
    worker_id: str = typer.Argument(..., help="Worker identifier"),
) -> None:
    """Get details for a specific worker."""
    client = client_from_config()
    try:
        response = client.get(f"/workers/{worker_id}", version_prefix=True)
    except HttpError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(json.dumps(response.json(), indent=2))
