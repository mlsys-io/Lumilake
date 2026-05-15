import json

import typer

from ..core import logging
from ..core.http import HttpError, client_from_config
from ..core.typer import get_typer

app = get_typer(help="Query runtime workers.")


_JSON_OPTION_HELP = (
    "Emit the server's JSON envelope verbatim — currently the default "
    "and the stable parse target for scripts."
)


def _emit_json(payload: object) -> None:
    logging.log(json.dumps(payload, indent=2))


@app.command("list")
def list_workers(
    json_output: bool = typer.Option(False, "--json", help=_JSON_OPTION_HELP),
) -> None:
    """List all available workers."""
    client = client_from_config()
    try:
        response = client.get("/workers", version_prefix=True)
    except HttpError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    _emit_json(response.json())


@app.command("get")
def get_worker(
    worker_id: str = typer.Argument(..., help="Worker identifier"),
    json_output: bool = typer.Option(False, "--json", help=_JSON_OPTION_HELP),
) -> None:
    """Get details for a specific worker."""
    client = client_from_config()
    try:
        response = client.get(f"/workers/{worker_id}", version_prefix=True)
    except HttpError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    _emit_json(response.json())
