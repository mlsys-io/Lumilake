import json
from typing import Any

import typer
from flowmesh.models.traces import ProfileSummary
from flowmesh.profile_views import (
    critical_path_dataframe,
    hardware_dataframe,
    network_dataframe,
    to_mermaid,
)
from rich.console import Console
from rich.table import Table

from ..core import logging
from ..core.http import HttpError, client_from_config
from ..core.typer import get_typer

app = get_typer(help="Query execution traces.")

OutputFormat = typer.Option(
    "summary",
    "--format",
    "-f",
    help="Output format: summary | json | mermaid",
)


@app.command("list")
def list_traces(
    limit: int = typer.Option(50, "--limit", "-n", help="Max traces to return (1-200)"),
) -> None:
    """List execution traces."""
    client = client_from_config()
    try:
        response = client.get(
            "/trace", version_prefix=True, params={"limit": str(limit)}
        )
    except HttpError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(json.dumps(response.json(), indent=2))


@app.command("get")
def get_trace(
    exec_id: str = typer.Argument(..., help="Execution trace identifier"),
    output_format: str = OutputFormat,
) -> None:
    """Retrieve and render a specific execution trace.

    Default ``summary`` shows the FlowMesh ``ProfileSummary`` as
    headline metrics + hardware / network / critical-path tables.
    ``json`` returns the raw payload; ``mermaid`` prints the lineage
    graph in Mermaid syntax.
    """
    client = client_from_config()
    try:
        response = client.get(f"/trace/{exec_id}", version_prefix=True)
    except HttpError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    payload = response.json()

    if output_format == "json":
        logging.log(json.dumps(payload, indent=2))
        return

    data = payload.get("data") or {}
    if data.get("status") != "ok" or data.get("trace") is None:
        logging.error(data.get("error") or "trace fetch returned no payload")
        raise typer.Exit(code=1)

    summary = ProfileSummary.model_validate(data["trace"])

    if output_format == "mermaid":
        logging.log(to_mermaid(summary))
        return

    if output_format != "summary":
        logging.error(f"unknown --format value: {output_format!r}")
        raise typer.Exit(code=1)

    _render_summary(summary)


def _render_summary(summary: ProfileSummary) -> None:
    console = Console()

    headline = Table(title="Trace summary", show_header=False, expand=False)
    headline.add_column("Metric", style="bold")
    headline.add_column("Value")
    headline.add_row("workflow_id", summary.workflow_id)
    headline.add_row("event_count", str(summary.event_count))
    headline.add_row("data_ids", str(len(summary.data_ids)))
    headline.add_row("assets", str(len(summary.assets)))
    headline.add_row("lineage edges", str(len(summary.lineage)))
    cp_nodes = (
        len(summary.critical_path.active_wait_breakdown.data_id)
        if summary.critical_path is not None
        else 0
    )
    headline.add_row("critical path nodes", str(cp_nodes))
    console.print(headline)

    _print_dataframe(
        console,
        "Hardware events (e2e)",
        hardware_dataframe(summary),
    )
    _print_dataframe(
        console,
        "Network events (e2e)",
        network_dataframe(summary),
    )
    _print_dataframe(
        console,
        "Critical path (active vs wait, seconds)",
        critical_path_dataframe(summary),
    )


def _print_dataframe(console: Console, title: str, frame: Any) -> None:
    if frame.empty:
        return
    table = Table(title=title)
    for column in frame.columns:
        table.add_column(str(column))
    for _, row in frame.iterrows():
        table.add_row(*(_fmt(value) for value in row.tolist()))
    console.print(table)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
