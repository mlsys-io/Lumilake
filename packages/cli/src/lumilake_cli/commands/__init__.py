"""Command modules for the Lumilake CLI."""

import typer

from .base import app as base_app
from .deploy import app as deploy_app
from .job import app as job_app
from .trace import app as trace_app
from .worker import app as worker_app


def register(app: typer.Typer) -> None:
    """Register command groups on the root app."""
    app.add_typer(base_app)
    app.add_typer(deploy_app, name="deploy")
    app.add_typer(job_app, name="job")
    app.add_typer(worker_app, name="worker")
    app.add_typer(trace_app, name="trace")
