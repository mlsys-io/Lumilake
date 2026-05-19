"""Command modules for the Lumilake CLI."""

from importlib.util import find_spec

import typer

from .base import app as base_app
from .job import app as job_app
from .trace import app as trace_app
from .worker import app as worker_app

# The ``[deploy]`` extra pulls ``lumilake-deploy`` and
# ``flowmesh-cli-stack``; gate on both up front so real ImportError
# from inside .deploy still bubbles up loudly.
_DEPLOY_AVAILABLE = (
    find_spec("lumilake_deploy") is not None
    and find_spec("flowmesh_cli_stack") is not None
)

if _DEPLOY_AVAILABLE:
    from .deploy import app as deploy_app
else:
    _DEPLOY_INSTALL_HINT = (
        "lumilake deploy requires the `deploy` extra. Install with "
        "`pip install 'lumilake[cli]'` or "
        "`pip install 'lumilake-cli[deploy]'`."
    )
    deploy_app = typer.Typer(
        help=(
            "Deploy and manage the Lumilake stack. Install with "
            "`pip install 'lumilake[cli]'` or "
            "`pip install 'lumilake-cli[deploy]'` to enable."
        )
    )

    def _deploy_hint() -> None:
        typer.echo(_DEPLOY_INSTALL_HINT, err=True)
        raise typer.Exit(code=2)

    @deploy_app.callback(invoke_without_command=True)
    def _deploy_no_subcommand(ctx: typer.Context) -> None:
        if ctx.resilient_parsing or ctx.invoked_subcommand is not None:
            return
        _deploy_hint()

    for _name in (
        "init",
        "doctor",
        "build",
        "pull",
        "up",
        "down",
        "clean",
        "purge",
        "reset",
        "status",
        "restart",
        "logs",
        "update-flowmesh",
    ):
        deploy_app.command(_name)(_deploy_hint)


def register(app: typer.Typer) -> None:
    """Register command groups on the root app."""
    app.add_typer(base_app)
    app.add_typer(deploy_app, name="deploy")
    app.add_typer(job_app, name="job")
    app.add_typer(worker_app, name="worker")
    app.add_typer(trace_app, name="trace")
