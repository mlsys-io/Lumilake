import inspect
import logging
import os
from argparse import ArgumentParser, Namespace
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from types import ModuleType
from typing import Any

import psycopg
import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from lumilake import envs
from lumilake.log import configure_default_logger, init_child_logger
from psycopg_pool import AsyncConnectionPool

from lumilake_server.hooks import HookBindings, register
from lumilake_server.middleware import TraceIdMiddleware
from lumilake_server.routes import jobs, trace, workers
from lumilake_server.runtime.server import LumilakeServer, LumilakeServerConfig

# Loud-fail before any server side-effect runs: require .env on disk
# (unless LUMILAKE_SKIP_DOTENV_CHECK=1 for Docker injection) and verify
# every required env var is set with a valid value.
envs.load_env_file_or_raise()
envs.validate()

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(jobs.router)
api_router.include_router(trace.router)
api_router.include_router(workers.router)

origins = ["*"]


def parse_args() -> Namespace:
    parser = ArgumentParser()
    # nosec B104: envs.validate() at import requires HOST/PORT.
    parser.add_argument("--host", default=envs.LUMILAKE_SERVER_HOST)  # nosec B104
    parser.add_argument("--port", type=int, default=envs.LUMILAKE_SERVER_PORT)
    return parser.parse_args()


def _default_server_config() -> LumilakeServerConfig:
    args = parse_args()
    return LumilakeServerConfig(host=args.host, port=args.port)


def _import_plugin(plugin_name: str) -> ModuleType:
    module = __import__(plugin_name, fromlist=["install"])
    if not isinstance(module, ModuleType):
        raise TypeError(f"{plugin_name} did not resolve to a module")
    return module


async def _resolve_plugin_bindings(
    plugin_name: str,
    install: Any,
    stack: AsyncExitStack,
) -> HookBindings:
    if not callable(install):
        raise TypeError(f"{plugin_name}.install must be callable")
    installed = install()
    if isinstance(installed, AbstractAsyncContextManager):
        bindings = await stack.enter_async_context(installed)
    elif inspect.isawaitable(installed):
        bindings = await installed
    else:
        bindings = installed
    if not isinstance(bindings, HookBindings):
        raise TypeError(
            f"{plugin_name}.install() must return HookBindings, got "
            f"{type(bindings).__name__}"
        )
    return bindings


async def _load_plugins(stack: AsyncExitStack, logger: logging.Logger) -> None:
    """One bad plugin does not take down the server: import, validation, and
    ``install()`` failures are logged and skipped."""
    for plugin_name in envs.LUMILAKE_PLUGINS:
        try:
            module = _import_plugin(plugin_name)
        except Exception:
            logger.exception("Plugin %r failed to import; skipping.", plugin_name)
            continue
        install = module.__dict__.get("install")
        if install is None:
            logger.error("Plugin %r does not define install(); skipping.", plugin_name)
            continue
        try:
            bindings = await _resolve_plugin_bindings(plugin_name, install, stack)
        except Exception:
            logger.exception(
                "Plugin %r install() failed validation; skipping.", plugin_name
            )
            continue
        register(bindings)
        logger.info("Plugin %r registered.", plugin_name)


def build_app(config: LumilakeServerConfig | None = None) -> FastAPI:
    server_config = config or _default_server_config()
    logger = init_child_logger("Server")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        compute_pool: AsyncConnectionPool | None = None

        async with AsyncExitStack() as plugin_stack:
            await _load_plugins(plugin_stack, logger=logger)

            if envs.DATABASE_URL:
                compute_pool = AsyncConnectionPool(
                    conninfo=envs.DATABASE_URL,
                    min_size=2,
                    max_size=10,
                    open=False,
                )
                await compute_pool.open()
            else:
                logger.warning(
                    "DATABASE_URL not configured; DBLocation read/write disabled"
                )

            app.state.compute_db_pool = compute_pool

            try:
                with LumilakeServer.serve_instance(config=server_config):
                    yield
            finally:
                if compute_pool is not None:
                    await compute_pool.close()
                app.state.compute_db_pool = None

    app = FastAPI(
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        title="Lumilake Server",
        description="A data analytics engine built for LLM-based agentic workflows",
        version="0.1.0",
    )
    app.state.logger = logger
    app.state.compute_db_pool = None
    app.include_router(api_router)

    @app.exception_handler(psycopg.OperationalError)
    async def db_connection_error(_request: Request, exc: psycopg.OperationalError):
        logger.error("Database connection error: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"detail": "Database unavailable"},
        )

    @app.exception_handler(psycopg.DatabaseError)
    async def db_error(_request: Request, exc: psycopg.DatabaseError):
        logger.error("Database error: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"detail": "Database error"},
        )

    app.add_middleware(TraceIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "service": "lumilake-server"}

    @app.get("/docs", include_in_schema=False)
    async def api_documentation(request: Request):
        return HTMLResponse(
            """
            <!doctype html>
            <html lang="en">
            <head>
                <meta charset="utf-8">
                <meta name="viewport"
                      content="width=device-width, initial-scale=1, shrink-to-fit=no">
                <title>Lumilake API Documentation</title>

                <script src="https://unpkg.com/@stoplight/elements/web-components.min.js"></script>
                <link rel="stylesheet" href="https://unpkg.com/@stoplight/elements/styles.min.css">
            </head>
            <body>

                <elements-api
                apiDescriptionUrl="openapi.json"
                router="hash"
                />

            </body>
            </html>"""
        )

    return app


app = build_app()


def run_api_server() -> None:
    # The server container always wants structured JSON logs; flip the
    # default here so log aggregators in the deploy stack ingest records
    # without per-deployment config. Local dev (``pytest``, ``python -m``)
    # leaves the variable unset and gets the human-readable formatter.
    os.environ.setdefault("LUMILAKE_LOG_JSON", "1")
    configure_default_logger(json_logging=True)
    args = parse_args()
    server_config = LumilakeServerConfig(host=args.host, port=args.port)
    uvicorn.run(
        "lumilake_server.main:build_app",
        factory=True,
        host=server_config.host,
        port=server_config.port,
    )


if __name__ == "__main__":
    run_api_server()
