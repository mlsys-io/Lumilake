"""
Adapted from https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output.
"""

import inspect
import json
import logging
import os
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from lumilake import envs

Logger = logging.Logger
LogLevel = int | str

_default_logger: Logger | None = None

# Request-scoped trace id; set by the server middleware (and any other entry
# point that wants its log lines tagged) and read by ``TraceIdFilter`` below.
# Defaults to an empty string so log lines outside a request still format
# cleanly.
trace_id_var: ContextVar[str] = ContextVar("lumilake_trace_id", default="")


def set_trace_id(trace_id: str | None) -> Any:
    """Set the current trace id; returns the token to restore later.

    Pass ``None`` to clear. Use with ``trace_id_var.reset(token)`` in a
    ``finally`` block, or inside a ``contextvars.copy_context()``-scoped task.
    """
    return trace_id_var.set(trace_id or "")


class TraceIdFilter(logging.Filter):
    """Inject the current ``trace_id`` context var into every ``LogRecord``.

    Attached at the handler level. ``record.trace_id`` is always set (empty
    string when no trace is active) so format strings and structured
    formatters can reference it unconditionally.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = trace_id_var.get()
        return True


class ColorFormatter(logging.Formatter):
    """
    See the list of bash colors here:
    https://gist.github.com/JBlond/2fea43a3049b38287e5e9cefc87b2124
    """

    blue: str = "\x1b[0;34m"
    green: str = "\x1b[0;32m"
    yellow: str = "\x1b[0;33m"
    red: str = "\x1b[0;31m"
    bold_red: str = "\x1b[1;31m"
    reset: str = "\x1b[0m"
    format_str: str = "[%(name)s | %(levelname)s] %(message)s"

    FORMATS: dict[int, str] = {
        logging.DEBUG: blue + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class JsonFormatter(logging.Formatter):
    """Structured JSON formatter for the server container.

    Emits one JSON object per record with stable top-level keys
    (``time``, ``level``, ``logger``, ``message``, ``trace_id``) and
    flattens any extra fields the caller passed via ``extra={...}``. Used
    when ``LUMILAKE_LOG_JSON=1`` (set by the server entrypoint) so log
    aggregators can index records out of the box. Pytest / local dev use
    ``ColorFormatter`` so output stays human-readable.
    """

    _RESERVED: frozenset[str] = frozenset(
        {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "asctime",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        trace_id = getattr(record, "trace_id", "") or ""
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": trace_id,
        }
        for key, value in record.__dict__.items():
            if key in self._RESERVED or key == "trace_id":
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                value = repr(value)
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, default=str)


def _use_json_logging() -> bool:
    """``LUMILAKE_LOG_JSON=1`` opts the default logger into JSON output.

    Set by the server container entrypoint; unset (or ``0``/``false``) for
    pytest and local dev so existing tests / interactive logs stay
    human-readable.
    """
    raw = os.environ.get("LUMILAKE_LOG_JSON", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def get_default_logger() -> Logger:
    global _default_logger

    if _default_logger is not None:
        return _default_logger

    log_level = envs.LUMILAKE_LOG_LEVEL

    logger = logging.getLogger("Lumilake")
    logger.setLevel(log_level)

    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    if _use_json_logging():
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(ColorFormatter())
    handler.addFilter(TraceIdFilter())

    logger.addHandler(handler)

    logger.propagate = False

    _default_logger = logger
    return logger


def init_child_logger(
    name: str, logger: Logger | None = None, log_level: LogLevel | None = None
) -> Logger:
    if logger is None:
        logger = get_default_logger()
    logger = logger.getChild(name)
    if log_level is not None:
        logger.setLevel(log_level)
    return logger


def log_on_exception_async(
    ignore: list[type[BaseException]] | None = None, is_method: bool = True
) -> Callable:
    def method_decorator(func):
        async def wrapper(self, *args, **kwargs) -> Any:
            try:
                return await func(self, *args, **kwargs)
            except BaseException as e:
                logger: Logger | None = getattr(self, "logger", None)
                if logger is None or not isinstance(logger, Logger):
                    return
                if ignore and any(isinstance(e, exc) for exc in ignore):
                    return
                logger.exception(
                    "Exception '%s' occurred in function '%s' (called by '%s')"
                    " (Detail: %s)",
                    e.__class__.__name__,
                    func.__name__,
                    inspect.stack()[1].function,
                    e,
                )
                raise

        return wrapper

    def decorator(func):
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except BaseException as e:
                logger = get_default_logger()
                logger.exception(
                    "Exception '%s' occurred in function '%s' (called by '%s')"
                    " (Detail: %s)",
                    e.__class__.__name__,
                    func.__name__,
                    inspect.stack()[1].function,
                    e,
                )
                raise

        return wrapper

    return method_decorator if is_method else decorator
