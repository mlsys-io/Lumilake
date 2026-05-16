"""Cover the structured-logging path in ``lumilake.log``."""

import json
import logging
from io import StringIO

from lumilake.log import (
    JsonFormatter,
    TraceIdFilter,
    set_trace_id,
    trace_id_var,
)


def test_trace_id_filter_attaches_current_context_var() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    token = set_trace_id("abc")
    try:
        TraceIdFilter().filter(record)
    finally:
        trace_id_var.reset(token)
    assert getattr(record, "trace_id") == "abc"


def test_trace_id_filter_defaults_to_empty_string() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    TraceIdFilter().filter(record)
    assert getattr(record, "trace_id") == ""


def test_json_formatter_emits_structured_record_with_trace_id() -> None:
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(TraceIdFilter())

    logger = logging.getLogger("test.json_formatter")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    token = set_trace_id("trace-xyz")
    try:
        logger.info("structured %s", "msg", extra={"job_id": "req-1"})
    finally:
        trace_id_var.reset(token)
        logger.handlers.clear()

    payload = json.loads(stream.getvalue().strip())
    assert payload["message"] == "structured msg"
    assert payload["level"] == "INFO"
    assert payload["trace_id"] == "trace-xyz"
    assert payload["logger"] == "test.json_formatter"
    assert payload["job_id"] == "req-1"
