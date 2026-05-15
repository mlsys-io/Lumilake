"""Request-scoped middleware shared between the live server and tests."""

from fastapi import Request
from lumilake.log import trace_id_var
from starlette.middleware.base import BaseHTTPMiddleware

from lumilake_server.utils.utils import unique_id

REQUEST_ID_HEADER = "X-Request-ID"


class TraceIdMiddleware(BaseHTTPMiddleware):
    """Bind a per-request trace id to the log context and echo it on the response."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        incoming = request.headers.get(REQUEST_ID_HEADER, "").strip()
        trace_id = incoming or f"req-{unique_id()}"
        request.state.trace_id = trace_id
        token = trace_id_var.set(trace_id)
        try:
            response = await call_next(request)
        finally:
            trace_id_var.reset(token)
        response.headers[REQUEST_ID_HEADER] = trace_id
        return response
