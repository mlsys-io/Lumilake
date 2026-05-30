"""Exception hierarchy for ``lumilake``."""


class LumilakeError(Exception):
    """Base class. Catch this to handle any SDK-side failure."""


class HttpError(LumilakeError):
    """Raised on non-2xx responses from the lumilake server.

    ``status`` is the HTTP code; ``body`` is the truncated response body.
    """

    def __init__(self, status: int, body: str, url: str = "") -> None:
        self.status = status
        self.body = body
        self.url = url
        suffix = f" url={url}" if url else ""
        super().__init__(f"HTTP {status}: {body[:300]}{suffix}")


class NotFoundError(HttpError):
    """404 — resource missing. Lets callers handle 'doesn't exist' separately."""


class ConfigNotFoundError(LumilakeError):
    """Saved ``~/.lumilake/config.toml`` is missing where one was expected."""


class ConfigInvalidError(LumilakeError):
    """``~/.lumilake/config.toml`` is unparsable or violates the schema."""


class DeployError(LumilakeError):
    """Raised when a deploy lifecycle call fails.

    ``action`` is the deploy verb (``up``, ``down``, ``logs``, …).
    ``exit_code`` is ``1`` when the underlying ``lumilake_deploy`` backend
    raises (translated at the resource boundary) and ``2`` for caller-side
    issues (missing ``[deploy]`` extra, unknown service name).
    ``stderr`` carries the backend message; preserved for diagnostics even
    though no subprocess is involved.
    """

    def __init__(self, action: str, exit_code: int, stderr: str) -> None:
        self.action = action
        self.exit_code = exit_code
        self.stderr = stderr
        super().__init__(
            f"deploy {action} failed (exit {exit_code}). " f"stderr: {stderr[-2000:]!r}"
        )
