"""Lumilake server runtime — image-only, not published to PyPI."""

import logging

__version__ = "0.1.4"

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
