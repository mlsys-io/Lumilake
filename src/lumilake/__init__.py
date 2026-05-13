"""Lumilake package root.

This module stays light: no eager imports of server / runtime / cli /
deploy code, no required environment variables. ``import lumilake``
succeeds with any combination of extras installed (or none). Each
capability lives under a subpackage gated on its own extra:

- ``lumilake.server``, ``lumilake.runtime``, ``lumilake.ops`` — ``[server]``
- ``lumilake.sdk`` — ``[sdk]``
- ``lumilake.cli`` — ``[cli]``
- ``lumilake.deploy`` — ``[deploy]``
"""

import logging

__version__ = "0.1.0"

# Quiet third-party loggers consumers don't care about. Safe to do at
# import time — adjusting levels has no side effects beyond logging.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
