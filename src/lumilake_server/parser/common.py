"""Shared helpers used by both the n8n and YAML parsers.

Both parsers derive op ids through :func:`make_id` so that equivalent
workflows compile to byte-identical ``_id`` values for the same input,
regardless of process or interpreter. The suffix is a deterministic
:func:`hashlib.blake2b` digest (not the hash-randomized built-in
:func:`hash`).

Anything that is specific to the n8n wire format lives in
:mod:`lumilake_server.parser.n8n` — this module stays neutral.
"""

import hashlib
import re


def make_id(scope: str, prefix: str, name: str) -> str:
    """Derive a stable op id from ``(scope, prefix, name)``.

    Called by both :mod:`lumilake_server.parser.n8n` and
    :mod:`lumilake_server.parser.yaml_parser`. The slug components ensure
    the id stays readable; the digest suffix disambiguates
    ``(scope, name)`` pairs that slugify to the same string.

    The suffix is a 16-bit :func:`hashlib.blake2b` digest of
    ``"{scope}:{name}"``. It is deterministic across Python processes
    (unlike the built-in :func:`hash`, which is salted by
    ``PYTHONHASHSEED``), so the same workflow always compiles to the same
    op ids.
    """
    scope_slug = re.sub(r"[^A-Za-z0-9]+", "_", scope).strip("_") or "graph"
    name_slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "node"
    digest = hashlib.blake2b(f"{scope}:{name}".encode(), digest_size=2)
    suffix = int(digest.hexdigest(), 16) % 10000
    return f"{prefix}_{scope_slug}_{name_slug}_{suffix}"
