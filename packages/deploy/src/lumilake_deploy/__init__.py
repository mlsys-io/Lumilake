"""Lumilake deploy orchestration implementation.

Public entry points called from ``cli/commands/deploy.py``:

- :func:`setup.run_setup` — stand up the stack from ``.env``.
- :func:`stop.run_stop` — stop all services (optionally purge volumes).
- :func:`update_flowmesh.run_update` — re-lock + install latest FlowMesh packages.

External binaries (``docker``, ``uv``) are invoked via subprocess; the
Python layer owns the control flow, parsing, and state.
"""
