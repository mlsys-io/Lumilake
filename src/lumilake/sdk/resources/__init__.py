"""Resource clients — one sync class + one async class per CLI command group.

Server-API resources (info, jobs, workers, traces) talk HTTP to the
lumilake server.

``Deploy`` / ``AsyncDeploy`` call ``lumilake.deploy`` directly; async
dispatches through ``asyncio.to_thread`` so the event loop stays responsive.
"""
