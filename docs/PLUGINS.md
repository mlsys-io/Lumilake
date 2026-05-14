# Hooks and Plugins

Lumilake uses `lumid-hooks` for shared extension points that can also be reused by other projects. Lumilake-specific extension points stay in this repository.

## Shared Hook Surfaces

| Hook | Purpose |
|------|---------|
| `IdentityProvider` | Resolve a request into a `PrincipalContext`. |
| `SubmissionGuard` | Allow or deny job submission before work is queued. |
| `PermissionChecker` | Authorize actions on resource kinds and resource IDs. |
| `ResourceRegistrar` | Register or deregister resource ownership/lifecycle records. |
| `UsageSink` | Receive usage rows emitted by the server. |

Lumilake resource kinds and actions live under `lumilake_hook` so hook implementations can avoid hard-coded strings.

## Lumilake-Owned Plugin Surface

Optimizer registration is Lumilake-specific. A plugin may register an optimizer implementation and select it with `LUMILAKE_OPTIMIZER_TYPE`.

The example plugin registers:

- shared hook implementations for identity, permissions, submission guards, registrar, and usage;
- a simple Lumilake optimizer plugin.

See `examples/plugins/simple_plugin/`.

## Enabling Plugins

Plugins are loaded inside the running server process (the Lumilake server image). The optimizer-registration and shared-hook surfaces — `lumilake_hook`, `lumilake_server.runtime.optimizer.*`, `lumilake_server.runtime.runtime_graph.*` — are reachable only from inside the image; they are intentionally not part of any published PyPI wheel. Develop plugins against a checked-out repository or against a custom Docker image that layers your plugin code on top of `ghcr.io/mlsys-io/lumilake_server`.

Point `PYTHONPATH` at the plugin package and set `LUMILAKE_PLUGINS`:

```bash
PYTHONPATH=./examples/plugins
LUMILAKE_PLUGINS=simple_plugin
```

Multiple plugins are comma-separated. Plugins should expose an `install()` function that wires their bindings or registrations.

`install()` may return `HookBindings` directly, return an awaitable that resolves to `HookBindings`, or be an async context manager that yields `HookBindings`. Context managers are entered during server startup and unwound during server shutdown.

## Design Rule

Shared hooks should contain only project-neutral contracts. Resource names, optimizer registration, and workflow/runtime-specific behavior should remain in Lumilake.
